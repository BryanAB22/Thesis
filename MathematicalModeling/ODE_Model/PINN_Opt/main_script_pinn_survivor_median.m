%% main_script_pinn_survivor_median.m
% TRUE PHYSICS-INFORMED NEURAL NETWORK (PINN) FITTING
%
% This replaces Differential Evolution with a true PINN.
% The neural network y_theta(t) learns the 17 state trajectories directly.
% The ODE parameters selected by fit_idx and PE(0) are trainable variables.
%
% The loss is:
%   total loss = data loss + ODE residual loss + small parameter regularization
% State nonnegativity is enforced directly by the PINN output transformation.
%
% Important difference from DE:
%   DE repeatedly calls ode23s.
%   This PINN does NOT call ode23s during training.
%   Instead, automatic differentiation computes dy_theta/dt and penalizes:
%       dy_theta/dt - model_rhs(y_theta, fitted_parameters) = 0
%
% Required in same folder:
%   Survivor_Data.csv
%   load_pars_Init_Copeland_Edited.m
%
% Recommended MATLAB toolbox:
%   Deep Learning Toolbox

close all;
clear;
clc;
totalTimer = tic;

csvFile = 'Survivor_Data.csv';
matFile = 'fit_pinn_results_positive_fastgate.mat';

usePreviousFit = true;
previousMatFile = 'fit_pinn_results_positive_fastgate.mat';

% When a previous MAT file exists, continue from the saved PINN weights,
% fitted ODE parameters, fitted initial condition, and state scaling.
oldFit = struct();
continuingTraining = false;
previousFittedPars = [];
previousFittedInit = [];

runningTime = 700;

% Use a denser collocation grid during the first 24 hours, where the
% cytokine peaks and rapid physiological changes occur.
earlyCollocationEnd = 24;
numEarlyCollocationPoints = 200;
numLateCollocationPoints = 200;

tEarly = linspace(0, earlyCollocationEnd, numEarlyCollocationPoints);
tLate  = linspace(earlyCollocationEnd, runningTime, numLateCollocationPoints);
tColloc = unique([tEarly, tLate], 'sorted');
numCollocationPoints = numel(tColloc);

% Training settings.
numEpochs = 150000;
learnRateNet  = 1e-5;
learnRatePars = 5e-6;

miniBatchCollocation = true;   

% Neural network size.
numHiddenUnits = 96;
numHiddenLayers = 5;

perturbPercent = 80;

% Loss weights. If the PINN follows the ODE but misses the data, increase lambdaData.
% If it fits the data but violates the model physics, increase lambdaODE.
lambdaData = 5000;
lambdaODE  = 1e-4;
lambdaNonNegative = 0;
lambdaParamReg = 1e-6;

% Ramp the ODE loss up slowly so the NN first learns the rough data shape.
% After odeRampEpochs, the full ODE penalty is active.
odeRampEpochs = 8000;

% Print every N epochs.
printEvery = 100;

% Early stopping based on original-data SSE.
% Training stops if OriginalDataSSE does not improve for this many epochs.
earlyStopPatience = 1000;

% Minimum SSE decrease required to count as an improvement.
% Use 0 for any decrease. Use 1 or 10 if tiny numerical changes keep resetting patience.
minSSEImprovement = .1;

% Optional GPU. Keep false unless your MATLAB GPU setup works.
useGPU = true;

rng(1);

%% ================= FILE SETUP =================
% These helpers make the uploaded files work even if ChatGPT/your browser
% renamed them with (1), (2), etc. In your actual folder, normal names are preferred.
if ~isfile(csvFile)
    candidates = dir('Survivor_Data*.csv');
    if ~isempty(candidates)
        csvFile = candidates(1).name;
    else
        error('Cannot find Survivor_Data.csv.');
    end
end

ensureExpectedFunctionFile('load_pars_Init_Copeland_Edited.m', 'load_pars_Init_Copeland_Edited*.m');

%% ================= LOAD DATA =================
T = readSurvivorTable(csvFile);
timeAll = T.Time;

HR     = T.HR_Mean;
TEMP   = T.TEMP_Mean;
TNF    = T.TNF_Mean;
IL6    = T.IL_6_Mean;
IL8    = T.IL_8_Mean;

data = struct();
data.hr   = HR;
data.TNF  = TNF;
data.IL6  = IL6;
data.IL8  = IL8;
data.temp = TEMP;

%% ================= LOAD BASELINE PARAMETERS AND INITIAL CONDITIONS =================
global ODE_TOL
ODE_TOL = 1e-8;

[base_pars, base_Init] = load_pars_Init_Copeland_Edited(data);
base_pars = base_pars(:);
base_Init = base_Init(:);

if usePreviousFit && isfile(previousMatFile)
    oldFit = load(previousMatFile);

    requiredFields = {'trainedNet','new_pars','new_Init','base_pars','base_Init'};
    for kField = 1:numel(requiredFields)
        if ~isfield(oldFit, requiredFields{kField})
            error(['Cannot resume training because %s does not contain the variable "%s". ' ...
                   'Use a MAT file saved by this PINN script.'], ...
                   previousMatFile, requiredFields{kField});
        end
    end

    % Keep the SAME parameter coordinate system and bounds used by the
    % saved run. The current fitted values are restored later through zPar
    % and zInit. This makes the restart a continuation instead of a new fit
    % centered around a different parameter baseline.
    base_pars = oldFit.base_pars(:);
    base_Init = oldFit.base_Init(:);
    previousFittedPars = oldFit.new_pars(:);
    previousFittedInit = oldFit.new_Init(:);
    continuingTraining = true;

    if isfield(oldFit, 'perturbPercent')
        if oldFit.perturbPercent ~= perturbPercent
            fprintf(['\nUsing saved perturbPercent = %.6g instead of the current value %.6g ' ...
                     'to preserve the previous parameter bounds.\n'], ...
                     oldFit.perturbPercent, perturbPercent);
        end
        perturbPercent = oldFit.perturbPercent;
    end

    fprintf('\nResuming PINN training from %s.\n', previousMatFile);
    fprintf('Reloading the saved trainedNet, fitted parameters, and fitted initial condition.\n');
elseif usePreviousFit
    fprintf('\nPrevious MAT file not found. Starting a new PINN training run.\n');
end

numStates = numel(base_Init);
numPars = numel(base_pars);

if numStates ~= 17
    error('Expected 17 initial conditions/states, but found %d.', numStates);
end
if numPars ~= 114
    error('Expected 114 parameters, but found %d.', numPars);
end

par_names = makeParameterNames();


fit_idx = [
    34;   %  1 hmpe
    76;   %  2 knom
    9;    %  3 kma
    91;   %  4 kpm
    98;   %  5 alpha
%     97;   %  6 xdn
%     100;  %  7 sM
%     93;   %  8 muno
%     21;   %  9 xmpe
    62;   % 10 xht
%     39;   % 11 sm
    26;   % 12 h66
%     107;  % 13 hmD
    % 14;   % 14 x66
%     102;  % 15 kD
%     20;   % 16 xtnf6
%     81;   % 17 hn10
%     95;   % 18 kdn
    7;    % 19 ktnf
%     96;   % 20 mud
%     77;   % 21 kno
%     63;   % 22 hht
    89;   % 23 kpg
%     31;   % 24 htnf6
    8;    % 25 ktnfm
%     32;   % 26 hm10
%     36;   % 27 s10
    48;   % 28 kt
%     2;    % 29 k10m
%     78;   % 30 xntnf
%     % 79;   % 31 xn10
%     % 1;    % 32 k10
%     % 43;   % 33 k106
    61;   % 34 kh
%     % 3;    % 35 k6
%     % 19;   % 36 xtnf10
%     % 45;   % 37 tau1
%     % 4;    % 38 k6m
%     % 94;   % 39 kpn
%     % 22;   % 40 xm10
%     % 80;   % 41 hntnf
    % 49;   % 42 kttnf
%     % 87;   % 43 xhp
%     % 41;   % 44 k6tnf
    5;    % 45 k8
    113;  % 46 ktnfhr
    % 42;   % 47 k8tnf
%     % 101;  % 48 kmp
%     % 64;   % 49 ppM
    % 50;   % 50 kt6
];
% fit_idx = [34; 76; 9; 91; 21; 62; 39; 26; 14; 102; 20; 95; 7; 63; 89; 8; ...
%            48; 2; 1; 43; 61; 3; 45; 4; 49; 87; 41; 5; 113; 42; 50];

fit_idx = fit_idx(:);


% Fit only PE(0), same as your DE setup.
init_idx = 7;
init_names = {'PE_initial'};

if continuingTraining
    if isfield(oldFit, 'fit_idx') && ~isequal(oldFit.fit_idx(:), fit_idx(:))
        error(['The current fit_idx does not match the fit_idx stored in %s. ' ...
               'Use the same fitted-parameter list when resuming the saved PINN.'], ...
               previousMatFile);
    end
    if isfield(oldFit, 'init_idx') && ~isequal(oldFit.init_idx(:), init_idx(:))
        error(['The current init_idx does not match the init_idx stored in %s. ' ...
               'Use the same fitted initial conditions when resuming.'], ...
               previousMatFile);
    end
end

fprintf('\n================ TRUE PINN SETUP ================\n');
fprintf('CSV file: %s\n', csvFile);
fprintf('Fitted parameters: %d\n', numel(fit_idx));
fprintf('Fitted initial conditions: %d\n', numel(init_idx));
fprintf('Collocation points: %d\n', numCollocationPoints);
fprintf('Epochs: %d\n', numEpochs);

displayStartingPars = base_pars;
displayStartingInit = base_Init;
if continuingTraining
    displayStartingPars = previousFittedPars;
    displayStartingInit = previousFittedInit;
end

fitTable = table((1:numel(fit_idx))', fit_idx, par_names(fit_idx)', displayStartingPars(fit_idx), ...
    'VariableNames', {'FitNumber','Index','Parameter','StartingValue'});
disp('---------------- PARAMETERS BEING FIT ----------------');
disp(fitTable);

initFitTable = table(init_idx(:), init_names(:), displayStartingInit(init_idx), ...
    'VariableNames', {'Index','InitialCondition','StartingValue'});
disp('---------------- INITIAL CONDITIONS BEING FIT ----------------');
disp(initFitTable);

%% ================= OBSERVED DATA STRUCTURE =================
% State mapping:
% 1 = TNF, 3 = IL-8/CXCL8, 4 = IL-6, 8 = Temperature, 14 = HR.
% Initialize obs with the same fields returned by makeObs.
% Do NOT use obs = struct([]), because MATLAB treats that as an empty
% structure with no fields, causing:
%   Subscripted assignment between dissimilar structures.
obs = struct('name', {}, 'state', {}, 't', {}, 'y', {}, ...
             'tNorm', {}, 'yDL', {}, 'scale', {});

obs(end+1) = makeObs('TNF',         1,  timeAll, TNF);
obs(end+1) = makeObs('IL-8/CXCL8',  3,  timeAll, IL8);
obs(end+1) = makeObs('IL-6',        4,  timeAll, IL6);
obs(end+1) = makeObs('Temperature', 8,  timeAll, TEMP);
obs(end+1) = makeObs('HR',          14, timeAll, HR);

%% ================= SCALING FOR STABLE PINN TRAINING =================
% These scales do NOT change the ODE. They only make the neural-network
% training numerically stable because states have very different units.
if continuingTraining && isfield(oldFit, 'stateScale')
    stateScale = oldFit.stateScale(:);
    if numel(stateScale) ~= numStates
        error('Saved stateScale has %d entries; expected %d.', numel(stateScale), numStates);
    end

    if isfield(oldFit, 'derivScale')
        derivScale = oldFit.derivScale(:);
    else
        derivScale = max(stateScale ./ runningTime, 1e-6);
    end

    fprintf('Reloaded the saved state and derivative scaling.\n');
else
    stateScale = max(abs(base_Init), 1);
    for k = 1:numel(obs)
        s = obs(k).state;
        y = obs(k).y;
        if any(isfinite(y))
            stateScale(s) = max([stateScale(s); abs(y(isfinite(y)))]);
        end
    end
    stateScale = max(stateScale(:), 1);
    derivScale = max(stateScale ./ runningTime, 1e-6);
end

%% ================= ADDITIVE BOUNDS FOR TRAINABLE PARAMETERS =================
if perturbPercent > 1
    perturbFraction = perturbPercent/100;
else
    perturbFraction = perturbPercent;
end
perturbFraction = abs(perturbFraction);

startingFitValues = [base_pars(fit_idx); base_Init(init_idx)];
deltaMax = perturbFraction .* abs(startingFitValues);
zeroMask = deltaMax == 0;
deltaMax(zeroMask) = perturbFraction;

lowerDelta = -deltaMax;
upperDelta =  deltaMax;

nonnegativeStart = startingFitValues >= 0;
lowerDelta(nonnegativeStart) = max(lowerDelta(nonnegativeStart), -0.999999*startingFitValues(nonnegativeStart));

nfitPar  = numel(fit_idx);
nfitInit = numel(init_idx);

lowerParDelta = lowerDelta(1:nfitPar);
upperParDelta = upperDelta(1:nfitPar);
lowerInitDelta = lowerDelta(nfitPar+1:end);
upperInitDelta = upperDelta(nfitPar+1:end);

% Basis matrices allow us to build full parameter/IC vectors differentiably.
BPar = zeros(numPars, nfitPar);
for j = 1:nfitPar
    BPar(fit_idx(j), j) = 1;
end
BInit = zeros(numStates, nfitInit);
for j = 1:nfitInit
    BInit(init_idx(j), j) = 1;
end

%% ================= BUILD NEURAL NETWORK =================
layers = [featureInputLayer(1, 'Normalization','none', 'Name','time')];
for j = 1:numHiddenLayers
    layers = [layers
        fullyConnectedLayer(numHiddenUnits, 'Name', sprintf('fc%d', j))
        functionLayer(@tanh, 'Name', sprintf('tanh%d', j), 'Formattable', true)]; %#ok<AGROW>
end
layers = [layers
    fullyConnectedLayer(numStates, 'Name','state_output')];

templateNet = dlnetwork(layers);

if continuingTraining
    validateNetworkCompatibility(oldFit.trainedNet, templateNet);
    net = oldFit.trainedNet;
    fprintf('Reloaded trainedNet. The neural-network weights will continue from the saved fit.\n');
else
    net = templateNet;
    fprintf('Created a new randomly initialized PINN.\n');
end

%% ================= CREATE DLARRAY TRAINING VARIABLES =================
% zPar and zInit are unconstrained trainable variables.
% They are mapped through a sigmoid into the additive delta bounds.
if continuingTraining
    previousDeltaPar = previousFittedPars(fit_idx) - base_pars(fit_idx);
    previousDeltaInit = previousFittedInit(init_idx) - base_Init(init_idx);

    tolPar = 1e-10 * max(1, max(abs([lowerParDelta; upperParDelta])));
    tolInit = 1e-10 * max(1, max(abs([lowerInitDelta; upperInitDelta])));

    if any(previousDeltaPar < lowerParDelta - tolPar) || ...
       any(previousDeltaPar > upperParDelta + tolPar)
        error('The saved fitted parameters fall outside the reconstructed parameter bounds.');
    end
    if any(previousDeltaInit < lowerInitDelta - tolInit) || ...
       any(previousDeltaInit > upperInitDelta + tolInit)
        error('The saved fitted initial condition falls outside the reconstructed bounds.');
    end

    % Reconstruct the unconstrained variables that produced the previous
    % fitted values. This preserves the actual ODE parameter starting point.
    zPar0 = boundedZFromDelta(previousDeltaPar, lowerParDelta, upperParDelta);
    zInit0 = boundedZFromDelta(previousDeltaInit, lowerInitDelta, upperInitDelta);
else
    zPar0 = boundedZFromDelta(zeros(nfitPar,1), lowerParDelta, upperParDelta);
    zInit0 = boundedZFromDelta(zeros(nfitInit,1), lowerInitDelta, upperInitDelta);
end

base_pars_DL = dlarray(base_pars);
base_Init_DL = dlarray(base_Init);
BPar_DL = dlarray(BPar);
BInit_DL = dlarray(BInit);
stateScale_DL = dlarray(stateScale);
derivScale_DL = dlarray(derivScale);
lowerParDelta_DL = dlarray(lowerParDelta);
upperParDelta_DL = dlarray(upperParDelta);
lowerInitDelta_DL = dlarray(lowerInitDelta);
upperInitDelta_DL = dlarray(upperInitDelta);
zPar = dlarray(zPar0);
zInit = dlarray(zInit0);

% Collocation time points where the ODE residual is enforced.
% tColloc was created above with extra resolution during the first 24 hours.
tCollocNorm = dlarray(tColloc ./ runningTime, 'CB');

for k = 1:numel(obs)
    obs(k).tNorm = dlarray(obs(k).t(:)' ./ runningTime, 'CB');
    obs(k).yDL = dlarray(obs(k).y(:)', 'CB');
    obs(k).scale = stateScale(obs(k).state);
end

if useGPU && exist('canUseGPU','file') && canUseGPU
    net = dlupdate(@gpuArray, net);
    base_pars_DL = gpuArray(base_pars_DL);
    base_Init_DL = gpuArray(base_Init_DL);
    BPar_DL = gpuArray(BPar_DL);
    BInit_DL = gpuArray(BInit_DL);
    stateScale_DL = gpuArray(stateScale_DL);
    derivScale_DL = gpuArray(derivScale_DL);
    lowerParDelta_DL = gpuArray(lowerParDelta_DL);
    upperParDelta_DL = gpuArray(upperParDelta_DL);
    lowerInitDelta_DL = gpuArray(lowerInitDelta_DL);
    upperInitDelta_DL = gpuArray(upperInitDelta_DL);
    zPar = gpuArray(zPar);
    zInit = gpuArray(zInit);
    tCollocNorm = gpuArray(tCollocNorm);
    for k = 1:numel(obs)
        obs(k).tNorm = gpuArray(obs(k).tNorm);
        obs(k).yDL = gpuArray(obs(k).yDL);
    end
    fprintf('Training on GPU.\n');
else
    net = dlupdate(@gather, net);
    fprintf('Training on CPU.\n');
end

%% ================= TRAIN PINN =================
trailingAvgNet = [];
trailingAvgSqNet = [];
trailingAvgPar = [];
trailingAvgSqPar = [];
trailingAvgInit = [];
trailingAvgSqInit = [];

history = table([], [], [], [], [], [], [], ...
    'VariableNames', {'Epoch','TotalLoss','DataLoss','ODELoss','NonnegativeLoss','ParamRegLoss','OriginalDataSSE'});

% Track the best epoch using the actual original-data SSE.
% The final saved fit will be restored to this best epoch, not simply the last epoch.
bestOriginalDataSSE = inf;
bestEpoch = 0;
epochsSinceBest = 0;
bestNet = net;
bestZPar = zPar;
bestZInit = zInit;
bestLoss = inf;
bestInfo = [];
stoppedEarly = false;
finalEpochTrained = 0;

fprintf('\n================ TRAINING TRUE PINN ================\n');
for epoch = 1:numEpochs
    finalEpochTrained = epoch;
    if continuingTraining
        % The saved PINN already completed its warm-up stage.
        lambdaODERamped = lambdaODE;
    else
        lambdaODERamped = lambdaODE * min(1, epoch / odeRampEpochs);
    end

    [loss, gradientsNet, gradientsPar, gradientsInit, info] = dlfeval(@modelGradientsPINN, ...
        net, zPar, zInit, ...
        base_pars_DL, base_Init_DL, BPar_DL, BInit_DL, ...
        lowerParDelta_DL, upperParDelta_DL, lowerInitDelta_DL, upperInitDelta_DL, ...
        stateScale_DL, derivScale_DL, tCollocNorm, obs, runningTime, ...
        lambdaData, lambdaODERamped, lambdaNonNegative, lambdaParamReg);

    currentOriginalDataSSE = gather(extractdata(info.originalDataSSE));

    % Save the best model/parameters before the Adam update, because info
    % corresponds to the current net, zPar, and zInit.
    if currentOriginalDataSSE < bestOriginalDataSSE - minSSEImprovement
        bestOriginalDataSSE = currentOriginalDataSSE;
        bestEpoch = epoch;
        epochsSinceBest = 0;

        bestNet = net;
        bestZPar = zPar;
        bestZInit = zInit;
        bestLoss = gather(extractdata(loss));

        bestInfo = struct();
        bestInfo.totalLoss = gather(extractdata(loss));
        bestInfo.dataLoss = gather(extractdata(info.dataLoss));
        bestInfo.odeLoss = gather(extractdata(info.odeLoss));
        bestInfo.nonnegativeLoss = gather(extractdata(info.nonnegativeLoss));
        bestInfo.paramRegLoss = gather(extractdata(info.paramRegLoss));
        bestInfo.originalDataSSE = currentOriginalDataSSE;
        bestInfo.lambdaODE = lambdaODERamped;
    else
        epochsSinceBest = epochsSinceBest + 1;
    end

    [net, trailingAvgNet, trailingAvgSqNet] = adamupdate(net, gradientsNet, ...
        trailingAvgNet, trailingAvgSqNet, epoch, learnRateNet);

    [zPar, trailingAvgPar, trailingAvgSqPar] = adamupdate(zPar, gradientsPar, ...
        trailingAvgPar, trailingAvgSqPar, epoch, learnRatePars);

    [zInit, trailingAvgInit, trailingAvgSqInit] = adamupdate(zInit, gradientsInit, ...
        trailingAvgInit, trailingAvgSqInit, epoch, learnRatePars);

    if mod(epoch, printEvery) == 0 || epoch == 1
        row = {epoch, gather(extractdata(loss)), gather(extractdata(info.dataLoss)), ...
            gather(extractdata(info.odeLoss)), gather(extractdata(info.nonnegativeLoss)), ...
            gather(extractdata(info.paramRegLoss)), currentOriginalDataSSE};
        historyRow = cell2table(row, 'VariableNames', history.Properties.VariableNames);
        history = [history; historyRow]; %#ok<AGROW>

        fprintf(['Epoch %5d | total %.4e | data %.4e | ODE %.4e | original SSE %.4e | ', ...
                 'best SSE %.4e at epoch %d | no improve %d/%d | lambdaODE %.3g\n'], ...
            epoch, row{2}, row{3}, row{4}, row{7}, ...
            bestOriginalDataSSE, bestEpoch, epochsSinceBest, earlyStopPatience, lambdaODERamped);
    end

    if epochsSinceBest >= earlyStopPatience
        stoppedEarly = true;
        fprintf('\nEarly stopping at epoch %d.\n', epoch);
        fprintf('No OriginalDataSSE improvement for %d epochs.\n', earlyStopPatience);
        fprintf('Best OriginalDataSSE %.6g occurred at epoch %d.\n', bestOriginalDataSSE, bestEpoch);
        break;
    end
end

% Restore the model and trainable variables to the best epoch before
% extracting parameters, plotting, and saving.
net = bestNet;
zPar = bestZPar;
zInit = bestZInit;

fprintf('\nUsing best epoch %d with lowest OriginalDataSSE %.6g.\n', bestEpoch, bestOriginalDataSSE);


%% ================= EXTRACT FITTED PARAMETERS AND PREDICTIONS =================
[parsFitDL, initFitDL, deltaParDL, deltaInitDL] = buildTrainableValues( ...
    zPar, zInit, base_pars_DL, base_Init_DL, BPar_DL, BInit_DL, ...
    lowerParDelta_DL, upperParDelta_DL, lowerInitDelta_DL, upperInitDelta_DL);

new_pars = gather(extractdata(parsFitDL));
new_Init = gather(extractdata(initFitDL));
delta_par = gather(extractdata(deltaParDL));
delta_init = gather(extractdata(deltaInitDL));

% Build result tables.
oldVals = base_pars(fit_idx);
newVals = new_pars(fit_idx);
additiveDelta = newVals - oldVals;
percentChange = 100 * additiveDelta ./ oldVals;
percentChange(~isfinite(percentChange)) = NaN;

changedParams = table(fit_idx(:), par_names(fit_idx)', oldVals(:), newVals(:), additiveDelta(:), percentChange(:), ...
    lowerParDelta(:), upperParDelta(:), ...
    'VariableNames', {'Index','Parameter','OldValue','NewValue','AdditiveDelta','PercentChange','DeltaLowerBound','DeltaUpperBound'});

initOldVals = base_Init(init_idx);
initNewVals = new_Init(init_idx);
initAdditiveDelta = initNewVals - initOldVals;
initPercentChange = 100 * initAdditiveDelta ./ initOldVals;
initPercentChange(~isfinite(initPercentChange)) = NaN;

changedInitialConditions = table(init_idx(:), init_names(:), initOldVals(:), initNewVals(:), initAdditiveDelta(:), initPercentChange(:), ...
    lowerInitDelta(:), upperInitDelta(:), ...
    'VariableNames', {'Index','InitialCondition','OldValue','NewValue','AdditiveDelta','PercentChange','DeltaLowerBound','DeltaUpperBound'});

fprintf('\n================ PINN FITTING RESULTS ================\n');
disp('---------------- CHANGED PARAMETERS ----------------');
disp(changedParams);
disp('---------------- CHANGED INITIAL CONDITIONS ----------------');
disp(changedInitialConditions);

%% ================= PREDICT AND PLOT =================
tPlot = linspace(0, runningTime, 1000);
tPlotNorm = dlarray(tPlot ./ runningTime, 'CB');
if useGPU && exist('canUseGPU','file') && canUseGPU
    tPlotNorm = gpuArray(tPlotNorm);
end
YPred = pinnPredict(net, tPlotNorm, initFitDL, stateScale_DL, runningTime);
YPred = gather(extractdata(YPred))';

figure('Name','True PINN fitted states vs observed data','Units','normalized','OuterPosition',[0 0 1 0.9]);
tl = tiledlayout(2,3,'TileSpacing','compact','Padding','compact');
title(tl, 'True PINN fit: neural trajectory + ODE residual + data');

plotStatePINN(nexttile, tPlot, YPred, 1,  obs, 'TNF');
plotStatePINN(nexttile, tPlot, YPred, 3,  obs, 'IL-8 / CXCL8');
plotStatePINN(nexttile, tPlot, YPred, 4,  obs, 'IL-6');
plotStatePINN(nexttile, tPlot, YPred, 8,  obs, 'Temperature');
plotStatePINN(nexttile, tPlot, YPred, 14, obs, 'Heart rate');

ax = nexttile;
axis(ax,'off');
summaryText = sprintf(['Method: true PINN, no ode23s during training\n', ...
    'Epochs trained: %d\nBest epoch: %d\nCollocation points: %d\nBest original data SSE: %.6g\nStopped early: %d\nElapsed minutes: %.2f'], ...
    finalEpochTrained, bestEpoch, numCollocationPoints, bestOriginalDataSSE, stoppedEarly, toc(totalTimer)/60);
text(ax,0.05,0.95,summaryText,'VerticalAlignment','top','Interpreter','none');

%% ================= SAVE RESULTS =================
trainedNet = net;
methodUsed = 'true_pinn_positive_fastgate_with_early_dense_collocation';
finalOriginalDataSSE = bestOriginalDataSSE;

save(matFile, 'trainedNet', 'new_pars', 'new_Init', 'base_pars', 'base_Init', ...
    'fit_idx', 'init_idx', 'par_names', 'init_names', 'changedParams', ...
    'changedInitialConditions', 'history', 'stateScale', 'derivScale', ...
    'delta_par', 'delta_init', 'perturbPercent', 'methodUsed', 'finalOriginalDataSSE', ...
    'bestEpoch', 'bestOriginalDataSSE', 'bestLoss', 'bestInfo', ...
    'earlyStopPatience', 'minSSEImprovement', 'stoppedEarly', 'finalEpochTrained', ...
    'continuingTraining', 'runningTime', 'numCollocationPoints', 'tColloc', ...
    'earlyCollocationEnd', 'numEarlyCollocationPoints', 'numLateCollocationPoints', ...
    'numHiddenUnits', 'numHiddenLayers', 'learnRateNet', 'learnRatePars', ...
    'lambdaData', 'lambdaODE', 'lambdaNonNegative', 'lambdaParamReg', 'odeRampEpochs');

writetable(changedParams, 'fit_pinn_changed_parameters.csv');
writetable(changedInitialConditions, 'fit_pinn_changed_initial_conditions.csv');
writetable(history, 'fit_pinn_history.csv');

fprintf('\nSaved files:\n');
fprintf('  %s\n', matFile);
fprintf('  fit_pinn_changed_parameters.csv\n');
fprintf('  fit_pinn_changed_initial_conditions.csv\n');
fprintf('  fit_pinn_history.csv\n');
fprintf('\nDone. Total elapsed minutes: %.2f\n', toc(totalTimer)/60);

%% ========================================================================
%% LOCAL FUNCTIONS
%% ========================================================================

function [loss, gradientsNet, gradientsPar, gradientsInit, info] = modelGradientsPINN( ...
    net, zPar, zInit, basePars, baseInit, BPar, BInit, ...
    lowerParDelta, upperParDelta, lowerInitDelta, upperInitDelta, ...
    stateScale, derivScale, tCollocNorm, obs, runningTime, ...
    lambdaData, lambdaODE, lambdaNonNegative, lambdaParamReg)

    [parsFit, initFit, deltaPar, deltaInit] = buildTrainableValues( ...
        zPar, zInit, basePars, baseInit, BPar, BInit, ...
        lowerParDelta, upperParDelta, lowerInitDelta, upperInitDelta);

    % Neural trajectory at collocation times.
    Yc = pinnPredict(net, tCollocNorm, initFit, stateScale, runningTime);

    % Compute dY/dt using automatic differentiation.
    numStates = size(Yc,1);
    dYdTnorm = dlarray(zeros(size(Yc), 'like', extractdata(Yc)));
    for s = 1:numStates
        retain = s < numStates;
        dYdTnorm(s,:) = dlgradient(sum(Yc(s,:), 'all'), tCollocNorm, ...
            'EnableHigherDerivatives', true, 'RetainData', retain);
    end
    dYdt = dYdTnorm ./ runningTime;

    rhs = odeRhsDL(Yc, parsFit);
    residual = dYdt - rhs;
    lossODE = mean((residual ./ derivScale).^2, 'all');

    % Data loss at observed data times.
    lossData = dlarray(0);
    originalDataSSE = dlarray(0);
    for k = 1:numel(obs)
        Yo = pinnPredict(net, obs(k).tNorm, initFit, stateScale, runningTime);
        yp = Yo(obs(k).state, :);
        rScaled = (yp - obs(k).yDL) ./ obs(k).scale;
        lossData = lossData + mean(rScaled.^2, 'all');

        rOriginal = yp - obs(k).yDL;
        originalDataSSE = originalDataSSE + mean(rOriginal.^2, 'all');
    end

    % Penalize negative states softly.
    negPart = max(0, -Yc ./ stateScale);
    lossNonnegative = mean(negPart.^2, 'all');

    % Small regularization keeps parameters from hitting bounds unless needed.
    parWidth = max(abs(upperParDelta - lowerParDelta), 1e-12);
    initWidth = max(abs(upperInitDelta - lowerInitDelta), 1e-12);
    lossParamReg = mean((deltaPar ./ parWidth).^2, 'all') + mean((deltaInit ./ initWidth).^2, 'all');

    loss = lambdaData*lossData + lambdaODE*lossODE + ...
           lambdaNonNegative*lossNonnegative + lambdaParamReg*lossParamReg;

    [gradientsNet, gradientsPar, gradientsInit] = dlgradient(loss, net.Learnables, zPar, zInit);

    info = struct();
    info.dataLoss = lossData;
    info.odeLoss = lossODE;
    info.nonnegativeLoss = lossNonnegative;
    info.paramRegLoss = lossParamReg;
    info.originalDataSSE = originalDataSSE;
end

function Y = pinnPredict(net, tNorm, initVec, stateScale, runningTime)
    raw = forward(net, tNorm);
    raw = stripdims(raw);
    tNorm = stripdims(tNorm);

    initVec = reshape(stripdims(initVec), [], 1);
    stateScale = reshape(stripdims(stateScale), [], 1);

    % Convert normalized time back to the model's actual time units.
    actualTime = tNorm .* runningTime;

    % Fast gate: equals zero at time zero, then opens rapidly during the
    % first few hours. This preserves the exact initial condition without
    % suppressing sharp early cytokine peaks by the factor t/runningTime.
    gateTime = 1.0;
    gate = 1 - exp(-actualTime ./ gateTime);

    % Bound the network output to avoid overflow in exp().
    z = 10 .* tanh(raw ./ 10);

    % Numerically stable smooth nonnegative activation.
    softplusZ = max(z, 0) + log(1 + exp(-abs(z)));

    % Positivity-preserving PINN trajectory.
    % At time zero: gate = 0 and Y = initVec exactly.
    % For time >= 0 and nonnegative initial conditions: Y >= 0.
    % The second term allows states that begin at zero to become positive.
    Y = initVec .* exp(gate .* z) + ...
        stateScale .* gate .* softplusZ;
end

function [parsFit, initFit, deltaPar, deltaInit] = buildTrainableValues( ...
    zPar, zInit, basePars, baseInit, BPar, BInit, ...
    lowerParDelta, upperParDelta, lowerInitDelta, upperInitDelta)

    deltaPar = boundedDeltaFromZ(zPar, lowerParDelta, upperParDelta);
    deltaInit = boundedDeltaFromZ(zInit, lowerInitDelta, upperInitDelta);

    parsFit = basePars + BPar * deltaPar;
    initFit = baseInit + BInit * deltaInit;
end

function delta = boundedDeltaFromZ(z, lowerDelta, upperDelta)
    s = 1 ./ (1 + exp(-z));
    delta = lowerDelta + (upperDelta - lowerDelta) .* s;
end

function z = boundedZFromDelta(delta, lowerDelta, upperDelta)
    p = (delta - lowerDelta) ./ (upperDelta - lowerDelta);
    p = min(max(p, 1e-6), 1 - 1e-6);
    z = log(p ./ (1 - p));
end

function rhs = odeRhsDL(Y, pars)
    eps0 = 1e-8;

    % States.
    tnf   = Y(1,:);
    il10  = Y(2,:);
    cxcl8 = Y(3,:);
    il6   = Y(4,:);
    ma    = Y(5,:);
    mr    = Y(6,:);
    pe    = Y(7,:);
    temp2 = Y(8,:);
    pp    = Y(9,:);
    Vla   = Y(10,:);
    Vsa   = Y(11,:);
    Vlv   = Y(12,:);
    Vsv   = Y(13,:);
    hr    = Y(14,:);
    no    = Y(15,:);
    rs    = Y(16,:);
    D     = Y(17,:);

    % Smooth-positive versions are used inside Hill functions so the PINN is differentiable.
    tnfP   = smoothPositive(tnf);
    il10P  = smoothPositive(il10);
    cxcl8P = smoothPositive(cxcl8); %#ok<NASGU>
    il6P   = smoothPositive(il6);
    maP    = smoothPositive(ma);
    mrP    = smoothPositive(mr);
    peP    = smoothPositive(pe);
    ppP    = smoothPositive(pp);
    noP    = smoothPositive(no);
    DP     = smoothPositive(D);

    % Parameters.
    k10    = pars(1);    k10m   = pars(2);    k6     = pars(3);    k6m    = pars(4);
    k8     = pars(5);    k8m    = pars(6);    ktnf   = pars(7);    ktnfm  = pars(8);
    kma    = pars(9);    kmpe   = pars(10);   kmr    = pars(11);   kpe    = pars(12); %#ok<NASGU>
    x610   = pars(13);   x66    = pars(14);   x6tnf  = pars(15);   x810   = pars(16);
    x8tnf  = pars(17);   x106   = pars(18);   xtnf10 = pars(19);   xtnf6  = pars(20);
    xmpe   = pars(21);   xm10   = pars(22);   xmtnf  = pars(23);   h106   = pars(24);
    h6tnf  = pars(25);   h66    = pars(26);   h610   = pars(27);   h8tnf  = pars(28);
    h810   = pars(29);   htnf10 = pars(30);   htnf6  = pars(31);   hm10   = pars(32);
    hmtnf  = pars(33);   hmpe   = pars(34);   stnf   = pars(35);   s10    = pars(36);
    s8     = pars(37);   s6     = pars(38);   sm     = pars(39);   mmax   = pars(40);
    k6tnf  = pars(41);   k8tnf  = pars(42);   k106   = pars(43);   kmtnf  = pars(44);

    tau1   = pars(45);   TM     = pars(46);   Tm     = pars(47);   kt     = pars(48);
    kttnf  = pars(49);   kt6    = pars(50);   kt10   = pars(51);   xttnf  = pars(52);
    xt6    = pars(53);   xt10   = pars(54);   httnf  = pars(55);   ht6    = pars(56);
    ht10   = pars(57);

    tau2   = pars(58);   HM     = pars(59);   Hm     = pars(60);   kh     = pars(61);
    xht    = pars(62);   hht    = pars(63);

    ppM    = pars(64);   kpepp  = pars(65);   kpp    = pars(66);

    Ra     = pars(67);   Rv     = pars(68);   Rs     = pars(69);   Cla    = pars(70);
    Csa    = pars(71);   Clv    = pars(72);   Csv    = pars(73);   Em     = pars(74);
    EM     = pars(75);

    knom   = pars(76);   kno    = pars(77);   xntnf  = pars(78);   xn10   = pars(79);
    hntnf  = pars(80);   hn10   = pars(81);   krpp   = pars(82);   krno   = pars(83);
    kr     = pars(84);   xrpp   = pars(85);   hrpp   = pars(86);   xhp    = pars(87);
    hhp    = pars(88);

    kpg    = pars(89);   peinf  = pars(90);   kpm    = pars(91);   xI10   = pars(92);
    muno   = pars(93);   kpn    = pars(94);

    kdn    = pars(95);   mud    = pars(96);   xdn    = pars(97);   alpha  = pars(98);
    hmI10  = pars(99);   sM     = pars(100);  kmp    = pars(101);

    kD     = pars(102);  xDam   = pars(103);  hmDa   = pars(104); %#ok<NASGU>
    xm10D  = pars(105);  hm10D  = pars(106);  hmD    = pars(107);
    knod   = pars(108);  xnDl10 = pars(109);  hnDl10 = pars(110); %#ok<NASGU>
    xn10D  = pars(111);  hn10D  = pars(112);  ktnfhr = pars(113);  BPo    = pars(114);

    % Immune/cytokine/pathogen/damage equations.
    dtnf = ktnfm*maP.*hillFS(il10P,xtnf10,htnf10).*hillFS(il6P,xtnf6,htnf6).*(1 + ktnfhr*smoothAbs(hr - Hm)) - ktnf*(tnf - stnf);

    dil6 = maP.*(k6m + k6tnf*hillF(tnfP,x6tnf,h6tnf)).*hillFS(il6P,x66,h66).*hillFS(il10P,x610,h610) - k6*(il6 - s6);

    dil8 = maP.*(k8m + k8tnf*hillF(tnfP,x8tnf,h8tnf)).*hillFS(il10P,x810,h810) - k8*(cxcl8 - s8);

    dil10 = maP.*(k10m + k106*hillF(il6P,x106,h106)) - k10*(il10 - s10);

    dmr = kmr*mrP.*(1 - mrP./mmax) - hillF(peP,xmpe,hmpe).*(sm + kmtnf*hillF(tnfP,xmtnf,hmtnf)).*hillFS(il10P,xm10,hm10).*mrP;

    dma = hillF(peP,xmpe,hmpe).*(sm + kmtnf*hillF(tnfP,xmtnf,hmtnf)).*hillFS(il10P,xm10,hm10).*mrP + ...
          kD*hillFS(il10P,xm10D,hm10D).*DP - kma*ma;

    dpe = kpg*peP.*(1 - peP./peinf) - (kpm*sM*peP)./(muno + kmp*peP + eps0) - kpn*maP.*peP.*hillFS(il10P,xI10,hmI10);

    dD = kdn*hillF(alpha*noP + peP, xdn, hmD) - mud*D;

    % Temperature.
    Ftemp = kt*(TM-Tm)*(kttnf*hillF(smoothAbs(tnf-stnf),xttnf,httnf) + ...
            kt6*hillF(smoothAbs(il6-s6),xt6,ht6) - ...
            kt10*(1 - hillFS(smoothAbs(il10-s10),xt10,ht10))) + Tm;
    dtemp2 = (-temp2 + Ftemp)./tau1;

    % Pain perception.
    dpp = -kpepp*peP.*ppP + kpp*(ppM - pp);

    % Cardiovascular volumes/pressures.
    pla = Vla./Cla;
    psa = Vsa./Csa;
    plv = Vlv./Clv;
    psv = Vsv./Csv;

    qa = (pla - psa)./Ra;
    qs = (psa - psv)./rs;
    qv = (psv - plv)./Rv;

    Vstr = -(pla./EM - plv./Em);
    Q = Vstr.*hr./60;

    dVla = Q - qa;
    dVsa = qa - qs;
    dVlv = qv - Q;
    dVsv = qs - qv;

    % Heart rate. The original model has an if-statement at BP=100.
    % A PINN needs differentiability, so this uses a smooth switch.
    bp = Vla./Cla;
    absTemp = smoothAbs(temp2 - Tm);
    absXht = smoothAbs(xht - Tm);
    tempTerm = absTemp.^hht ./ (absTemp.^hht + absXht.^hht + eps0);

    absBp = smoothAbs(bp - BPo);
    absXhp = smoothAbs(xhp - BPo);
    denomPressure = absBp.^hhp + absXhp.^hhp + eps0;
    pressureHigh = absXhp.^hhp ./ denomPressure;
    pressureLow  = absBp.^hhp  ./ denomPressure;

    switchSharpness = 0.25;
    wHigh = 1 ./ (1 + exp(-switchSharpness*(bp - 100)));
    pressureTerm = wHigh.*pressureHigh + (1 - wHigh).*pressureLow;

    ft = kh*(HM - Hm).*tempTerm.*pressureTerm + Hm;
    dhr = (-hr + ft)./tau2;

    % Nitric oxide.
    dno = knom*maP.*hillF(tnfP,xntnf,hntnf).*hillFS(il10P,xn10,hn10) + ...
          knod*DP.*hillFS(il10P,xn10D,hn10D) - kno*no;

    % Resistance.
    dppAbsPow = smoothAbs(dpp).^hrpp;
    drs = krpp*(dppAbsPow ./ (dppAbsPow + xrpp.^hrpp + eps0)) - krno*no - kr*(rs - Rs);

    rhs = [dtnf; dil10; dil8; dil6; dma; dmr; dpe; dtemp2; dpp; ...
           dVla; dVsa; dVlv; dVsv; dhr; dno; drs; dD];
end

function y = hillF(v,x,hill)
    eps0 = 1e-8;
    v = smoothPositive(v);
    x = smoothPositive(x);
    vp = v.^hill;
    xp = x.^hill;
    y = vp ./ (vp + xp + eps0);
end

function y = hillFS(v,x,hill)
    eps0 = 1e-8;
    v = smoothPositive(v);
    x = smoothPositive(x);
    vp = v.^hill;
    xp = x.^hill;
    y = xp ./ (vp + xp + eps0);
end

function y = smoothPositive(x)
    y = 0.5*(x + sqrt(x.^2 + 1e-8));
end

function y = smoothAbs(x)
    y = sqrt(x.^2 + 1e-8);
end

function obs = makeObs(name, state, t, y)
    valid = isfinite(t(:)) & isfinite(y(:));
    obs = struct();
    obs.name = name;
    obs.state = state;
    obs.t = t(valid);
    obs.y = y(valid);
    obs.tNorm = [];
    obs.yDL = [];
    obs.scale = [];
end

function T = readSurvivorTable(csvFile)
    opts = detectImportOptions(csvFile, 'TreatAsMissing', {'#DIV/0!', 'NaN', 'nan', ''});
    Traw = readtable(csvFile, opts);

    for j = 1:width(Traw)
        vname = Traw.Properties.VariableNames{j};
        col = Traw.(vname);
        if iscell(col) || isstring(col) || ischar(col)
            converted = str2double(string(col));
            if any(isfinite(converted))
                Traw.(vname) = converted;
            end
        end
    end

    T = table();
    T.Time      = getNumericColumn(Traw, {'Time','time','TIME','Hour','Hours','hours'});
    T.HR_Mean   = getNumericColumn(Traw, {'HR_Mean','HR_Median','HR'});
    T.TEMP_Mean = getNumericColumn(Traw, {'TEMP_Mean','TEMP_Median','Temp_Mean','Temp_Median','TEMP','Temp'});
    T.TNF_Mean  = getNumericColumn(Traw, {'TNF_Mean','TNF_Median','TNF'});
    T.IL_6_Mean = getNumericColumn(Traw, {'IL_6_Mean','IL_6_Median','IL6_Mean','IL6_Median','IL6','IL_6'});
    T.IL_8_Mean = getNumericColumn(Traw, {'IL_8_Mean','IL_8_Median','IL8_Mean','IL8_Median','IL8','IL_8'});
end

function x = getNumericColumn(T, candidates)
    names = T.Properties.VariableNames;
    lowerNames = lower(names);

    for k = 1:numel(candidates)
        idx = find(strcmpi(names, candidates{k}), 1);
        if ~isempty(idx)
            x = T.(names{idx});
            x = x(:);
            return
        end
    end

    normalizedNames = regexprep(lowerNames, '[^a-z0-9]', '');
    for k = 1:numel(candidates)
        target = regexprep(lower(candidates{k}), '[^a-z0-9]', '');
        idx = find(strcmp(normalizedNames, target), 1);
        if ~isempty(idx)
            x = T.(names{idx});
            x = x(:);
            return
        end
    end

    error('Could not find any of these columns in the CSV: %s', strjoin(candidates, ', '));
end

function plotStatePINN(ax, tPlot, YPred, stateNum, obs, titleText)
    hold(ax,'on');
    grid(ax,'on');
    box(ax,'on');
    plot(ax, tPlot, YPred(:,stateNum), 'LineWidth', 1.8, 'DisplayName','PINN trajectory');

    for k = 1:numel(obs)
        if obs(k).state == stateNum
            plot(ax, obs(k).t, obs(k).y, 'o', 'MarkerSize', 5, 'DisplayName','Observed data');
            break
        end
    end

    xlabel(ax,'Time');
    ylabel(ax,titleText);
    title(ax, sprintf('State %d: %s', stateNum, titleText));
    legend(ax,'Location','best');
end

function validateNetworkCompatibility(savedNet, templateNet)
    if ~isa(savedNet, 'dlnetwork')
        error('The saved trainedNet is not a dlnetwork object.');
    end

    savedLearnables = savedNet.Learnables;
    templateLearnables = templateNet.Learnables;

    if height(savedLearnables) ~= height(templateLearnables)
        error(['The saved trainedNet architecture does not match the current architecture. ' ...
               'Keep numHiddenUnits and numHiddenLayers unchanged when resuming.']);
    end

    for k = 1:height(savedLearnables)
        sameLayer = string(savedLearnables.Layer(k)) == string(templateLearnables.Layer(k));
        sameParameter = string(savedLearnables.Parameter(k)) == string(templateLearnables.Parameter(k));
        sameSize = isequal(size(savedLearnables.Value{k}), size(templateLearnables.Value{k}));

        if ~(sameLayer && sameParameter && sameSize)
            error(['The saved trainedNet architecture does not match the current architecture. ' ...
                   'Keep the same layer names, numHiddenUnits, and numHiddenLayers.']);
        end
    end
end

function ensureExpectedFunctionFile(expectedName, pattern)
    if isfile(expectedName)
        return
    end

    candidates = dir(pattern);
    if isempty(candidates)
        error('Cannot find %s. Put it in the same folder as this script.', expectedName);
    end

    copyfile(candidates(1).name, expectedName);
    fprintf('Copied %s to %s so MATLAB can call the function.\n', candidates(1).name, expectedName);
end

function par_names = makeParameterNames()
    par_names = { ...
        'k10','k10m','k6','k6m','k8','k8m','ktnf','ktnfm','kma','kmpe', ...
        'kmr','kpe','x610','x66','x6tnf','x810','x8tnf','x106','xtnf10','xtnf6', ...
        'xmpe','xm10','xmtnf','h106','h6tnf','h66','h610','h8tnf','h810','htnf10', ...
        'htnf6','hm10','hmtnf','hmpe','stnf','s10','s8','s6','sm','mmax', ...
        'k6tnf','k8tnf','k106','kmtnf', ...
        'tau1','TM','Tm','kt','kttnf','kt6','kt10','xttnf','xt6','xt10', ...
        'httnf','ht6','ht10', ...
        'tau2','HM','HI','kh','xht','hht', ...
        'ppM','kpepp','kpp', ...
        'Ra','Rv','Rs','Cla','Csa','Clv','Csv','Em','EM', ...
        'knom','kno','xntnf','xn10','hntnf','hn10', ...
        'krpp','krno','kr','xrpp','hrpp','xhp','hhp', ...
        'kpg','peinf','kpm','xI10','muno','kpn', ...
        'kdn','mud','xdn','alpha','hmI10', ...
        'sM','kmp', ...
        'kD','xDam','hmDa','xm10D','hm10D','hmD','knod','xnDl10','hnDl10','xn10D','hn10D','ktnfhr','BPo'};
end
