% main_script_bayesopt_survivor_median.m
% BAYESIAN OPTIMIZATION ONLY.
%
% Updated version:
%   1) Uses sum of per-state averaged squared residuals:
%        SSE = (1/N_HR)sum(r_HR^2) + (1/N_TEMP)sum(r_TEMP^2) + ...
%      There is NO final division by the number of measured states.
%   2) Uses additive changes, not multiplicative perturbations:
%        new_parameter = starting_parameter + delta
%   3) Allows the pathogen initial condition PE(0) = Init(7) to be changed
%      additively.
%   4) Saves both new_pars and new_Init to fit_bayesopt_results.mat.
%   5) If fit_bayesopt_results.mat already exists, it can load saved
%      new_pars/new_Init as the starting baseline for another BayesOpt run.

close all;
clear;
clc;

%% ================= USER SETTINGS =================
csvFile = 'Survivor_Data.csv';
matFile = 'fit_bayesopt_results.mat';

% TRUE  = if matFile exists, start BayesOpt from saved new_pars/new_Init
% FALSE = ignore matFile and start from load_pars_Init_Copeland_Edited baseline
LOAD_EXISTING_MAT_AS_START = true;

% If TRUE, make a timestamped backup before overwriting matFile.
BACKUP_EXISTING_MAT_BEFORE_SAVE = true;

runningTime = 700;
tspan = 0:0.1:runningTime;

useLogFit = false;

% Additive change bound, expressed as a percent of each starting value.
% Example: 80 means each fitted value can change by +/- 0.80*abs(starting value).
% The code applies this as: new_value = starting_value + delta.
% It does NOT use new_value = starting_value*(1 + theta).
perturbPercent = 50;

% Bayesian optimization settings.
maxObjectiveEvaluations = 2500;
numSeedPoints = 123;
useParallel = true;

% Optional local refinement after bayesopt. Keep false if you only want bayesopt.
doLocalRefineAfterBayes = false;
maxIterRefine = 500;
maxFunEvalsRefine = 2000;

rng('shuffle');

%% ================= LOAD CSV DATA =================
opts = detectImportOptions(csvFile, 'TreatAsMissing', {'#DIV/0!'});
T = readtable(csvFile, opts);

for j = 1:width(T)
    vname = T.Properties.VariableNames{j};
    if iscell(T{:,j}) || isstring(T{:,j}) || ischar(T{1,j})
        T.(vname) = str2double(string(T{:,j}));
    end
end

timeAll = T.Time;

% Survivor_Median.csv uses Median columns.
HR     = T.HR_Mean;      HRsd   = T.HR_SD;
TEMP   = T.TEMP_Mean;    TEMPsd = T.TEMP_SD;
TNF    = T.TNF_Mean;     TNFsd  = T.TNF_SD;
IL6    = T.IL_6_Mean;    IL6sd  = T.IL_6_SD;
IL8    = T.IL_8_Mean;    IL8sd  = T.IL_8_SD;

data = struct();
data.hr   = HR;
data.TNF  = TNF;
data.IL6  = IL6;
data.IL8  = IL8;
data.temp = TEMP;

%% ================= LOAD PARAMETERS AND INITIAL CONDITIONS =================
global ODE_TOL
ODE_TOL = 1e-6;

[pars, Init] = load_pars_Init_Copeland_Edited(data);
pars = pars(:);
Init = Init(:);

% Keep the original baseline separately for reference.
original_base_pars = pars;
original_base_Init = Init;

% Default starting point for this BayesOpt run.
base_pars = pars;
base_Init = Init;
startSource = 'original load_pars_Init_Copeland_Edited baseline';
previousBestCost = NaN;
previousInitialCost = NaN;

% =====================================================================
% RESUME / CONTINUE FROM SAVED .MAT FILE
% =====================================================================
% If fit_bayesopt_results.mat exists, use the saved fitted values as the
% starting baseline for this new BayesOpt run. Then delta = 0 means the
% previously fitted parameter set, not the original parameter set.
if LOAD_EXISTING_MAT_AS_START && isfile(matFile)
    fprintf('\nFound existing %s. Loading saved fitted values as BayesOpt starting point...\n', matFile);
    Rprev = load(matFile);

    if isfield(Rprev, 'new_pars') && numel(Rprev.new_pars) == numel(base_pars)
        base_pars = Rprev.new_pars(:);
        startSource = sprintf('saved new_pars from %s', matFile);
    else
        warning('Existing mat file does not contain valid new_pars. Using original parameter baseline.');
    end

    if isfield(Rprev, 'new_Init') && numel(Rprev.new_Init) == numel(base_Init)
        base_Init = Rprev.new_Init(:);
        startSource = sprintf('%s and saved new_Init from %s', startSource, matFile);
    else
        warning('Existing mat file does not contain valid new_Init. Using original initial conditions.');
    end

    if isfield(Rprev, 'bestCost')
        previousBestCost = Rprev.bestCost;
    end
    if isfield(Rprev, 'initialCost')
        previousInitialCost = Rprev.initialCost;
    end

    fprintf('Starting source: %s\n', startSource);
    if isfinite(previousBestCost)
        fprintf('Previous saved best cost = %.6g\n', previousBestCost);
    end
else
    fprintf('\nNo existing mat file loaded. Starting from original baseline values.\n');
end

par_names = { ...
    'k10','k10m','k6','k6m','k8','k8m','ktnf','ktnfm','kma','kmpe', ...          % 1-10
    'kmr','kpe','x610','x66','x6tnf','x810','x8tnf','x106','xtnf10','xtnf6', ... % 11-20
    'xmpe','xm10','xmtnf','h106','h6tnf','h66','h610','h8tnf','h810','htnf10', ... % 21-30
    'htnf6','hm10','hmtnf','hmpe','stnf','s10','s8','s6','sm','mmax', ...       % 31-40
    'k6tnf','k8tnf','k106','kmtnf', ...                                        % 41-44
    'tau1','TM','Tm','kt','kttnf','kt6','kt10','xttnf','xt6','xt10', ...        % 45-54
    'httnf','ht6','ht10', ...                                                   % 55-57
    'tau2','HM','HI','kh','xht','hht', ...                                      % 58-63
    'ppM','kpepp','kpp', ...                                                    % 64-66
    'Ra','Rv','Rs','Cla','Csa','Clv','Csv','Em','EM', ...                       % 67-75
    'knom','kno','xntnf','xn10','hntnf','hn10', ...                             % 76-81
    'krpp','krno','kr','xrpp','hrpp','xhp','hhp', ...                           % 82-88
    'kpg','peinf','kpm','xI10','muno','kpn', ...                                % 89-94
    'kdn','mud','xdn','alpha','hmI10', ...                                      % 95-99
    'sM','kmp', ...                                                             % 100-101
    'kD','xDam','hmDa','xm10D','hm10D','hmD','knod','xnDl10','hnDl10','xn10D','hn10D','ktnfhr','BPo'}; % 102-114

if numel(par_names) ~= numel(pars)
    error('par_names has %d names, but pars has %d values.', numel(par_names), numel(pars));
end

parIndexTable = table((1:numel(base_pars))', par_names(:), base_pars(:), ...
    'VariableNames', {'Index','Parameter','StartingValue'});

%% ================= SELECT PARAMETERS TO FIT =================
fit_idx = [
    34;   %  1 hmpe
    76;   %  2 knom
    9;    %  3 kma
    91;   %  4 kpm
    98;   %  5 alpha
    % 97;   %  6 xdn
    % 100;  %  7 sM
    % 93;   %  8 muno
    % 21;   %  9 xmpe
    62;   % 10 xht
    % 39;   % 11 sm
    26;   % 12 h66
    % 107;  % 13 hmD
    % 14;   % 14 x66
    % 102;  % 15 kD
    % 20;   % 16 xtnf6
    % 81;   % 17 hn10
    % 95;   % 18 kdn
    % 7;    % 19 ktnf
    % 96;   % 20 mud
    % 77;   % 21 kno
    % 63;   % 22 hht
    % 89;   % 23 kpg
    % 31;   % 24 htnf6
    % 8;    % 25 ktnfm
    % 32;   % 26 hm10
    % 36;   % 27 s10
    % 48;   % 28 kt
    % 2;    % 29 k10m
    % 78;   % 30 xntnf
    % 79;   % 31 xn10
    % 1;    % 32 k10
    % 43;   % 33 k106
    % 61;   % 34 kh
    % 3;    % 35 k6
    % 19;   % 36 xtnf10
    % 45;   % 37 tau1
    % 4;    % 38 k6m
    % 94;   % 39 kpn
    % 22;   % 40 xm10
    % 80;   % 41 hntnf
    % 49;   % 42 kttnf
    % 87;   % 43 xhp
    % 41;   % 44 k6tnf
    % 5;    % 45 k8
    % 113;  % 46 ktnfhr
    % 42;   % 47 k8tnf
    % 101;  % 48 kmp
    % 64;   % 49 ppM
    % 50;   % 50 kt6
];
fit_idx = fit_idx(:);

% Initial condition index to fit.
% Init = [tnfI il10I il8I il6I maI mrI peI tempI ppI VlaI VsaI VlvI VsvI HI noI Rs DI]
% Therefore pathogen initial condition PE(0) is Init(7).
init_idx = 7;
init_names = {'PE_initial'};

morrisRank = (1:numel(fit_idx))';
fitTable = [table(morrisRank, 'VariableNames', {'MorrisRank'}), parIndexTable(fit_idx,:)];

disp('---------------- PARAMETERS BEING FIT ----------------');
disp(fitTable);

initFitTable = table(init_idx(:), init_names(:), base_Init(init_idx), ...
    'VariableNames', {'Index','InitialCondition','StartingValue'});

disp('---------------- INITIAL CONDITIONS BEING FIT ----------------');
disp(initFitTable);

%% ================= DATA USED IN OBJECTIVE FUNCTION =================
fitData = struct();
fitData.hr.time    = timeAll;
fitData.hr.mean    = HR;
fitData.hr.sd      = HRsd;
fitData.hr.state   = 14;
fitData.hr.name    = 'HR';

fitData.temp.time  = timeAll;
fitData.temp.mean  = TEMP;
fitData.temp.sd    = TEMPsd;
fitData.temp.state = 8;
fitData.temp.name  = 'Temperature';

fitData.tnf.time   = timeAll;
fitData.tnf.mean   = TNF;
fitData.tnf.sd     = TNFsd;
fitData.tnf.state  = 1;
fitData.tnf.name   = 'TNF';

fitData.il6.time   = timeAll;
fitData.il6.mean   = IL6;
fitData.il6.sd     = IL6sd;
fitData.il6.state  = 4;
fitData.il6.name   = 'IL-6';

fitData.il8.time   = timeAll;
fitData.il8.mean   = IL8;
fitData.il8.sd     = IL8sd;
fitData.il8.state  = 3;
fitData.il8.name   = 'IL-8';

% The cost function compares the model only at CSV/data times.
fitTimes = unique(timeAll(isfinite(timeAll)), 'sorted');
costTspan = unique([0; fitTimes(:); runningTime], 'sorted');

%% ================= ADDITIVE DELTA BOUNDS =================
if perturbPercent > 1
    perturbFraction = perturbPercent/100;
else
    perturbFraction = perturbPercent;
end
perturbFraction = abs(perturbFraction);

%% ================= OBJECTIVE FUNCTION =================
nfitPar  = numel(fit_idx);
nfitInit = numel(init_idx);
nfit     = nfitPar + nfitInit;

% delta = additive changes for parameters followed by additive changes for ICs.
% delta = 0 means the current starting parameters and initial conditions.
initialDelta = zeros(nfit,1);

% Build variable-specific additive bounds.
% Example: base value = 10 and perturbPercent = 80 gives delta in [-8, +8].
startingFitValues = [base_pars(fit_idx); base_Init(init_idx)];
deltaMax = perturbFraction .* abs(startingFitValues);

% If a selected value is exactly zero, allow a small default additive range.
% Adjust zeroValueAdditiveScale if you intentionally fit a zero-valued parameter/IC.
zeroValueAdditiveScale = 1;
zeroMask = deltaMax == 0;
deltaMax(zeroMask) = perturbFraction * zeroValueAdditiveScale;

lowerDelta = -deltaMax;
upperDelta =  deltaMax;

% Keep nonnegative starting values from crossing below zero.
nonnegativeStart = startingFitValues >= 0;
lowerDelta(nonnegativeStart) = max(lowerDelta(nonnegativeStart), -0.999999*startingFitValues(nonnegativeStart));

initialCost = cost_function_2(initialDelta, fit_idx, base_pars, base_Init, fitData, costTspan, useLogFit, perturbPercent, init_idx);
fprintf('\nInitial cost using starting parameters and initial conditions = %.6g\n', initialCost);

% bayesopt needs one scalar objective value.
obj = @(delta) cost_function_2(delta, fit_idx, base_pars, base_Init, fitData, costTspan, useLogFit, perturbPercent, init_idx);

varNames = arrayfun(@(k) sprintf('delta%02d', k), 1:nfit, 'UniformOutput', false);

vars = optimizableVariable.empty(nfit,0);
for k = 1:nfit
    vars(k) = optimizableVariable(varNames{k}, [lowerDelta(k) upperDelta(k)], 'Type', 'real');
end

initialX = array2table(zeros(1,nfit), 'VariableNames', varNames);

bayesObjective = @(X) bayesDeltaObjective(X, obj, varNames);

fprintf('\n================ BAYESIAN OPTIMIZATION ================\n');
fprintf('Additive change bound: +/- %.2f%% of each starting value\n', 100*perturbFraction);
fprintf('Bayesian objective evaluations: %d\n', maxObjectiveEvaluations);
fprintf('Seed points: %d\n', numSeedPoints);
fprintf('Fitted parameters: %d\n', nfitPar);
fprintf('Fitted initial conditions: %d\n', nfitInit);

results = bayesopt(bayesObjective, vars, ...
    'MaxObjectiveEvaluations', maxObjectiveEvaluations, ...
    'NumSeedPoints', numSeedPoints, ...
    'InitialX', initialX, ...
    'IsObjectiveDeterministic', true, ...
    'AcquisitionFunctionName', 'expected-improvement-plus', ...
    'UseParallel', useParallel, ...
    'Verbose', 1);

bayesBestX = results.XAtMinObjective;
delta_hat_raw = rowToDelta(bayesBestX, varNames);
delta_hat = clampAdditiveDelta(delta_hat_raw, lowerDelta, upperDelta);
bestCost = obj(delta_hat);
methodUsed = 'bayesopt_additive_delta';
exitflag = NaN;
output = struct();
output.bayesoptResults = results;
output.objectiveEvaluations = height(results.XTrace);
output.lowerDelta = lowerDelta;
output.upperDelta = upperDelta;

fprintf('\nBest Bayesian optimization cost = %.6g\n', bestCost);

%% ================= OPTIONAL LOCAL REFINEMENT =================
if doLocalRefineAfterBayes
    fprintf('\n================ LOCAL REFINE AFTER BAYESOPT ================\n');
    refineOptions = optimset('Display','iter', ...
                             'MaxIter',maxIterRefine, ...
                             'MaxFunEvals',maxFunEvalsRefine, ...
                             'TolX',1e-12, ...
                             'TolFun',1e-12);

    [delta_refined_raw, cost_refined, exitflag_refine, output_refine] = fminsearch(obj, delta_hat, refineOptions);
    delta_refined = clampAdditiveDelta(delta_refined_raw, lowerDelta, upperDelta);
    cost_refined_checked = obj(delta_refined);

    if cost_refined_checked < bestCost
        delta_hat_raw = delta_refined_raw;
        delta_hat = delta_refined;
        bestCost = cost_refined_checked;
        methodUsed = 'bayesopt_additive_delta_plus_fminsearch_refine';
        exitflag = exitflag_refine;
        output.fminsearchRefine = output_refine;
        output.fminsearchReportedCost = cost_refined;
        fprintf('Local refine improved cost to %.6g\n', bestCost);
    else
        fprintf('Local refine did not improve the Bayesian optimization result.\n');
    end
end

%% ================= BUILD NEW PARAMETERS AND INITIAL CONDITIONS =================
delta_par  = delta_hat(1:nfitPar);
delta_init = delta_hat(nfitPar+1:end);

new_pars = base_pars;
new_pars(fit_idx) = base_pars(fit_idx) + delta_par;

new_Init = base_Init;
new_Init(init_idx) = base_Init(init_idx) + delta_init;

%% ================= PARAMETER RESULTS TABLE =================
idx = fit_idx(:);
paramNames = par_names(idx);
paramNames = paramNames(:);
oldVals = base_pars(idx);
newVals = new_pars(idx);
additiveDelta = newVals - oldVals;
percentChange = 100 * additiveDelta ./ oldVals;
percentChange(~isfinite(percentChange)) = NaN;
deltaBoundLower = lowerDelta(1:nfitPar);
deltaBoundUpper = upperDelta(1:nfitPar);
boundHit = abs(additiveDelta(:) - deltaBoundLower(:)) < 1e-12 | ...
           abs(additiveDelta(:) - deltaBoundUpper(:)) < 1e-12;

changedParams = table( ...
    idx, ...
    paramNames, ...
    oldVals, ...
    newVals, ...
    additiveDelta, ...
    percentChange, ...
    deltaBoundLower(:), ...
    deltaBoundUpper(:), ...
    boundHit(:), ...
    'VariableNames', {'Index','Parameter','OldValue','NewValue','AdditiveDelta','PercentChange','DeltaLowerBound','DeltaUpperBound','HitAdditiveBound'} ...
);

%% ================= INITIAL CONDITION RESULTS TABLE =================
initOldVals = base_Init(init_idx);
initNewVals = new_Init(init_idx);
initAdditiveDelta = initNewVals - initOldVals;
initPercentChange = 100 * initAdditiveDelta ./ initOldVals;
initPercentChange(~isfinite(initPercentChange)) = NaN;
initLowerDelta = lowerDelta(nfitPar+1:end);
initUpperDelta = upperDelta(nfitPar+1:end);
initBoundHit = abs(initAdditiveDelta(:) - initLowerDelta(:)) < 1e-12 | ...
               abs(initAdditiveDelta(:) - initUpperDelta(:)) < 1e-12;

changedInitialConditions = table( ...
    init_idx(:), ...
    init_names(:), ...
    initOldVals(:), ...
    initNewVals(:), ...
    initAdditiveDelta(:), ...
    initPercentChange(:), ...
    initLowerDelta(:), ...
    initUpperDelta(:), ...
    initBoundHit(:), ...
    'VariableNames', {'Index','InitialCondition','OldValue','NewValue','AdditiveDelta','PercentChange','DeltaLowerBound','DeltaUpperBound','HitAdditiveBound'} ...
);

%% ================= DISPLAY RESULTS =================
disp('---------------- FITTING RESULTS ----------------');
fprintf('Initial cost = %.6g\n', initialCost);
fprintf('Best cost = %.6g\n', bestCost);
fprintf('Method used = %s\n', methodUsed);

disp('---------------- CHANGED PARAMETERS ----------------');
disp(changedParams);

disp('---------------- CHANGED INITIAL CONDITIONS ----------------');
disp(changedInitialConditions);

%% ================= SAVE RESULTS =================
bayesTrace = results.XTrace;
bayesObjectiveTrace = results.ObjectiveTrace;
bayesEvaluationTable = [bayesTrace table(bayesObjectiveTrace, 'VariableNames', {'Objective'})];

% Backup the previous mat file before overwriting it, so you do not lose the
% earlier fitted values when continuing optimization.
if BACKUP_EXISTING_MAT_BEFORE_SAVE && isfile(matFile)
    backupName = sprintf('fit_bayesopt_results_backup_%s.mat', datestr(now, 'yyyymmdd_HHMMSS'));
    copyfile(matFile, backupName);
    fprintf('\nBacked up previous mat file to: %s\n', backupName);
end

save(matFile, ...
    'base_pars', 'new_pars', 'fit_idx', 'par_names', ...
    'base_Init', 'new_Init', 'init_idx', 'init_names', ...
    'original_base_pars', 'original_base_Init', ...
    'startSource', 'previousBestCost', 'previousInitialCost', ...
    'delta_hat', 'delta_hat_raw', 'delta_par', 'delta_init', ...
    'lowerDelta', 'upperDelta', 'perturbPercent', 'bestCost', 'initialCost', ...
    'methodUsed', 'exitflag', 'output', ...
    'changedParams', 'changedInitialConditions', ...
    'bayesEvaluationTable', 'results');

writetable(changedParams, 'fit_bayesopt_changed_parameters.csv');
writetable(changedInitialConditions, 'fit_bayesopt_changed_initial_conditions.csv');
writetable(bayesEvaluationTable, 'fit_bayesopt_evaluations.csv');

fprintf('\nSaved files:\n');
fprintf('  %s\n', matFile);
fprintf('  fit_bayesopt_changed_parameters.csv\n');
fprintf('  fit_bayesopt_changed_initial_conditions.csv\n');
fprintf('  fit_bayesopt_evaluations.csv\n');
fprintf('\nRun run.m to graph either the baseline parameters or the saved Bayesian fitted parameters/initial conditions.\n');

%% ================= LOCAL FUNCTIONS =================
function objectiveValue = bayesDeltaObjective(X, obj, varNames)
    delta = rowToDelta(X, varNames);

    try
        objectiveValue = obj(delta);
    catch ME
        warning('Objective failed: %s', ME.message);
        objectiveValue = 1e20;
    end

    if isempty(objectiveValue) || ~isscalar(objectiveValue) || ~isfinite(objectiveValue)
        objectiveValue = 1e20;
    end
end

function delta = rowToDelta(X, varNames)
    delta = table2array(X(1,varNames));
    delta = delta(:);
end

function delta = clampAdditiveDelta(delta, lowerDelta, upperDelta)
    delta = delta(:);
    lowerDelta = lowerDelta(:);
    upperDelta = upperDelta(:);
    delta = min(max(delta, lowerDelta), upperDelta);
end
