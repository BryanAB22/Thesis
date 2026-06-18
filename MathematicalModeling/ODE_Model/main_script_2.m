% main_script_2.m
% Fits selected parameters by direct percent perturbation around baseline.
% No exponential parameterization and no penalty term are used.

close all;
clear;
clc;

csvFile = 'placebo_plotted_data(Survivor_Data).csv';
runningTime = 700;
tspan = 0:0.1:runningTime;   % dense time grid for plotting only

useLogFit = true;

maxIter = 50000;
maxFunEvals = 15000;

nStarts = 1;
perturbPercent = 30;         % CHANGE THIS: 30 means parameters stay within +/-30%
costGoal = 1e-8;
rng('shuffle');              % different random starts each time

opts = detectImportOptions(csvFile, 'TreatAsMissing', {'#DIV/0!'});
T = readtable(csvFile, opts);

for j = 1:width(T)
    vname = T.Properties.VariableNames{j};
    if iscell(T{:,j}) || isstring(T{:,j}) || ischar(T{1,j})
        T.(vname) = str2double(string(T{:,j}));
    end
end

timeAll = T{:,1};

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

global ODE_TOL
ODE_TOL = 1e-8;

[pars, Init] = load_pars_Init_Copeland_Edited(data);
pars = pars(:);
base_pars = pars;

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

parIndexTable = table((1:numel(pars))', par_names(:), pars(:), ...
    'VariableNames', {'Index','Parameter','StartingValue'});

% Parameters to fit
fit_idx = [76 9 91 98 100 93 39 102 95 7 96 77 89 8 36];
fit_idx = fit_idx(:);

fitTable = parIndexTable(fit_idx,:);
disp(fitTable);

% Data used in the objective function
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

% Use only CSV/data times in the cost function. The dense tspan above is only for plotting.
fitTimes = unique(timeAll(isfinite(timeAll)), 'sorted');
costTspan = unique([0; fitTimes(:); runningTime], 'sorted');

% Convert perturbPercent to fraction. Example: 30 -> 0.30.
if perturbPercent > 1
    perturbFraction = perturbPercent/100;
else
    perturbFraction = perturbPercent;
end
perturbFraction = abs(perturbFraction);

if perturbFraction >= 1
    error('perturbPercent must be less than 100%%.');
end

initialTheta = zeros(numel(fit_idx),1);   % zero means original/base parameters
initialCost = cost_function_2(initialTheta, fit_idx, base_pars, Init, fitData, costTspan, useLogFit, perturbPercent);
fprintf('\nInitial cost using original parameters = %.6g\n', initialCost);

[t_base, sol_base] = modelDriver(base_pars, Init, tspan);

obj = @(theta) cost_function_2(theta, fit_idx, base_pars, Init, fitData, costTspan, useLogFit, perturbPercent);

options = optimset('Display','iter', ...
                   'MaxIter',maxIter, ...
                   'MaxFunEvals',maxFunEvals, ...
                   'TolX',1e-12, ...
                   'TolFun',1e-12, ...
                   'OutputFcn', @(x,optimValues,state) stopIfCostSmall(x,optimValues,state,costGoal));

nfit = numel(fit_idx);
bestCostOverall = Inf;
bestThetaOverallRaw = [];
bestExitflag = NaN;
bestOutput = struct();
bestStartPercentChange = [];
bestStartMultiplier = [];
startSummary = table();

for s = 1:nStarts
    fprintf('\n================ RANDOM START %d / %d ================\n', s, nStarts);

    % Random starting percent changes inside [-perturbFraction, +perturbFraction].
    % Example with perturbPercent = 30: theta0 values are between -0.30 and +0.30.
    theta0 = -perturbFraction + 2*perturbFraction*rand(nfit,1);
    startPercentChange = 100*theta0;
    startMultiplier = 1 + theta0;

    fprintf('Allowed perturbation: +/- %.2f%%\n', 100*perturbFraction);
    fprintf('Starting percent-change range: %.4g%% to %.4g%%\n', min(startPercentChange), max(startPercentChange));
    fprintf('Starting multiplier range: %.4g to %.4g\n', min(startMultiplier), max(startMultiplier));
    fprintf('Starting cost = %.6g\n', obj(theta0));

    [theta_try_raw, cost_try, exitflag_try, output_try] = fminsearch(obj, theta0, options);

    fprintf('Random start %d finished with cost %.6g, exitflag %d\n', s, cost_try, exitflag_try);

    startSummary = [startSummary; table(s, cost_try, exitflag_try, output_try.iterations, output_try.funcCount, ...
        min(startPercentChange), max(startPercentChange), ...
        'VariableNames', {'Start','BestCost','Exitflag','Iterations','FuncCount','StartMinPercent','StartMaxPercent'})]; %#ok<AGROW>

    if cost_try < bestCostOverall
        bestCostOverall = cost_try;
        bestThetaOverallRaw = theta_try_raw;
        bestExitflag = exitflag_try;
        bestOutput = output_try;
        bestStartPercentChange = startPercentChange;
        bestStartMultiplier = startMultiplier;
    end
end

theta_hat_raw = bestThetaOverallRaw;
theta_hat = clampPercentChange(theta_hat_raw, perturbFraction);   % final used percent changes
bestCost = bestCostOverall;
exitflag = bestExitflag;
output = bestOutput;

disp('---------------- RANDOM START SUMMARY ----------------');
disp(startSummary);
fprintf('Best overall cost = %.6g\n', bestCost);

new_pars = base_pars;
multipliers = 1 + theta_hat(:);
new_pars(fit_idx) = base_pars(fit_idx).*multipliers;

idx = fit_idx(:);

paramNames = par_names(idx);
paramNames = paramNames(:);

oldVals = base_pars(idx);
oldVals = oldVals(:);

newVals = new_pars(idx);
newVals = newVals(:);

absoluteChange = newVals - oldVals;
percentChange = 100*theta_hat(:);
boundHit = abs(theta_hat(:)) >= (perturbFraction - 1e-12);

changedParams = table( ...
    idx, ...
    paramNames, ...
    oldVals, ...
    newVals, ...
    absoluteChange, ...
    percentChange, ...
    multipliers(:), ...
    boundHit, ...
    'VariableNames', {'Index','Parameter','OldValue','NewValue','AbsoluteChange','PercentChange','Multiplier','HitPercentBound'} ...
);

disp('---------------- FITTING RESULTS ----------------');
fprintf('Best cost = %.6g\n', bestCost);
fprintf('exitflag = %d\n', exitflag);
disp(output);
disp(changedParams);

save('fit_results.mat', 'base_pars', 'new_pars', 'fit_idx', 'par_names', ...
    'theta_hat', 'theta_hat_raw', 'perturbPercent', 'bestCost', 'exitflag', ...
    'output', 'changedParams', 'startSummary', 'bestStartPercentChange', 'bestStartMultiplier');
writetable(changedParams, 'fit_changed_parameters.csv');
writetable(startSummary, 'fit_random_start_summary.csv');

[t_fit, sol_fit] = modelDriver(new_pars, Init, tspan);

plotFields = {'hr','temp','tnf','il6','il8'};
plotTitles = {'HR','Temperature','TNF','IL-6','IL-8'};

fig = figure('Units','normalized','OuterPosition',[0 0 0.95 0.85], ...
             'Name','Fitted Model vs Data');
tl = tiledlayout(2,3,'TileSpacing','compact','Padding','compact');
title(tl, sprintf('Fitted model vs CSV data | cost = %.4g', bestCost));

for k = 1:numel(plotFields)
    obs = fitData.(plotFields{k});
    ax = nexttile(tl,k);
    hold(ax,'on');
    grid(ax,'on');

    plot(ax, t_base, sol_base(:,obs.state), '--', 'LineWidth', 1.5, 'DisplayName','old model');
    plot(ax, t_fit,  sol_fit(:,obs.state),  '-',  'LineWidth', 2.0, 'DisplayName','new fitted model');

    validMean = isfinite(obs.time(:)) & isfinite(obs.mean(:));
    validErr  = validMean & isfinite(obs.sd(:)) & obs.sd(:) > 0;
    validOnly = validMean & ~validErr;

    if any(validErr)
        errorbar(ax, obs.time(validErr), obs.mean(validErr), obs.sd(validErr), ...
            'o', 'LineStyle','none', 'MarkerSize',6, 'CapSize',0, ...
            'DisplayName','data +/- sd');
    end

    if any(validOnly)
        plot(ax, obs.time(validOnly), obs.mean(validOnly), ...
            'o', 'LineStyle','none', 'MarkerSize',6, 'DisplayName','data');
    end

    xlabel(ax,'Time (hr)');
    ylabel(ax,plotTitles{k});
    title(ax,plotTitles{k});
    xlim(ax,[tspan(1), tspan(end)]);
    legend(ax,'Location','best');
end

% Empty 6th tile for summary text
ax = nexttile(tl,6);
axis(ax,'off');
summaryText = sprintf('Initial cost: %.4g\nBest cost: %.4g\nFitted parameters: %d\nRandom starts: %d\nAllowed change: +/- %.1f%%\nuseLogFit: %d', ...
    initialCost, bestCost, numel(fit_idx), nStarts, 100*perturbFraction, useLogFit);
text(ax,0.05,0.8,summaryText,'FontSize',12,'VerticalAlignment','top');

saveas(fig, 'fit_model_vs_data.fig');
try
    exportgraphics(fig, 'fit_model_vs_data.pdf', 'ContentType','vector');
    exportgraphics(fig, 'fit_model_vs_data.png', 'Resolution',300);
catch
    saveas(fig, 'fit_model_vs_data.png');
end

fprintf('\nSaved files:\n');
fprintf('  fit_results.mat\n');
fprintf('  fit_changed_parameters.csv\n');
fprintf('  fit_random_start_summary.csv\n');
fprintf('  fit_model_vs_data.fig\n');
fprintf('  fit_model_vs_data.pdf\n');
fprintf('  fit_model_vs_data.png\n');

function theta = clampPercentChange(theta, perturbFraction)
    theta = theta(:);
    theta = max(min(theta, perturbFraction), -perturbFraction);
end

function stop = stopIfCostSmall(x, optimValues, state, costGoal)
    %#ok<INUSD>
    stop = false;

    if strcmp(state,'iter') && optimValues.fval <= costGoal
        fprintf('\nStopping because cost %.6g is below cost goal %.6g\n', optimValues.fval, costGoal);
        stop = true;
    end
end
