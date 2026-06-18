%% compare_de_results_vs_baseline.m
% Loads fit_bayesopt_results.mat, runs the baseline model and the fitted
% DE model, and plots both model curves against the observed data.
%
% Put this file in the same folder as:
%   fit_bayesopt_results.mat
%   Survivor_Data.csv or Survivor_Median.csv
%   model.m
%   load_pars_Init_Copeland_Edited.m
%
% Then run:
%   compare_bayesopt_results_vs_baseline

close all;
clear;
clc;

%% ================= USER SETTINGS =================
matFile = 'fit_results.mat';

% Leave as 'auto' to use Survivor_Data.csv first, then Survivor_Median.csv.
csvFile = 'auto';

runningTime = 100;
tspan = 0:0.1:runningTime;

SAVE_COMBINED_FIGURE = false;
SAVE_INDIVIDUAL_PLOTS = false;
outputFolder = 'de_vs_baseline_plots';

SHOW_LEGEND_IN_COMBINED = false;
SHOW_LEGEND_IN_INDIVIDUAL = true;
legendLocation = 'best';

% Plot these ODE states.
% Important model state numbers:
% 1 = TNF, 3 = IL-8/CXCL8, 4 = IL-6, 7 = pathogen PE,
% 8 = temperature, 14 = HR.
modelStateNums = [1 3 4 5 7 8 14 17];

stateNames = { ...
    'TNF', ...                       % state 1
    'IL-8 / CXCL8', ...              % state 3
    'IL-6', ...                      % state 4
    'Activated macrophages, MA', ... % state 5
    'Pathogen, PE', ...              % state 7
    'Temperature', ...               % state 8
    'Heart rate, HR', ...            % state 14
    'Damage, D' ...                  % state 17
};

stateYLabels = { ...
    'TNF', ...
    'IL-8 / CXCL8', ...
    'IL-6', ...
    'MA', ...
    'PE', ...
    'Temperature', ...
    'HR', ...
    'Damage' ...
};

%% ================= BASIC FILE CHECKS =================
if ~isfile(matFile)
    error('Cannot find %s. Put this script in the same folder as fit_bayesopt_results.mat.', matFile);
end

if ~isfile('model.m')
    error('Cannot find model.m. Rename your model file to model.m and put it in this folder.');
end

if ~isfile('load_pars_Init_Copeland_Edited.m')
    error('Cannot find load_pars_Init_Copeland_Edited.m. Put it in this folder.');
end

csvFile = chooseCsvFile(csvFile);
fprintf('\nUsing CSV file: %s\n', csvFile);
fprintf('Using MAT file: %s\n', matFile);

%% ================= LOAD DATA =================
global ODE_TOL
ODE_TOL = 1e-6;

[fitData, data, obsByState] = readSurvivorDataFlexible(csvFile);

%% ================= LOAD BASELINE VALUES =================
[base_pars, base_Init] = load_pars_Init_Copeland_Edited(data);
base_pars = base_pars(:);
base_Init = base_Init(:);

fprintf('\nBaseline parameter count: %d\n', numel(base_pars));
fprintf('Baseline initial-condition count: %d\n', numel(base_Init));

%% ================= LOAD FITTED VALUES FROM MAT FILE =================
R = load(matFile);
[selected_pars, selected_Init] = getBayesoptValuesFromMat(R, base_pars, base_Init);

fprintf('\nLoaded DE fitted values.\n');

if isfield(R, 'initialCost')
    fprintf('Initial cost saved in MAT file: %.6g\n', R.initialCost);
end
if isfield(R, 'bestCost')
    fprintf('Best cost saved in MAT file: %.6g\n', R.bestCost);
end
if isfield(R, 'methodUsed')
    fprintf('Method saved in MAT file: %s\n', string(R.methodUsed));
end

if isfield(R, 'changedParams')
    disp('---------------- CHANGED PARAMETERS FROM MAT FILE ----------------');
    disp(R.changedParams);
end

if isfield(R, 'changedInitialConditions')
    disp('---------------- CHANGED INITIAL CONDITIONS FROM MAT FILE ----------------');
    disp(R.changedInitialConditions);
end

%% ================= RUN BOTH MODELS =================
fprintf('\nRunning baseline model...\n');
[t_base, sol_base] = runModelAtTimes(base_pars, base_Init, tspan);

fprintf('Running DE fitted model...\n');
[t_fit, sol_fit] = runModelAtTimes(selected_pars, selected_Init, tspan);

maxStateNum = max(modelStateNums);
if size(sol_base,2) < maxStateNum
    error('Baseline solution has %d states, but state %d was requested.', size(sol_base,2), maxStateNum);
end
if size(sol_fit,2) < maxStateNum
    error('Fitted solution has %d states, but state %d was requested.', size(sol_fit,2), maxStateNum);
end

%% ================= COMBINED FIGURE =================
fig = figure('Units','normalized', ...
             'OuterPosition',[0 0 1 0.95], ...
             'Name','DE fitted model vs baseline model');

nPlots = numel(modelStateNums);
nCols = 4;
nRows = ceil((nPlots + 1) / nCols);

tl = tiledlayout(nRows, nCols, ...
    'TileSpacing','compact', ...
    'Padding','compact');

title(tl, 'DE fitted model vs baseline model');

for plotIdx = 1:nPlots
    stateNum = modelStateNums(plotIdx);
    ax = nexttile(tl, plotIdx);

    plotOneState(ax, plotIdx, stateNum, stateNames, stateYLabels, ...
        t_base, sol_base, t_fit, sol_fit, obsByState, tspan, ...
        legendLocation, SHOW_LEGEND_IN_COMBINED);
end

% Summary tile
ax = nexttile(tl, nPlots + 1);
axis(ax, 'off');
summaryText = buildSummaryText(R, matFile, csvFile);
text(ax, 0.05, 0.95, summaryText, ...
    'FontSize', 10, ...
    'VerticalAlignment', 'top', ...
    'Interpreter', 'none');

% Turn off unused tiles
for emptyTile = (nPlots + 2):(nRows * nCols)
    ax = nexttile(tl, emptyTile);
    axis(ax, 'off');
end

%% ================= OPTIONAL SAVE COMBINED FIGURE =================
if SAVE_COMBINED_FIGURE
    if ~exist(outputFolder, 'dir')
        mkdir(outputFolder);
    end

    combinedBase = fullfile(outputFolder, 'de_vs_baseline_combined');
    saveas(fig, [combinedBase '.fig']);

    try
        exportgraphics(fig, [combinedBase '.png'], 'Resolution', 300);
        exportgraphics(fig, [combinedBase '.pdf'], 'ContentType', 'vector');
    catch
        saveas(fig, [combinedBase '.png']);
    end

    fprintf('\nSaved combined figure to folder: %s\n', outputFolder);
end

%% ================= OPTIONAL INDIVIDUAL FIGURES =================
if SAVE_INDIVIDUAL_PLOTS
    if ~exist(outputFolder, 'dir')
        mkdir(outputFolder);
    end

    for plotIdx = 1:nPlots
        stateNum = modelStateNums(plotIdx);

        figOne = figure('Name', sprintf('State %02d: %s', stateNum, stateNames{plotIdx}));
        ax = axes('Parent', figOne);

        plotOneState(ax, plotIdx, stateNum, stateNames, stateYLabels, ...
            t_base, sol_base, t_fit, sol_fit, obsByState, tspan, ...
            legendLocation, SHOW_LEGEND_IN_INDIVIDUAL);

        title(ax, sprintf('State %02d: %s', stateNum, stateNames{plotIdx}));

        safeName = regexprep(lower(stateNames{plotIdx}), '[^a-z0-9]+', '_');
        safeName = regexprep(safeName, '^_|_$', '');

        outBase = fullfile(outputFolder, sprintf('state_%02d_%s_de_vs_baseline', stateNum, safeName));
        saveas(figOne, [outBase '.fig']);

        try
            exportgraphics(figOne, [outBase '.png'], 'Resolution', 300);
            exportgraphics(figOne, [outBase '.pdf'], 'ContentType', 'vector');
        catch
            saveas(figOne, [outBase '.png']);
        end
    end

    fprintf('Saved individual figures to folder: %s\n', outputFolder);
end

fprintf('\nDone. Compared DE fitted model against baseline model.\n');

%% ========================================================================
%% LOCAL FUNCTIONS
%% ========================================================================

function csvFile = chooseCsvFile(csvFile)
    if ~strcmpi(csvFile, 'auto')
        if ~isfile(csvFile)
            error('Cannot find requested csvFile: %s', csvFile);
        end
        return
    end

    candidates = {'Survivor_Data.csv', 'Survivor_Median.csv'};
    for k = 1:numel(candidates)
        if isfile(candidates{k})
            csvFile = candidates{k};
            return
        end
    end

    error('Could not find Survivor_Data.csv or Survivor_Median.csv. Put one of them in this folder.');
end

function [fitData, data, obsByState] = readSurvivorDataFlexible(csvFile)
    opts = detectImportOptions(csvFile, 'TreatAsMissing', {'#DIV/0!', 'NaN', 'nan', ''});
    T = readtable(csvFile, opts);

    % Convert text/string/cell columns to numeric when possible.
    for j = 1:width(T)
        vname = T.Properties.VariableNames{j};
        col = T.(vname);

        if iscell(col) || isstring(col) || ischar(col)
            converted = str2double(string(col));
            if any(isfinite(converted))
                T.(vname) = converted;
            end
        end
    end

    timeAll = getNumericColumn(T, {'Time','time','TIME','Hour','Hours','hours'});

    HR     = getNumericColumn(T, {'HR_Mean','HR_Median','HR'});
    HRsd   = getNumericColumn(T, {'HR_SD','HR_sd','HR_STD','HR_Std'});

    TEMP   = getNumericColumn(T, {'TEMP_Mean','TEMP_Median','Temp_Mean','Temp_Median','TEMP','Temp'});
    TEMPsd = getNumericColumn(T, {'TEMP_SD','Temp_SD','TEMP_sd','TEMP_STD','Temp_Std'});

    TNF    = getNumericColumn(T, {'TNF_Mean','TNF_Median','TNF'});
    TNFsd  = getNumericColumn(T, {'TNF_SD','TNF_sd','TNF_STD','TNF_Std'});

    IL6    = getNumericColumn(T, {'IL_6_Mean','IL_6_Median','IL6_Mean','IL6_Median','IL6','IL_6'});
    IL6sd  = getNumericColumn(T, {'IL_6_SD','IL6_SD','IL_6_sd','IL6_sd','IL_6_STD','IL6_STD'});

    IL8    = getNumericColumn(T, {'IL_8_Mean','IL_8_Median','IL8_Mean','IL8_Median','IL8','IL_8'});
    IL8sd  = getNumericColumn(T, {'IL_8_SD','IL8_SD','IL_8_sd','IL8_sd','IL_8_STD','IL8_STD'});

    data = struct();
    data.hr   = HR;
    data.TNF  = TNF;
    data.IL6  = IL6;
    data.IL8  = IL8;
    data.temp = TEMP;

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

    % Cell array indexed by actual model state number.
    obsByState = cell(17, 1);
    obsByState{1}  = fitData.tnf;
    obsByState{3}  = fitData.il8;
    obsByState{4}  = fitData.il6;
    obsByState{8}  = fitData.temp;
    obsByState{14} = fitData.hr;
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

    % Try a looser normalized-name match.
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

function [selected_pars, selected_Init] = getBayesoptValuesFromMat(R, base_pars, base_Init)
    selected_pars = base_pars;
    selected_Init = base_Init;

    % Preferred path: use full fitted parameter vector saved by the new main script.
    if isfield(R, 'new_pars') && numel(R.new_pars) == numel(base_pars)
        selected_pars = R.new_pars(:);
    elseif isfield(R, 'fit_idx') && isfield(R, 'theta_hat')
        fit_idx = R.fit_idx(:);
        theta_hat = R.theta_hat(:);
        nParFit = numel(fit_idx);

        if numel(theta_hat) < nParFit
            error('theta_hat has fewer entries than fit_idx. Cannot reconstruct fitted parameters.');
        end

        theta_par = theta_hat(1:nParFit);
        selected_pars(fit_idx) = base_pars(fit_idx) .* (1 + theta_par(:));
    else
        error('MAT file must contain either new_pars or both fit_idx and theta_hat.');
    end

    % Preferred path: use full fitted initial condition vector saved by the new main script.
    if isfield(R, 'new_Init') && numel(R.new_Init) == numel(base_Init)
        selected_Init = R.new_Init(:);
    elseif isfield(R, 'init_idx') && isfield(R, 'theta_hat') && isfield(R, 'fit_idx')
        init_idx = R.init_idx(:);
        theta_hat = R.theta_hat(:);
        nParFit = numel(R.fit_idx(:));
        theta_init = theta_hat(nParFit+1:end);

        if numel(init_idx) ~= numel(theta_init)
            error('init_idx has %d entries, but theta_init has %d entries.', numel(init_idx), numel(theta_init));
        end

        selected_Init(init_idx) = base_Init(init_idx) .* (1 + theta_init(:));
    else
        fprintf('MAT file does not contain new_Init/init_idx. Using baseline initial conditions.\n');
    end
end

function [t, sol] = runModelAtTimes(pars, Init, tspan)
    global ODE_TOL

    if isempty(ODE_TOL)
        ODE_TOL = 1e-6;
    end

    options = odeset('RelTol', ODE_TOL, ...
                     'AbsTol', ODE_TOL, ...
                     'NonNegative', 1:numel(Init));

    [t, sol] = ode23s(@(tt, yy) model(tt, yy, pars), tspan, Init, options);
end

function plotOneState(ax, plotIdx, stateNum, stateNames, stateYLabels, ...
    t_base, sol_base, t_fit, sol_fit, obsByState, tspan, legendLocation, showLegend)

    hold(ax, 'on');
    grid(ax, 'on');
    box(ax, 'on');

    plot(ax, t_base, sol_base(:, stateNum), '--', ...
        'LineWidth', 1.3, ...
        'DisplayName', 'baseline model');

    plot(ax, t_fit, sol_fit(:, stateNum), '-', ...
        'LineWidth', 1.8, ...
        'DisplayName', 'DE fitted model');

    obs = [];
    if stateNum <= numel(obsByState)
        obs = obsByState{stateNum};
    end

    if ~isempty(obs)
        plotObservedData(ax, obs);
    end

    xlabel(ax, 'Time');
    ylabel(ax, stateYLabels{plotIdx});
    title(ax, sprintf('%02d: %s', stateNum, stateNames{plotIdx}));
    xlim(ax, [tspan(1), tspan(end)]);
    set(ax, 'FontSize', 8);

    if showLegend
        legend(ax, 'Location', legendLocation, 'Interpreter', 'none');
    else
        legend(ax, 'off');
    end
end

function plotObservedData(ax, obs)
    time = obs.time(:);
    meanVal = obs.mean(:);
    sdVal = obs.sd(:);

    validMean = isfinite(time) & isfinite(meanVal);
    validErr  = validMean & isfinite(sdVal) & sdVal > 0;
    validOnly = validMean & ~validErr;

    if any(validErr)
        errorbar(ax, time(validErr), meanVal(validErr), sdVal(validErr), ...
            'o', ...
            'MarkerSize', 4, ...
            'LineWidth', 0.8, ...
            'DisplayName', [obs.name ' data +/- SD']);
    end

    if any(validOnly)
        plot(ax, time(validOnly), meanVal(validOnly), 'o', ...
            'MarkerSize', 4, ...
            'DisplayName', [obs.name ' data']);
    end
end

function summaryText = buildSummaryText(R, matFile, csvFile)
    lines = {};
    lines{end+1} = sprintf('MAT file: %s', matFile);
    lines{end+1} = sprintf('CSV file: %s', csvFile);

    if isfield(R, 'initialCost')
        lines{end+1} = sprintf('Initial cost: %.4g', R.initialCost);
    end
    if isfield(R, 'bestCost')
        lines{end+1} = sprintf('Best cost: %.4g', R.bestCost);
    end
    if isfield(R, 'perturbPercent')
        lines{end+1} = sprintf('Perturb bound: +/- %.1f%%', R.perturbPercent);
    end
    if isfield(R, 'methodUsed')
        lines{end+1} = sprintf('Method: %s', string(R.methodUsed));
    end
    if isfield(R, 'new_pars')
        lines{end+1} = 'Loaded fitted parameters: new_pars';
    else
        lines{end+1} = 'Reconstructed parameters from theta_hat';
    end
    if isfield(R, 'new_Init')
        lines{end+1} = 'Loaded fitted initial conditions: new_Init';
    elseif isfield(R, 'init_idx')
        lines{end+1} = 'Reconstructed initial conditions from theta_hat';
    else
        lines{end+1} = 'Initial conditions: baseline values';
    end

    summaryText = strjoin(lines, newline);
end
