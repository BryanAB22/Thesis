clear; clc; close all;

% -------------------------------------------------------------------------
% compute_basin.m
%
% Basin scan for kpg versus initial pathogen level Pe(0).
%
% This version is fixed to use Survivor_Data.csv instead of DataCopeland.mat.
% It uses the same model files as your current project:
%   - load_pars_Init_Copeland_Edited.m
%   - modelDriver.m
%   - model.m
%
% IMPORTANT:
% If MATLAB shows "Undefined function", make sure these files are named
% exactly as above. For example:
%   model(3).m                    -> model.m
%   modelDriver(2).m              -> modelDriver.m
%   load_pars_Init_Copeland_Edited(3).m -> load_pars_Init_Copeland_Edited.m
% -------------------------------------------------------------------------

global Init

% -------------------------------------------------------------------------
% Check that required model files are visible to MATLAB
% -------------------------------------------------------------------------
if exist('load_pars_Init_Copeland_Edited', 'file') ~= 2
    error(['Cannot find load_pars_Init_Copeland_Edited.m. ', ...
           'Make sure it is in the same folder or on the MATLAB path.']);
end

if exist('modelDriver', 'file') ~= 2
    error(['Cannot find modelDriver.m. ', ...
           'Make sure it is in the same folder or on the MATLAB path.']);
end

if exist('model', 'file') ~= 2
    error(['Cannot find model.m. ', ...
           'Make sure it is in the same folder or on the MATLAB path.']);
end

% -------------------------------------------------------------------------
% Load Survivor_Data.csv
% -------------------------------------------------------------------------
thisFile = mfilename('fullpath');

if isempty(thisFile)
    baseDir = pwd;
else
    baseDir = fileparts(thisFile);
end

csvFile = fullfile(baseDir, 'Survivor_Data.csv');

if ~isfile(csvFile)
    error('Could not find Survivor_Data.csv in this folder: %s', baseDir);
end

opts = detectImportOptions(csvFile, ...
    'TreatAsMissing', {'#DIV/0!', 'NaN', 'nan', ''});

T = readtable(csvFile, opts);

% Convert any text/string/cell columns to numeric.
for j = 1:width(T)
    varName = T.Properties.VariableNames{j};
    rawVar  = T.(varName);

    if iscell(rawVar) || isstring(rawVar) || ischar(rawVar) || iscategorical(rawVar)
        T.(varName) = str2double(string(rawVar));
    end
end

requiredCols = {'Time', 'HR_Mean', 'TEMP_Mean', 'TNF_Mean', ...
                'IL_6_Mean', 'IL_8_Mean'};

missingCols = setdiff(requiredCols, T.Properties.VariableNames);

if ~isempty(missingCols)
    error('The CSV is missing these required columns: %s', strjoin(missingCols, ', '));
end

% -------------------------------------------------------------------------
% Build the data structure expected by load_pars_Init_Copeland_Edited
% -------------------------------------------------------------------------
data = struct();

data.time = T.Time(:);
data.hr   = T.HR_Mean(:);
data.temp = T.TEMP_Mean(:);
data.TNF  = T.TNF_Mean(:);
data.IL6  = T.IL_6_Mean(:);
data.IL8  = T.IL_8_Mean(:);

% These are included only for completeness. Your current parameter loader
% mostly uses hard-coded baseline values, but this keeps the structure clean.
data.age    = 29;
data.weight = 79.9;
data.height = 177;
data.HM     = 207 - 0.7 * data.age;

% -------------------------------------------------------------------------
% Load baseline parameters and initial conditions
% -------------------------------------------------------------------------
[pars, Init] = load_pars_Init_Copeland_Edited(data);

pars0 = pars;
Init0 = Init;

% -------------------------------------------------------------------------
% Basin settings
% -------------------------------------------------------------------------
runningTime = 5000;
time = [0 runningTime];

% Parameter/state indexes based on your model:
% pars(89) = kpg, state y(5) = Ma, y(7) = Pe, y(17) = Damage.
kpg_idx = 89;

Ma_idx     = 5;
Pe_idx     = 7;
Damage_idx = 17;

threshold = 1e-3;

% Grid resolution.
% Use 25 first for a fast test, then increase to 100 for final results.
nP0  = 100;
nKpg = 100;

P0_grid  = linspace(0, 3.0, nP0);
kpg_grid = linspace(0, 3.5, nKpg);

% Labels:
%  1 = recovered / cleared: Ma, Pe, Damage all small
%  2 = pathogen cleared but inflammation/damage remains
%  3 = persistent pathogen with inflammation/damage
%  4 = other bounded final state
% -1 = solver failed
% -2 = solver stopped before final time
% -3 = NaN/Inf in solution
label = zeros(numel(P0_grid), numel(kpg_grid));

fprintf('Starting basin scan: %d simulations\n', numel(P0_grid) * numel(kpg_grid));
fprintf('Running time per simulation: t = 0 to %g\n', runningTime);
tic

% -------------------------------------------------------------------------
% kpg versus Pe(0)
% -------------------------------------------------------------------------
for ik = 1:numel(P0_grid)

    Init_base = Init0;
    Init_base(Pe_idx) = P0_grid(ik);

    for ip = 1:numel(kpg_grid)

        pars_test = pars0;
        pars_test(kpg_idx) = kpg_grid(ip);

        try
            [t, y] = modelDriver(pars_test, Init_base, time);

            if isempty(t) || t(end) < time(end) - 1e-9
                label(ik, ip) = -2;
                continue
            end

            if isempty(y) || any(isnan(y(:))) || any(isinf(y(:)))
                label(ik, ip) = -3;
                continue
            end

            Ma_final     = y(end, Ma_idx);
            Pe_final     = y(end, Pe_idx);
            Damage_final = y(end, Damage_idx);

            if (Ma_final <= threshold) && ...
               (Pe_final <= threshold) && ...
               (Damage_final <= threshold)

                label(ik, ip) = 1;

            elseif (Pe_final <= threshold) && ...
                   (Ma_final > threshold) && ...
                   (Damage_final > threshold)

                label(ik, ip) = 2;

            elseif (Pe_final > threshold) && ...
                   (Ma_final > threshold) && ...
                   (Damage_final > threshold)

                label(ik, ip) = 3;

            else
                label(ik, ip) = 4;
            end

        catch ME
            label(ik, ip) = -1;

            % Uncomment this if you want to see every solver error.
            % fprintf('Failed at Pe0=%g, kpg=%g: %s\n', ...
            %     P0_grid(ik), kpg_grid(ip), ME.message);
        end
    end

    if mod(ik, 10) == 0 || ik == numel(P0_grid)
        fprintf('Finished %d of %d Pe(0) rows\n', ik, numel(P0_grid));
    end
end

elapsedTime = toc;
fprintf('Basin scan finished in %.2f seconds\n', elapsedTime);

% -------------------------------------------------------------------------
% Save numerical results
% -------------------------------------------------------------------------
save(fullfile(baseDir, 'basin_kpg_vs_Pe0_results.mat'), ...
    'P0_grid', 'kpg_grid', 'label', 'threshold', 'runningTime');

% -------------------------------------------------------------------------
% Plot basin
% -------------------------------------------------------------------------
figure('Color', 'w');
imagesc(kpg_grid, P0_grid, label);
set(gca, 'YDir', 'normal');

xlabel('k_{pg}', 'Interpreter', 'tex');
ylabel('Pe(0)', 'Interpreter', 'tex');
title('Basin scan: k_{pg} versus Pe(0)', 'Interpreter', 'tex');

cb = colorbar;
ylabel(cb, 'Final-state label');

grid on;
box on;

% Optional: keep all label categories visible on the color scale.
caxis([-3 4]);

% Save figure
saveas(gcf, fullfile(baseDir, 'basin_kpg_vs_Pe0.png'));
