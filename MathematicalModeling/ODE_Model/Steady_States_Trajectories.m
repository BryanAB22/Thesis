close all;
clear;
clc;

global Init

% ============================================================
% Variables to plot
% ============================================================

idx = [1 2 3 4 5 7 8 14 15 17 18 19];

names = { ...
    'Tnf', ...
    'IL10', ...
    'IL08', ...
    'IL06', ...
    'Macrophages', ...
    'Pathogens', ...
    'Temperature', ...
    'Heart Rate', ...
    'Nitric Oxide', ...
    'Damage', 'Cardiac Output', 'Blood pressure'};

% ============================================================
% Load data only for initializing the model
% ============================================================

opts = detectImportOptions('placebo_plotted_data(Survivor_Data).csv', ...
    'TreatAsMissing', {'#DIV/0!'});

T = readtable('placebo_plotted_data(Survivor_Data).csv', opts);

for j = 1:width(T)
    if iscell(T{:,j}) || isstring(T{:,j}) || ischar(T{1,j})
        T.(j) = str2double(string(T{:,j}));
    end
end

HR   = T.HR_Mean;
TEMP = T.TEMP_Mean;
TNF  = T.TNF_Mean;
IL6  = T.IL_6_Mean;
IL8  = T.IL_8_Mean;

data = struct();
data.hr   = HR;
data.TNF  = TNF;
data.IL6  = IL6;
data.IL8  = IL8;
data.temp = TEMP;

% ============================================================
% Load baseline parameters and initial conditions
% ============================================================

[pars0, Init0] = load_pars_Init_Copeland_Edited(data);

pars0 = pars0(:);
Init0 = Init0(:);

% ============================================================
% Simulation time
% ============================================================

runningTime = 150;
tspan = 0:0.1:runningTime;

% ============================================================
% Scenario settings
% ============================================================

kpg_idx = 89;
Pe_idx  = 7;

scenarios = struct([]);

scenarios(1).name = 'Healthy';
scenarios(1).kpg  = 0.10;
scenarios(1).Pe0  = 0.00;

scenarios(2).name = 'Septic';
scenarios(2).kpg  = 1.00;
scenarios(2).Pe0  = 1.00;

scenarios(3).name = 'Aseptic';
scenarios(3).kpg  = 0.50;
scenarios(3).Pe0  = 2.50;

% Colors and line styles
colors = lines(numel(scenarios));

% Healthy = solid
% Septic  = solid
% Aseptic = dashed
lineStyles = {'-', '-', '--'};

% ============================================================
% Create one figure with all scenarios on each subplot
% ============================================================

figure( ...
    'Units', 'normalized', ...
    'OuterPosition', [0 0 0.92 0.9], ...
    'Name', 'Healthy vs Septic vs Aseptic Trajectories');

nTiles = numel(idx);
nRows  = ceil(sqrt(nTiles));
nCols  = ceil(nTiles / nRows);

tl = tiledlayout(nRows, nCols, ...
    'TileSpacing', 'compact', ...
    'Padding', 'compact');

title(tl, 'Healthy vs Septic vs Aseptic Trajectories', ...
    'Interpreter', 'tex');

axList = gobjects(nTiles, 1);

for k = 1:nTiles

    axList(k) = nexttile(tl, k);
    hold(axList(k), 'on');
    grid(axList(k), 'on');

    xlabel(axList(k), 'Time (hr)');
    ylabel(axList(k), names{k}, 'Interpreter', 'none');
    title(axList(k), names{k}, 'Interpreter', 'none');

    xlim(axList(k), [tspan(1), tspan(end)]);
    set(axList(k), 'FontSize', 11);
end

% This stores one line handle per scenario for the single legend
legendHandles = gobjects(numel(scenarios), 1);

% ============================================================
% Run each scenario and add it to every subplot
% ============================================================

for s = 1:numel(scenarios)

    pars = pars0;
    Init = Init0;

    % Change kpg and Pe_0
    pars(kpg_idx) = scenarios(s).kpg;
    Init(Pe_idx)  = scenarios(s).Pe0;

    % Solve model
    [t, sol] = modelDriver(pars, Init, tspan);

    % ========================================================
    % Compute cardiac output and left atrial pressure
    % ========================================================

    Cla = pars(70);
    Clv = pars(72);
    Em  = pars(74);
    EM  = pars(75);

    Vla = sol(:,10);
    Vlv = sol(:,12);
    hr  = sol(:,14);

    pla = Vla ./ Cla;
    plv = Vlv ./ Clv;

    Vstr = -(pla ./ EM - plv ./ Em);
    Q = ((1/6) * Vstr .* hr) / 60;

    % Add Q and pla as extra variables
    sol_plot = [sol, Q, pla];

    % ========================================================
    % Plot this scenario on every variable subplot
    % ========================================================

    for k = 1:nTiles

        yk = sol_plot(:, idx(k));

        p = plot(axList(k), t, yk, ...
            'LineWidth', 2, ...
            'Color', colors(s,:), ...
            'LineStyle', lineStyles{s}, ...
            'DisplayName', scenarios(s).name);

        % Save only one handle per scenario for the shared legend
        if k == 1
            legendHandles(s) = p;
        end
    end
end

% ============================================================
% One shared legend/key for the whole figure
% ============================================================

legendLabels = {scenarios.name};

lgd = legend(axList(1), legendHandles, legendLabels, ...
    'Orientation', 'horizontal', ...
    'Location', 'southoutside');

lgd.Layout.Tile = 'south';