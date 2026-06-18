close all;
clc;

global Init

idx   = [1 3 4 7 8 14 ];

names = {'tnf','Il08','Il06', ...
    'Pathogens','Temperature','Heart Rate'};

opts = detectImportOptions('placebo_plotted_data(Survivor_Data).csv', ...
    'TreatAsMissing', {'#DIV/0!'});

T = readtable('placebo_plotted_data(Survivor_Data).csv', opts);

for j = 1:width(T)
    if iscell(T{:,j}) || isstring(T{:,j}) || ischar(T{1,j})
        T.(j) = str2double(string(T{:,j}));
    end
end

% Time column
timeAll = T{:,1};

% Extract means and standard deviations from the CSV
HRtime   = timeAll;
HR       = T.HR_Mean;
HRsd     = T.HR_SD;

TEMPtime = timeAll;
TEMP     = T.TEMP_Mean;
TEMPsd   = T.TEMP_SD;

IMMUNEtime = timeAll;
TNF     = T.TNF_Mean;
TNFsd   = T.TNF_SD;

IL6     = T.IL_6_Mean;
IL6sd   = T.IL_6_SD;

IL8     = T.IL_8_Mean;
IL8sd   = T.IL_8_SD;

% Cardiac output data from CSV
COtime  = timeAll;
CO      = T.CO_Mean;
COsd    = T.CO_SD;

% Final simulation time
runningTime = 700;
tspan = 0:0.1:runningTime;

% Build measured-data structure for initializing the model
data = struct();
data.hr   = HR;
data.TNF  = TNF;
data.IL6  = IL6;
data.IL8  = IL8;
data.temp = TEMP;

% Load parameters and initial conditions
[pars, Init] = load_pars_Init_Copeland_Fitted(data);

% Solve the model
[t, sol] = modelDriver(pars, Init, tspan);

% Compute cardiac output Q from solved states
Cla = pars(70);
Clv = pars(72);
Em  = pars(74);
EM  = pars(75);

Vla = sol(:,10);
Vlv = sol(:,12);
hr  = sol(:,14);

pla  = Vla ./ Cla;
plv  = Vlv ./ Clv;
Vstr = -(pla./EM - plv./Em);
Q    = ((1/6)*Vstr .* hr) / 60;

% Add CO (= Q) as an extra column for plotting
sol_plot = [sol, Q, pla];

% Create figure
fig = figure('Units','normalized','OuterPosition',[0 0 0.92 0.9], ...
             'Name', 'Model vs Measured Data');

nTiles = numel(idx);
nRows  = ceil(sqrt(nTiles));
nCols  = ceil(nTiles / nRows);

tl = tiledlayout(nRows, nCols, ...
    'TileSpacing','compact', ...
    'Padding','compact');

title(tl, sprintf('Healthy Trajectories'), ...
      'Interpreter','tex');

% Store handles for one shared legend only
hModel = [];
hErr   = [];
hOnly  = [];

for k = 1:nTiles

    ax = nexttile(tl, k);
    hold(ax, 'on');
    grid(ax, 'on');

    % Model solution for the current variable
    yk = sol_plot(:, idx(k));

    p = plot(ax, t, yk, ...
        'LineWidth', 2, ...
        'DisplayName', 'model');

    if isempty(hModel)
        hModel = p;
    end

    % Match model variable name to measured data
    key = lower(regexprep(names{k}, '\s+', ''));

    tdat = [];
    ydat = [];
    sdat = [];

    switch key

        case {'hr','heartrate'}
            tdat = HRtime(:);
            ydat = HR(:);
            sdat = HRsd(:);

        case {'temp','temperature'}
            tdat = TEMPtime(:);
            ydat = TEMP(:);
            sdat = TEMPsd(:);

        case {'tnf'}
            tdat = IMMUNEtime(:);
            ydat = TNF(:);
            sdat = TNFsd(:);

        case {'il6','il-6','il06'}
            tdat = IMMUNEtime(:);
            ydat = IL6(:);
            sdat = IL6sd(:);

        case {'il8','il-8','il08'}
            tdat = IMMUNEtime(:);
            ydat = IL8(:);
            sdat = IL8sd(:);

        case {'q','co','cardiacoutput'}
            tdat = COtime(:);
            ydat = CO(:);
            sdat = COsd(:);
    end

    % Plot measured data
    if ~isempty(tdat)

        % Valid mean points
        mMean = isfinite(tdat) & isfinite(ydat);

        % Valid mean and SD points
        mErr  = mMean & isfinite(sdat);

        % Valid mean but missing SD
        mOnly = mMean & ~isfinite(sdat);

        % Plot error bars where SD exists
        if any(mErr)
            e = errorbar(ax, tdat(mErr), ydat(mErr), sdat(mErr), ...
                'o', ...
                'LineStyle', 'none', ...
                'MarkerSize', 6, ...
                'CapSize', 0, ...
                'DisplayName', 'measured ± sd');

            if isempty(hErr)
                hErr = e;
            end
        end

        % Plot plain points where SD is missing
        if any(mOnly)
            m = plot(ax, tdat(mOnly), ydat(mOnly), ...
                'o', ...
                'LineStyle', 'none', ...
                'MarkerSize', 6, ...
                'DisplayName', 'measured');

            if isempty(hOnly)
                hOnly = m;
            end
        end
    end

    xlabel(ax, 'Time (hr)');
    ylabel(ax, names{k}, 'Interpreter', 'none');
    xlim(ax, [tspan(1) tspan(end)]);
    set(ax, 'FontSize', 11);
end

% ============================================================
% One shared legend only
% ============================================================

legendHandles = [];
legendLabels  = {};

if ~isempty(hModel)
    legendHandles = [legendHandles, hModel];
    legendLabels{end+1} = 'model';
end

if ~isempty(hErr)
    legendHandles = [legendHandles, hErr];
    legendLabels{end+1} = 'measured ± sd';
end

if ~isempty(hOnly)
    legendHandles = [legendHandles, hOnly];
    legendLabels{end+1} = 'measured';
end

lgd = legend(legendHandles, legendLabels, ...
    'Orientation', 'horizontal', ...
    'Location', 'southoutside');

lgd.Layout.Tile = 'south';