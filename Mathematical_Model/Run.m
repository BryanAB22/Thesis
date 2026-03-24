
close all; 
clear; clc;
clc;

global Init Tm BPo

matFiles = { 'DataCopeland.mat' };


% idx   = 1:17;                    
% names = {'tnf','IL10', 'Il08','IL06',...
%     'MA','MR','Pathogens','Temp',...
%     'Pain','VLA','VSA','VLV',...
%     'VSV','Heart Rate', 'NO','Rs',...
%     'Damage'};  
idx = [ 2 5 7 17];
names={'IL10','MA','Pathogens','Damage'};

% idx   = [1 3 4 7 8 14];  
% names = {'tnf','il-8','il-6','temp','hr'};
% names = {'tnf','il-8','il-6','Patho','temp','hr'};
% idx=14;
% names={'hr'};

assert(numel(idx) == numel(names), 'idx and names must match in length');

runningTime = 500; 
tspan       = 0:0.1:runningTime;

saveFigures = false;
outDir = 'Figures_SelectedFiles_DefaultPars';
if saveFigures && ~exist(outDir,'dir'), mkdir(outDir); end

for f = 1:numel(matFiles)
    fn = matFiles{f};
    if ~isfile(fn)
        warning('Error', fn);
        continue;
    end

    S = load(fn);

    
    BPtime = S.BPt;  BP = S.BPm;  BPsd = S.BPse*sqrt(10);
    HRtime = S.HRt;  HR = S.HRm;  HRsd = S.HRse*sqrt(10);

    IMMUNEtime = S.TNFt;
    TNF = S.TNFm;   TNFsd = S.TNFse*sqrt(10);

    IL6 = S.IL6m;   
    IL6sd = S.IL6se*sqrt(10);
    IL8 = S.IL8m;   
    IL8sd = S.IL8se*sqrt(10);

    TEMPtime = S.TEMPt(1:7);
    TEMP     = S.TEMPm(1:7);
    TEMPsd   = S.TEMPse(1:7) * sqrt(10);

    data = struct();
    data.BP   = BP;
    data.hr   = HR;
    data.TNF  = TNF;
    data.IL6  = IL6;
    data.IL8  = IL8;
    data.temp = TEMP;

    data.age    = 29;
    data.weight = 79.9;
    data.height = 177;
    data.HM     = 207 - 0.7 * data.age;

    BPo = BP(1);           
    Tm  = data.temp(1);    

    [pars, Init] = load_pars_Init_Copeland_Edited(data);

    [t, sol] = modelDriver(pars, Init, tspan);

    fig = figure('Units','normalized','OuterPosition',[0 0 0.92 0.9], ...
                 'Name', sprintf('Single Run (default pars): %s', fn));
    nTiles = numel(idx);
    nRows  = ceil(sqrt(nTiles));
    nCols  = ceil(nTiles / nRows);

    tl = tiledlayout(nRows, nCols, 'TileSpacing','compact','Padding','compact');
    title(tl, sprintf('%s  |  t \\in [%g,%g] ', ...
          fn, tspan(1), tspan(end)), 'Interpreter','tex');

    for k = 1:nTiles
        ax = nexttile(tl, k); hold(ax, 'on'); grid(ax, 'on');

        yk = sol(:, idx(k));
        p = plot(ax, t, yk, 'LineWidth', 2, 'DisplayName','model');

        key = lower(regexprep(names{k}, '\s+', ''));
        tdat = []; ydat = []; sdat = [];

        switch key
            case {'bp','bloodpressure'}
                tdat = BPtime(:); ydat = BP(:); sdat = BPsd(:);
            case {'heartrate','hr'}
                tdat = HRtime(:); ydat = HR(:); sdat = HRsd(:);
            case {'tnf'}
                tdat = IMMUNEtime(:); ydat = TNF(:); sdat = TNFsd(:);
            case {'il6','il-6','il06'}
                tdat = IMMUNEtime(:); ydat = IL6(:); sdat = IL6sd(:);
            case {'il8','il-8','il08'}
                tdat = IMMUNEtime(:); ydat = IL8(:); sdat = IL8sd(:);
            case {'temp','temperature'}
                tdat = TEMPtime(:); ydat = TEMP(:); sdat = TEMPsd(:);
            otherwise
        end

        if ~isempty(tdat) && ~isempty(ydat)
            m = isfinite(tdat) & isfinite(ydat);
            tdat = tdat(m); ydat = ydat(m);
            if ~isempty(sdat) && numel(sdat) >= numel(m)
                sdat = sdat(m);
            else
                sdat = [];
            end

            if ~isempty(sdat) && any(isfinite(sdat))
                errorbar(ax, tdat, ydat, sdat, 'o', 'LineStyle','none', ...
                         'MarkerSize', 6, 'CapSize', 0, 'DisplayName','measured');
            else
                plot(ax, tdat, ydat, 'o', 'LineStyle','none', ...
                     'MarkerSize', 6, 'DisplayName','measured');
            end
            legend(ax, 'Location','best');
        else
            legend(ax, p, 'Location','best');
        end

        xlabel(ax, 'Time (hr)');
        ylabel(ax, names{k}, 'Interpreter','none');
        xlim(ax, [tspan(1) tspan(end)]);
        set(ax, 'FontSize', 11);
    end

    if saveFigures
        base = regexprep(fn, '\.mat$', '');
        outName = fullfile(outDir, sprintf('%graph.png', base));
        saveas(fig, outName);
        fprintf('Saved figure: %s\n', outName);
    end

end

