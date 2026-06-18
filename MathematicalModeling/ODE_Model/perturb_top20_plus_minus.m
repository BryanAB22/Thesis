function [origCounts, summaryTbl, detailTbl] = perturb_top20_plus_minus(csvFile, top_pars, pert_frac, dataCsv, outPrefix, makePlot)
%PERTURB_TOP20_PLUS_MINUS Re-run virtual patients after +/- parameter perturbations.
%
% Usage:
%   [origCounts, summaryTbl, detailTbl] = perturb_top20_plus_minus(csvFile)
%   [origCounts, summaryTbl, detailTbl] = perturb_top20_plus_minus(csvFile, top_pars, pert_frac)
%   [origCounts, summaryTbl, detailTbl] = perturb_top20_plus_minus(csvFile, top_pars, pert_frac, dataCsv)
%
% Example:
%   csvFile = "virtual_patients_500.csv";
%   dataCsv = "placebo_plotted_data(Survivor_Data).csv";
%   top_pars = [89 91 100 93 9 34 94 21 101 39 1 3 4 14 20 22 31 80 62 99];
%   [origCounts, summaryTbl, detailTbl] = perturb_top20_plus_minus(csvFile, top_pars, 0.15, dataCsv);
%
% What this fixed version does:
%   1. Does NOT use DataCopeland.mat.
%   2. Builds the baseline data structure from your CSV, just like virtual_patients_generator.m.
%   3. Reads each virtual patient's saved kpg, Pe_0, TNF_0, IL10_0, IL8_0, IL6_0, MR_0, and temp_0.
%   4. Perturbs exactly one parameter at a time by +pert_frac and -pert_frac.
%   5. Re-runs the ODE and records whether the patient changed class.
%   6. Writes summary/details CSV files and makes a stacked horizontal bar plot.
%
% Requires:
%   load_pars_Init_Copeland_Edited.m
%   modelDriver.m
%   model.m

    if nargin < 1 || isempty(csvFile)
        csvFile = "virtual_patients_500.csv";
    end

    if nargin < 2 || isempty(top_pars)
        % Default top-20 list matching the parameters shown in your perturbation plot.
        % top_pars = [1 3 5 7 9 11 ...
        %     35 36 37 38 39 48  61 77 84 ... 
        %     89 94 95 96 102 108 9];
        top_pars=[78 9 91 102 85 7];
    end

    if nargin < 3 || isempty(pert_frac)
        pert_frac = 0.15;
    end

    if nargin < 4 || isempty(dataCsv)
        dataCsv = "placebo_plotted_data(Survivor_Data).csv";
    end

    if nargin < 5 || isempty(outPrefix)
        [folderName, baseName, ~] = fileparts(char(csvFile));
        if isempty(folderName)
            outPrefix = "perturb_plus_minus_" + string(baseName);
        else
            outPrefix = string(fullfile(folderName, "perturb_plus_minus_" + string(baseName)));
        end
    else
        outPrefix = string(outPrefix);
    end

    if nargin < 6 || isempty(makePlot)
        makePlot = true;
    end

    top_pars = top_pars(:)';

    time   = [0 5000];
    epsThr = 1e-3;

    % -------------------------------------------------
    % Load baseline model data from CSV, not DataCopeland.mat
    % -------------------------------------------------
    data = read_patient_data_from_csv(dataCsv);

    % Some local versions of the model use a global BPo.
    global BPo
    if isfield(data, 'BP') && ~isempty(data.BP) && isfinite(data.BP(1))
        BPo = data.BP(1);
    else
        BPo = 118.4;
    end

    [basePars, Init] = load_pars_Init_Copeland_Edited(data);

    % State indices in your ODE model.
    tnf_idx    = 1;
    il10_idx   = 2;
    il8_idx    = 3;
    il6_idx    = 4;
    ma_idx     = 5;
    mr_idx     = 6;
    pe_idx     = 7;
    temp_idx   = 8;
    damage_idx = 17;

    % -------------------------------------------------
    % Read virtual-patient CSV
    % -------------------------------------------------
    if ~isfile(csvFile)
        error('Could not find virtual-patient CSV: %s', csvFile);
    end

    Tin = readtable(csvFile);
    N   = height(Tin);
    vns = string(Tin.Properties.VariableNames);

    if any(strcmpi(vns, "label"))
        labelCol = Tin.Properties.VariableNames{find(strcmpi(vns, "label"), 1)};
        origLabels = normalize_labels(string(Tin.(labelCol)));
    else
        error('The virtual-patient CSV must contain a label column.');
    end

    cats = ["septic"; "non_septic"; "aseptic"; "failed"];
    origCounts = table(cats, zeros(numel(cats), 1), ...
        'VariableNames', {'label', 'count'});

    for c = 1:numel(cats)
        origCounts.count(c) = sum(origLabels == cats(c));
    end

    % -------------------------------------------------
    % Detail results: rows = patients * parameters * 2 directions
    % -------------------------------------------------
    nPars = numel(top_pars);
    directions = ["plus"; "minus"];
    nDirs = numel(directions);
    nRows = N * nPars * nDirs;

    patient_id      = zeros(nRows, 1);
    par_index       = zeros(nRows, 1);
    par_name        = strings(nRows, 1);
    direction       = strings(nRows, 1);
    pct_change      = zeros(nRows, 1);
    original_value  = nan(nRows, 1);
    perturbed_value = nan(nRows, 1);
    original_label  = strings(nRows, 1);
    new_label       = strings(nRows, 1);
    label_changed   = false(nRows, 1);
    solver_ok       = false(nRows, 1);
    Ma_end          = nan(nRows, 1);
    Pe_end          = nan(nRows, 1);
    Damage_end      = nan(nRows, 1);

    row = 0;

    fprintf('\nRunning +/- %.1f%% perturbation analysis on %d virtual patients...\n', 100 * pert_frac, N);
    fprintf('Testing %d parameters, so total ODE runs = %d\n\n', nPars, nRows);

    % -------------------------------------------------
    % Loop over patients
    % -------------------------------------------------
    for i = 1:N

        if mod(i, 25) == 0 || i == 1 || i == N
            fprintf('Patient %d of %d\n', i, N);
        end

        % Original patient-specific parameter vector and initial condition.
        pars0  = basePars;
        Init_i = Init;

        % Load any par_# columns saved in the VP CSV.
        for c = 1:numel(vns)
            nm = vns(c);
            if strncmpi(char(nm), 'par_', 4)
                idx = sscanf(char(nm), 'par_%d');
                if ~isempty(idx) && idx >= 1 && idx <= numel(pars0)
                    val = Tin{i, char(nm)};
                    val = val(1);
                    if isnumeric(val) && isfinite(val)
                        pars0(idx) = max(val, 0);
                    end
                end
            end
        end

        % kpg is sometimes saved separately as kpg instead of par_89.
        val = get_table_value(Tin, i, ["kpg", "par_89"], NaN);
        if isfinite(val)
            pars0(89) = max(val, 0);
        end

        % Load patient-specific initial conditions from the VP CSV.
        val = get_table_value(Tin, i, "Pe_0", NaN);
        if isfinite(val), Init_i(pe_idx) = max(val, 0); end

        val = get_table_value(Tin, i, "TNF_0", NaN);
        if isfinite(val), Init_i(tnf_idx) = max(val, 0); end

        val = get_table_value(Tin, i, "IL10_0", NaN);
        if isfinite(val), Init_i(il10_idx) = max(val, 0); end

        val = get_table_value(Tin, i, "IL8_0", NaN);
        if isfinite(val), Init_i(il8_idx) = max(val, 0); end

        val = get_table_value(Tin, i, "IL6_0", NaN);
        if isfinite(val), Init_i(il6_idx) = max(val, 0); end

        val = get_table_value(Tin, i, "MR_0", NaN);
        if isfinite(val), Init_i(mr_idx) = max(val, 0); end

        val = get_table_value(Tin, i, "temp_0", NaN);
        if isfinite(val), Init_i(temp_idx) = val; end

        % -------------------------------------------------
        % For each parameter, run +pert_frac and -pert_frac separately.
        % -------------------------------------------------
        for k = 1:nPars
            j = top_pars(k);

            if j < 1 || j > numel(basePars)
                error('Parameter index %d is outside the valid range 1:%d.', j, numel(basePars));
            end

            oldVal = pars0(j);
            if ~isfinite(oldVal)
                oldVal = basePars(j);
            end

            for d = 1:nDirs
                row = row + 1;

                patient_id(row)     = i;
                par_index(row)      = j;
                par_name(row)       = parameter_display_name(j);
                original_label(row) = origLabels(i);
                original_value(row) = oldVal;
                direction(row)      = directions(d);

                pars = pars0;  % reset to original patient values every run

                if directions(d) == "plus"
                    pct_change(row) = 100 * pert_frac;
                    newVal = oldVal * (1 + pert_frac);
                else
                    pct_change(row) = -100 * pert_frac;
                    newVal = oldVal * (1 - pert_frac);
                end

                newVal = max(0, newVal);
                perturbed_value(row) = newVal;

                % Perturb only this one parameter.
                pars(j) = newVal;

                try
                    [~, sol] = modelDriver(pars, Init_i, time);
                catch
                    new_label(row)     = "failed";
                    label_changed(row) = new_label(row) ~= original_label(row);
                    continue;
                end

                if isempty(sol) || any(~isfinite(sol(end, :)))
                    new_label(row)     = "failed";
                    label_changed(row) = new_label(row) ~= original_label(row);
                    continue;
                end

                solver_ok(row) = true;

                Ma_end(row)     = sol(end, ma_idx);
                Pe_end(row)     = sol(end, pe_idx);
                Damage_end(row) = sol(end, damage_idx);

                new_label(row) = classify_outcome(Ma_end(row), Pe_end(row), Damage_end(row), epsThr);
                label_changed(row) = new_label(row) ~= original_label(row);
            end
        end
    end

    % -------------------------------------------------
    % Detail table
    % -------------------------------------------------
    detailTbl = table( ...
        patient_id, par_index, par_name, direction, pct_change, ...
        original_value, perturbed_value, ...
        original_label, new_label, label_changed, solver_ok, ...
        Ma_end, Pe_end, Damage_end);

    % -------------------------------------------------
    % Summary table: one row per parameter and direction
    % -------------------------------------------------
    nSummaryRows = nPars * nDirs;

    summary_par_index  = zeros(nSummaryRows, 1);
    summary_par_name   = strings(nSummaryRows, 1);
    summary_direction  = strings(nSummaryRows, 1);
    summary_pct_change = zeros(nSummaryRows, 1);
    run_label          = strings(nSummaryRows, 1);

    n_septic     = zeros(nSummaryRows, 1);
    n_non_septic = zeros(nSummaryRows, 1);
    n_aseptic    = zeros(nSummaryRows, 1);
    n_failed     = zeros(nSummaryRows, 1);
    n_changed    = zeros(nSummaryRows, 1);
    n_unchanged  = zeros(nSummaryRows, 1);
    n_solver_ok  = zeros(nSummaryRows, 1);

    septic_to_aseptic     = zeros(nSummaryRows, 1);
    septic_to_nonseptic   = zeros(nSummaryRows, 1);
    septic_to_failed      = zeros(nSummaryRows, 1);

    aseptic_to_septic     = zeros(nSummaryRows, 1);
    aseptic_to_nonseptic  = zeros(nSummaryRows, 1);
    aseptic_to_failed     = zeros(nSummaryRows, 1);

    nonseptic_to_septic   = zeros(nSummaryRows, 1);
    nonseptic_to_aseptic  = zeros(nSummaryRows, 1);
    nonseptic_to_failed   = zeros(nSummaryRows, 1);

    failed_to_septic      = zeros(nSummaryRows, 1);
    failed_to_aseptic     = zeros(nSummaryRows, 1);
    failed_to_nonseptic   = zeros(nSummaryRows, 1);

    srow = 0;
    for k = 1:nPars
        for d = 1:nDirs
            srow = srow + 1;

            j = top_pars(k);
            dirName = directions(d);

            idx = (detailTbl.par_index == j) & (detailTbl.direction == dirName);

            oldL = detailTbl.original_label(idx);
            newL = detailTbl.new_label(idx);

            summary_par_index(srow)  = j;
            summary_par_name(srow)   = parameter_display_name(j);
            summary_direction(srow)  = dirName;

            if dirName == "plus"
                summary_pct_change(srow) = 100 * pert_frac;
                run_label(srow) = sprintf('%s (par %d) +%.0f%%', char(summary_par_name(srow)), j, 100 * pert_frac);
            else
                summary_pct_change(srow) = -100 * pert_frac;
                run_label(srow) = sprintf('%s (par %d) -%.0f%%', char(summary_par_name(srow)), j, 100 * pert_frac);
            end

            n_septic(srow)     = sum(newL == "septic");
            n_non_septic(srow) = sum(newL == "non_septic");
            n_aseptic(srow)    = sum(newL == "aseptic");
            n_failed(srow)     = sum(newL == "failed");
            n_changed(srow)    = sum(detailTbl.label_changed(idx));
            n_unchanged(srow)  = sum(~detailTbl.label_changed(idx));
            n_solver_ok(srow)  = sum(detailTbl.solver_ok(idx));

            septic_to_aseptic(srow)    = sum((oldL == "septic")     & (newL == "aseptic"));
            septic_to_nonseptic(srow)  = sum((oldL == "septic")     & (newL == "non_septic"));
            septic_to_failed(srow)     = sum((oldL == "septic")     & (newL == "failed"));

            aseptic_to_septic(srow)    = sum((oldL == "aseptic")    & (newL == "septic"));
            aseptic_to_nonseptic(srow) = sum((oldL == "aseptic")    & (newL == "non_septic"));
            aseptic_to_failed(srow)    = sum((oldL == "aseptic")    & (newL == "failed"));

            nonseptic_to_septic(srow)  = sum((oldL == "non_septic") & (newL == "septic"));
            nonseptic_to_aseptic(srow) = sum((oldL == "non_septic") & (newL == "aseptic"));
            nonseptic_to_failed(srow)  = sum((oldL == "non_septic") & (newL == "failed"));

            failed_to_septic(srow)     = sum((oldL == "failed")     & (newL == "septic"));
            failed_to_aseptic(srow)    = sum((oldL == "failed")     & (newL == "aseptic"));
            failed_to_nonseptic(srow)  = sum((oldL == "failed")     & (newL == "non_septic"));
        end
    end

    summaryTbl = table( ...
        summary_par_index, summary_par_name, summary_direction, summary_pct_change, run_label, ...
        n_septic, n_non_septic, n_aseptic, n_failed, ...
        n_changed, n_unchanged, n_solver_ok, ...
        septic_to_aseptic, septic_to_nonseptic, septic_to_failed, ...
        aseptic_to_septic, aseptic_to_nonseptic, aseptic_to_failed, ...
        nonseptic_to_septic, nonseptic_to_aseptic, nonseptic_to_failed, ...
        failed_to_septic, failed_to_aseptic, failed_to_nonseptic);

    % Sort summary by how many patients changed class.
    [~, sortIdx] = sort(summaryTbl.n_changed, 'descend');
    summaryTbl = summaryTbl(sortIdx, :);

    % -------------------------------------------------
    % Write outputs
    % -------------------------------------------------
    summaryFile = outPrefix + "_summary.csv";
    detailFile  = outPrefix + "_details.csv";

    writetable(summaryTbl, summaryFile);
    writetable(detailTbl,  detailFile);

    fprintf('\n================ ORIGINAL COUNTS ================\n');
    disp(origCounts)

    fprintf('\n================ SUMMARY BY PARAMETER AND DIRECTION ================\n');
    disp(summaryTbl)

    fprintf('\nWrote:\n');
    fprintf('  %s\n', char(summaryFile));
    fprintf('  %s\n', char(detailFile));

    if makePlot
        pngFile = plot_perturbation_distribution(origCounts, summaryTbl, N, outPrefix, pert_frac);
        fprintf('  %s\n\n', char(pngFile));
    end
end

% =====================================================================
% Helper functions
% =====================================================================

function data = read_patient_data_from_csv(dataCsv)
%READ_PATIENT_DATA_FROM_CSV Build data struct from the same CSV style used in run_2.m.

    if ~isfile(dataCsv)
        error('Could not find baseline data CSV: %s', dataCsv);
    end

    opts = detectImportOptions(dataCsv, 'TreatAsMissing', {'#DIV/0!'});
    T = readtable(dataCsv, opts);

    % Convert string/cell columns to numeric when possible.
    for j = 1:width(T)
        varName = T.Properties.VariableNames{j};
        col = T.(varName);
        if iscell(col) || isstring(col) || ischar(col)
            T.(varName) = str2double(string(col));
        end
    end

    data = struct();
    data.hr   = get_first_existing_column(T, {"HR_Mean", "HR", "HeartRate_Mean", "Heart_Rate_Mean"});
    data.TNF  = get_first_existing_column(T, {"TNF_Mean", "TNF"});
    data.IL6  = get_first_existing_column(T, {"IL_6_Mean", "IL6_Mean", "IL6", "IL_6"});
    data.IL8  = get_first_existing_column(T, {"IL_8_Mean", "IL8_Mean", "IL8", "IL_8"});
    data.temp = get_first_existing_column(T, {"TEMP_Mean", "Temp_Mean", "Temperature_Mean", "TEMP", "Temp"});

    % BP is optional. If your local parameter loader uses data.BP, this keeps it safe.
    data.BP = get_optional_column(T, {"BP_Mean", "BloodPressure_Mean", "Blood_Pressure_Mean", "BP"}, 118.4);

    data.age    = 29;
    data.weight = 79.9;
    data.height = 177;
    data.HM     = 207 - 0.7 * data.age;
end

function x = get_first_existing_column(T, possibleNames)
    names = string(T.Properties.VariableNames);

    for k = 1:numel(possibleNames)
        idx = find(strcmpi(names, possibleNames{k}), 1);
        if ~isempty(idx)
            x = T.(T.Properties.VariableNames{idx});
            x = x(:);
            return;
        end
    end

    error('Missing required CSV column. Tried: %s', strjoin(possibleNames, ', '));
end

function x = get_optional_column(T, possibleNames, defaultValue)
    names = string(T.Properties.VariableNames);

    for k = 1:numel(possibleNames)
        idx = find(strcmpi(names, possibleNames{k}), 1);
        if ~isempty(idx)
            x = T.(T.Properties.VariableNames{idx});
            x = x(:);
            return;
        end
    end

    n = height(T);
    x = defaultValue * ones(n, 1);
end

function val = get_table_value(T, rowIdx, possibleNames, defaultValue)
    val = defaultValue;
    names = string(T.Properties.VariableNames);

    for k = 1:numel(possibleNames)
        idx = find(strcmpi(names, possibleNames(k)), 1);
        if ~isempty(idx)
            tmp = T{rowIdx, T.Properties.VariableNames{idx}};
            tmp = tmp(1);
            if isnumeric(tmp)
                val = tmp;
            else
                val = str2double(string(tmp));
            end
            return;
        end
    end
end

function labs = normalize_labels(labs)
    labs = lower(strtrim(string(labs)));

    labs(labs == "nonseptic")    = "non_septic";
    labs(labs == "non-septic")   = "non_septic";
    labs(labs == "non septic")   = "non_septic";

    labs(labs == "a_septic")     = "aseptic";
    labs(labs == "a-septic")     = "aseptic";

    % Treat unclassified as failed so all points are counted in the plot.
    labs(labs == "unclassified") = "failed";

    allowed = ["septic", "aseptic", "non_septic", "failed"];
    bad = ~ismember(labs, allowed);
    labs(bad) = "failed";
end

function lab = classify_outcome(Ma, Pe, Damage, epsThr)
    if (Ma <= epsThr) && (Pe <= epsThr) && (Damage <= epsThr)
        lab = "non_septic";
    elseif (Pe <= epsThr) && (Ma > epsThr) && (Damage > epsThr)
        lab = "aseptic";
    elseif (Pe > epsThr) && (Ma > epsThr) && (Damage > epsThr)
        lab = "septic";
    else
        lab = "failed";
    end
end

function nm = parameter_display_name(j)
    switch j
        case 1
            nm = "k10";
        case 3
            nm = "k6";
        case 4
            nm = "k6m";
        case 9
            nm = "kma";
        case 14
            nm = "x66";
        case 20
            nm = "xtnf6";
        case 21
            nm = "xmpe";
        case 22
            nm = "xm10";
        case 31
            nm = "htnf6";
        case 34
            nm = "hmpe";
        case 39
            nm = "sm";
        case 62
            nm = "xht";
        case 80
            nm = "hntnf";
        case 89
            nm = "kpg";
        case 91
            nm = "kpm";
        case 93
            nm = "muno";
        case 94
            nm = "kpn";
        case 99
            nm = "hmI10";
        case 100
            nm = "sM";
        case 101
            nm = "kmp";
        otherwise
            nm = "par" + string(j);
    end
end

function pngFile = plot_perturbation_distribution(origCounts, summaryTbl, N, outPrefix, pert_frac)
%PLOT_PERTURBATION_DISTRIBUTION Make the stacked horizontal bar plot.

    % Keep the plot readable: baseline plus sorted perturbation rows.
    nRows = height(summaryTbl) + 1;

    septic0     = get_count(origCounts, "septic");
    nonseptic0  = get_count(origCounts, "non_septic");
    aseptic0    = get_count(origCounts, "aseptic");
    failed0     = get_count(origCounts, "failed");

    yLabels = ["Baseline (original labels)"; summaryTbl.run_label];

    % Plot order: septic, non-septic, aseptic, failed.
    C = zeros(nRows, 4);
    C(1, :) = [septic0, nonseptic0, aseptic0, failed0];
    C(2:end, :) = [summaryTbl.n_septic, summaryTbl.n_non_septic, summaryTbl.n_aseptic, summaryTbl.n_failed];

    changed = [0; summaryTbl.n_changed];

    figH = max(7, min(20, 0.28 * nRows));
    fig = figure('Units', 'inches', 'Position', [1 1 14 figH], ...
        'Name', 'Perturbation outcome distribution', 'Color', 'w');

    ax = axes(fig);
    hold(ax, 'on');
    grid(ax, 'on');
    box(ax, 'on');

    b = barh(ax, C, 'stacked', 'BarWidth', 0.82);
    b(1).FaceColor = [0.0000 0.4470 0.7410]; % septic
    b(2).FaceColor = [0.8500 0.3250 0.0980]; % non-septic
    b(3).FaceColor = [0.4660 0.6740 0.1880]; % aseptic
    b(4).FaceColor = [0.6350 0.0780 0.1840]; % failed

    set(ax, 'YDir', 'reverse');
    yticks(ax, 1:nRows);
    yticklabels(ax, yLabels);
    ax.FontSize = 8;

    xlabel(ax, 'Number of patients');
    ylabel(ax, 'Baseline and perturbed runs');

    title(ax, sprintf('Outcome distribution after +/- %.0f%% perturbations of sensitive parameters', 100 * pert_frac));

    xlim(ax, [0, max(N * 1.16, N + 8)]);

    for r = 1:nRows
        text(ax, N + max(2, 0.01 * N), r, sprintf('changed = %d', changed(r)), ...
            'VerticalAlignment', 'middle', 'FontSize', 8);
    end

    legend(ax, {'Septic', 'Non-septic', 'Aseptic', 'Failed'}, ...
        'Location', 'southoutside', 'Orientation', 'horizontal');

    pngFile = outPrefix + "_distribution.png";

    try
        exportgraphics(fig, pngFile, 'Resolution', 300);
    catch
        saveas(fig, pngFile);
    end
end

function n = get_count(countTbl, lab)
    idx = countTbl.label == lab;
    if any(idx)
        n = countTbl.count(find(idx, 1));
    else
        n = 0;
    end
end
