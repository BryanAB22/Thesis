function [origCounts, summaryTbl, detailTbl] = perturb_top20_plus_minus(csvFile, top_pars, pert_frac)

    if nargin < 1 || isempty(csvFile)
        csvFile = "virtual_patients_50.csv";
    end

    if nargin < 2 || isempty(top_pars)
        top_pars = [ ...
              1   ... % k10
             22   ... % xm10
             99   ... % hml10
             80   ... % hntnf
              3   ... % k6
             14   ... % x66
              4   ... % k6m
             34   ... % hmpe
             31   ... % htnf6
             62   ... % xht
             21   ... % xmpe
             20   ... % xtnf6
             39   ... % sm
            101   ... % kmp
              9   ... % kma
             94   ... % kpn
             93   ... % muno
            100   ... % sM
             91   ... % kpm
             89   ... % kpg
        ];
    end

    if nargin < 3 || isempty(pert_frac)
        pert_frac = 0.15;
    end

    time   = [0 5000];
    epsThr = 1e-3;

    % -------------------------------------------------
    % Load baseline model data
    % -------------------------------------------------
    S = load('DataCopeland.mat');
    data = struct();

    BPt  = S.BPt;  HRt  = S.HRt;  TNFt  = S.TNFt; %#ok<NASGU>
    BPm  = S.BPm;  HRm  = S.HRm;  TNFm  = S.TNFm;
    BPse = S.BPse; HRse = S.HRse; TNFse = S.TNFse; %#ok<NASGU>

    IL6m  = S.IL6m;  IL8m  = S.IL8m;  TEMPt  = S.TEMPt; %#ok<NASGU>
    IL6se = S.IL6se; IL8se = S.IL8se; TEMPm  = S.TEMPm; TEMPse = S.TEMPse; %#ok<NASGU>

    data.BP   = BPm;
    data.hr   = HRm;
    data.TNF  = TNFm;
    data.IL6  = IL6m;
    data.IL8  = IL8m;
    data.temp = TEMPm(1:7);

    data.age    = 29;
    data.weight = 79.9;
    data.height = 177;
    data.HM     = 207 - 0.7 * data.age;

    global BPo
    BPo = data.BP(1);

    [basePars, Init] = load_pars_Init_Copeland_Edited(data);

    % State indices
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
    % Read CSV
    % -------------------------------------------------
    Tin = readtable(csvFile);
    N   = height(Tin);
    vns = string(Tin.Properties.VariableNames);

    if any(vns == "label")
        origLabels = normalize_labels(string(Tin.label));
    else
        error('The CSV must contain a "label" column.');
    end

    cats = ["septic"; "aseptic"; "non_septic"; "failed"];
    origCounts = table(cats, zeros(numel(cats),1), ...
        'VariableNames', {'label','count'});

    for c = 1:numel(cats)
        origCounts.count(c) = sum(origLabels == cats(c));
    end

    % -------------------------------------------------
    % Detail results
    % rows = patients * parameters * 2 directions
    % -------------------------------------------------
    nPars = numel(top_pars);
    directions = ["plus15"; "minus15"];
    nDirs = numel(directions);
    nRows = N * nPars * nDirs;

    patient_id      = zeros(nRows,1);
    par_index       = zeros(nRows,1);
    direction       = strings(nRows,1);
    pct_change      = zeros(nRows,1);
    original_value  = nan(nRows,1);
    perturbed_value = nan(nRows,1);
    original_label  = strings(nRows,1);
    new_label       = strings(nRows,1);
    label_changed   = false(nRows,1);
    solver_ok       = false(nRows,1);
    Ma_end          = nan(nRows,1);
    Pe_end          = nan(nRows,1);
    Damage_end      = nan(nRows,1);

    row = 0;

    % -------------------------------------------------
    % Loop over patients
    % -------------------------------------------------
    for i = 1:N

        % Original patient-specific parameter vector
        pars0  = basePars;
        Init_i = Init;

        % Load par_# columns from CSV
        for c = 1:numel(vns)
            nm = vns(c);
            if startsWith(nm, "par_")
                idx = sscanf(char(nm), 'par_%d');
                if ~isempty(idx)
                    val = Tin{i, char(nm)};
                    val = val(1);
                    if isfinite(val)
                        pars0(idx) = max(val, 0);
                    end
                end
            end
        end

        % Handle kpg if separate
        if any(vns == "kpg")
            val = Tin{i, "kpg"};
            val = val(1);
            if isfinite(val)
                pars0(89) = max(val, 0);
            end
        elseif any(vns == "par_89")
            val = Tin{i, "par_89"};
            val = val(1);
            if isfinite(val)
                pars0(89) = max(val, 0);
            end
        end

        % Load patient-specific initial conditions
        if any(vns == "Pe_0")
            val = Tin{i, "Pe_0"}; val = val(1);
            if isfinite(val), Init_i(pe_idx) = max(val,0); end
        end
        if any(vns == "TNF_0")
            val = Tin{i, "TNF_0"}; val = val(1);
            if isfinite(val), Init_i(tnf_idx) = max(val,0); end
        end
        if any(vns == "IL10_0")
            val = Tin{i, "IL10_0"}; val = val(1);
            if isfinite(val), Init_i(il10_idx) = max(val,0); end
        end
        if any(vns == "IL8_0")
            val = Tin{i, "IL8_0"}; val = val(1);
            if isfinite(val), Init_i(il8_idx) = max(val,0); end
        end
        if any(vns == "IL6_0")
            val = Tin{i, "IL6_0"}; val = val(1);
            if isfinite(val), Init_i(il6_idx) = max(val,0); end
        end
        if any(vns == "MR_0")
            val = Tin{i, "MR_0"}; val = val(1);
            if isfinite(val), Init_i(mr_idx) = max(val,0); end
        end
        if any(vns == "temp_0")
            val = Tin{i, "temp_0"}; val = val(1);
            if isfinite(val), Init_i(temp_idx) = val; end
        end

        % -------------------------------------------------
        % For each parameter, run +15% and -15% separately
        % -------------------------------------------------
        for k = 1:nPars
            j = top_pars(k);

            oldVal = pars0(j);
            if ~isfinite(oldVal)
                oldVal = basePars(j);
            end

            for d = 1:nDirs
                row = row + 1;

                patient_id(row)     = i;
                par_index(row)      = j;
                original_label(row) = origLabels(i);
                original_value(row) = oldVal;
                direction(row)      = directions(d);

                pars = pars0;  % reset to original patient values every run

                if directions(d) == "plus15"
                    pct_change(row) = 100 * pert_frac;
                    newVal = oldVal * (1 + pert_frac);
                else
                    pct_change(row) = -100 * pert_frac;
                    newVal = oldVal * (1 - pert_frac);
                end

                newVal = max(0, newVal);
                perturbed_value(row) = newVal;

                % perturb only one parameter
                pars(j) = newVal;

                try
                    [tSol, sol] = modelDriver(pars, Init_i, time); %#ok<ASGLU>
                catch
                    new_label(row)     = "failed";
                    label_changed(row) = new_label(row) ~= original_label(row);
                    continue;
                end

                if isempty(sol)
                    new_label(row)     = "failed";
                    label_changed(row) = new_label(row) ~= original_label(row);
                    continue;
                end

                solver_ok(row) = true;

                Ma_end(row)     = sol(end, ma_idx);
                Pe_end(row)     = sol(end, pe_idx);
                Damage_end(row) = sol(end, damage_idx);

                if (Ma_end(row) <= epsThr) && (Pe_end(row) <= epsThr) && (Damage_end(row) <= epsThr)
                    new_label(row) = "non_septic";

                elseif (Pe_end(row) <= epsThr) && (Ma_end(row) > epsThr) && (Damage_end(row) > epsThr)
                    new_label(row) = "aseptic";

                elseif (Pe_end(row) > epsThr) && (Ma_end(row) > epsThr) && (Damage_end(row) > epsThr)
                    new_label(row) = "septic";

                else
                    new_label(row) = "failed";
                end

                label_changed(row) = new_label(row) ~= original_label(row);
            end
        end
    end

    % -------------------------------------------------
    % Detail table
    % -------------------------------------------------
    detailTbl = table( ...
        patient_id, par_index, direction, pct_change, ...
        original_value, perturbed_value, ...
        original_label, new_label, label_changed, solver_ok, ...
        Ma_end, Pe_end, Damage_end);

    % -------------------------------------------------
    % Summary table
    % One row per (parameter, direction)
    % -------------------------------------------------
    summary_par_index   = zeros(nPars*nDirs,1);
    summary_direction   = strings(nPars*nDirs,1);
    summary_pct_change  = zeros(nPars*nDirs,1);

    n_septic            = zeros(nPars*nDirs,1);
    n_aseptic           = zeros(nPars*nDirs,1);
    n_non_septic        = zeros(nPars*nDirs,1);
    n_failed            = zeros(nPars*nDirs,1);
    n_changed           = zeros(nPars*nDirs,1);
    n_unchanged         = zeros(nPars*nDirs,1);
    n_solver_ok         = zeros(nPars*nDirs,1);

    septic_to_aseptic     = zeros(nPars*nDirs,1);
    septic_to_nonseptic   = zeros(nPars*nDirs,1);
    septic_to_failed      = zeros(nPars*nDirs,1);

    aseptic_to_septic     = zeros(nPars*nDirs,1);
    aseptic_to_nonseptic  = zeros(nPars*nDirs,1);
    aseptic_to_failed     = zeros(nPars*nDirs,1);

    nonseptic_to_septic   = zeros(nPars*nDirs,1);
    nonseptic_to_aseptic  = zeros(nPars*nDirs,1);
    nonseptic_to_failed   = zeros(nPars*nDirs,1);

    failed_to_septic      = zeros(nPars*nDirs,1);
    failed_to_aseptic     = zeros(nPars*nDirs,1);
    failed_to_nonseptic   = zeros(nPars*nDirs,1);

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
            summary_direction(srow)  = dirName;
            if dirName == "plus15"
                summary_pct_change(srow) = 100 * pert_frac;
            else
                summary_pct_change(srow) = -100 * pert_frac;
            end

            n_septic(srow)     = sum(newL == "septic");
            n_aseptic(srow)    = sum(newL == "aseptic");
            n_non_septic(srow) = sum(newL == "non_septic");
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
        summary_par_index, summary_direction, summary_pct_change, ...
        n_septic, n_aseptic, n_non_septic, n_failed, ...
        n_changed, n_unchanged, n_solver_ok, ...
        septic_to_aseptic, septic_to_nonseptic, septic_to_failed, ...
        aseptic_to_septic, aseptic_to_nonseptic, aseptic_to_failed, ...
        nonseptic_to_septic, nonseptic_to_aseptic, nonseptic_to_failed, ...
        failed_to_septic, failed_to_aseptic, failed_to_nonseptic);

    % -------------------------------------------------
    % Write outputs
    % -------------------------------------------------
    writetable(summaryTbl, "perturb_plus_minus_summary_50.csv");
    writetable(detailTbl,  "perturb_plus_minus_details_50.csv");

    fprintf('\n================ ORIGINAL COUNTS ================\n');
    disp(origCounts)

    fprintf('\n================ SUMMARY BY PARAMETER AND DIRECTION ================\n');
    disp(summaryTbl)

    fprintf('\nWrote:\n');
    fprintf('  perturb_plus_minus_summary_50.csv\n');
    fprintf('  perturb_plus_minus_details_50.csv\n\n');
end


function labs = normalize_labels(labs)
    labs = lower(strtrim(string(labs)));

    labs(labs == "nonseptic")  = "non_septic";
    labs(labs == "non-septic") = "non_septic";
    labs(labs == "non septic") = "non_septic";

    labs(labs == "a_septic")   = "aseptic";
    labs(labs == "a-septic")   = "aseptic";

    allowed = ["septic","aseptic","non_septic","failed"];
    bad = ~ismember(labs, allowed);
    labs(bad) = "failed";
end