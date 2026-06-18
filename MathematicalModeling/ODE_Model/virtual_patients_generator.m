function [counts, T] = virtual_patients_generator(sens_pars, N, rel_bound, outFile, dataCsv)
%VIRTUAL_PATIENTS_GENERATOR Generate virtual patients without DataCopeland.mat.
%
% Usage:
%   [counts, T] = virtual_patients_generator(sens_pars, N, rel_bound, outFile, dataCsv)
%
% Example:
%   sens_pars = [89];
%   N = 50;
%   rel_bound = 0.15;
%   outFile = "virtual_patients_50.csv";
%   dataCsv = "placebo_plotted_data(Survivor_Data).csv";
%   [counts, T] = virtual_patients_generator(sens_pars, N, rel_bound, outFile, dataCsv);
%
% Notes:
%   - No DataCopeland.mat is used.
%   - The baseline data structure is built from your CSV, like run_2.m.
%   - kpg = pars(89) is sampled uniformly from [0, 2].
%   - Pe_0 = Init(7) is sampled uniformly from [0, 2].
%   - Other selected sensitive parameters are perturbed by +/- rel_bound.
%   - Failed ODE solves are saved with label = "failed".
%   - Requires load_pars_Init_Copeland_Edited.m, modelDriver.m, and model.m.

    if nargin < 1 || isempty(sens_pars)
        sens_pars = [89];
    end
    if nargin < 2 || isempty(N)
        N = 50;
    end
    if nargin < 3 || isempty(rel_bound)
        rel_bound = 0.15;
    end
    if nargin < 4 || isempty(outFile)
        outFile = "virtual_patients_50.csv";
    end
    if nargin < 5 || isempty(dataCsv)
        dataCsv = "placebo_plotted_data(Survivor_Data).csv";
    end

    rng('shuffle');

    % Simulate long enough for final classification.
    time = [0 5000];

    % Build baseline data from CSV instead of DataCopeland.mat.
    data = read_patient_data_from_csv(dataCsv);

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

    % Storage.
    labels     = strings(N, 1);
    parSamples = nan(N, numel(sens_pars));

    kpg_89 = nan(N, 1);
    Pe_0   = nan(N, 1);
    TNF_0  = nan(N, 1);
    IL10_0 = nan(N, 1);
    IL8_0  = nan(N, 1);
    IL6_0  = nan(N, 1);
    MR_0   = nan(N, 1);
    temp_0 = nan(N, 1);

    Ma_final     = nan(N, 1);
    Pe_final     = nan(N, 1);
    Damage_final = nan(N, 1);

    epsThr     = 1e-3;
    pert_frac  = 0.15;
    maxRetries = 3;

    for i = 1:N
        ok = false;

        % Keep the final attempted values so failed runs can still be plotted.
        last_tmpSample = nan(1, numel(sens_pars));
        last_kpg       = nan;
        last_Pe0       = nan;
        last_TNF0      = nan;
        last_IL100     = nan;
        last_IL80      = nan;
        last_IL60      = nan;
        last_MR0       = nan;
        last_temp0     = nan;

        for attempt = 1:maxRetries
            pars   = basePars;
            Init_i = Init;

            tmpSample = nan(1, numel(sens_pars));

            % Perturb chosen sensitive parameters.
            for k = 1:numel(sens_pars)
                j = sens_pars(k);
                p0 = pars(j);

                pars(j) = p0 * (1 + rel_bound * (2 * rand - 1));
                pars(j) = max(pars(j), 0);
                tmpSample(k) = pars(j);
            end

            % Main virtual-patient coordinates for the scatter plot.
            pars(89)       = 2 * rand;     % kpg in [0, 2]
            Init_i(pe_idx) = 2 * rand;     % Pe_0 in [0, 2]

            % If parameter 89 was included in sens_pars, store the true sampled kpg.
            idx89 = find(sens_pars == 89, 1);
            if ~isempty(idx89)
                tmpSample(idx89) = pars(89);
            end

            % Initial-condition variation.
            Init_i(tnf_idx)  = max(0, Init(tnf_idx)  * (1 + pert_frac * (2 * rand - 1)));
            Init_i(il10_idx) = max(0, Init(il10_idx) * (1 + pert_frac * (2 * rand - 1)));
            Init_i(il8_idx)  = max(0, Init(il8_idx)  * (1 + pert_frac * (2 * rand - 1)));
            Init_i(il6_idx)  = max(0, Init(il6_idx)  * (1 + pert_frac * (2 * rand - 1)));
            Init_i(mr_idx)   = max(0, Init(mr_idx)   * (1 + pert_frac * (2 * rand - 1)));
            Init_i(temp_idx) = 36.5 + (37.5 - 36.5) * rand;

            % Save the last attempted values even if the ODE solve fails.
            last_tmpSample = tmpSample;
            last_kpg       = pars(89);
            last_Pe0       = Init_i(pe_idx);
            last_TNF0      = Init_i(tnf_idx);
            last_IL100     = Init_i(il10_idx);
            last_IL80      = Init_i(il8_idx);
            last_IL60      = Init_i(il6_idx);
            last_MR0       = Init_i(mr_idx);
            last_temp0     = Init_i(temp_idx);

            try
                [tSol, sol] = modelDriver(pars, Init_i, time); %#ok<ASGLU>
            catch ME
                fprintf('Patient %d attempt %d failed: %s\n', i, attempt, ME.message);
                continue;
            end

            if isempty(sol) || any(~isfinite(sol(end, :)))
                continue;
            end

            ok = true;
            break;
        end

        if ~ok
            parSamples(i, :) = last_tmpSample;
            kpg_89(i) = last_kpg;
            Pe_0(i)   = last_Pe0;
            TNF_0(i)  = last_TNF0;
            IL10_0(i) = last_IL100;
            IL8_0(i)  = last_IL80;
            IL6_0(i)  = last_IL60;
            MR_0(i)   = last_MR0;
            temp_0(i) = last_temp0;
            labels(i) = "failed";
            continue;
        end

        parSamples(i, :) = tmpSample;

        kpg_89(i) = pars(89);
        Pe_0(i)   = Init_i(pe_idx);
        TNF_0(i)  = Init_i(tnf_idx);
        IL10_0(i) = Init_i(il10_idx);
        IL8_0(i)  = Init_i(il8_idx);
        IL6_0(i)  = Init_i(il6_idx);
        MR_0(i)   = Init_i(mr_idx);
        temp_0(i) = Init_i(temp_idx);

        Ma_final(i)     = sol(end, ma_idx);
        Pe_final(i)     = sol(end, pe_idx);
        Damage_final(i) = sol(end, damage_idx);

        Ma     = Ma_final(i);
        Pe     = Pe_final(i);
        Damage = Damage_final(i);

        if (Ma <= epsThr) && (Pe <= epsThr) && (Damage <= epsThr)
            labels(i) = "non_septic";
        elseif (Pe <= epsThr) && (Ma > epsThr) && (Damage > epsThr)
            labels(i) = "aseptic";
        elseif (Pe > epsThr) && (Ma > epsThr) && (Damage > epsThr)
            labels(i) = "septic";
        else
            labels(i) = "unclassified";
        end
    end

    % Build output table.
    T = array2table(parSamples);
    T.Properties.VariableNames = "par_" + string(sens_pars);

    T.kpg          = kpg_89;
    T.Pe_0         = Pe_0;
    T.TNF_0        = TNF_0;
    T.IL10_0       = IL10_0;
    T.IL8_0        = IL8_0;
    T.IL6_0        = IL6_0;
    T.MR_0         = MR_0;
    T.temp_0       = temp_0;
    T.Ma_final     = Ma_final;
    T.Pe_final     = Pe_final;
    T.Damage_final = Damage_final;
    T.label        = labels;

    writetable(T, outFile);

    counts = local_label_counts(labels);
    disp(counts);
    fprintf('\nSaved virtual patients to: %s\n', outFile);
end

function data = read_patient_data_from_csv(dataCsv)
%READ_PATIENT_DATA_FROM_CSV Build data struct from the same CSV style used in run_2.m.

    if ~isfile(dataCsv)
        error('Could not find data CSV: %s', dataCsv);
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

    % BP is optional because your current load_pars_Init_Copeland_Edited.m
    % uses a hard-coded BPo. Keep this here in case your local version uses data.BP.
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

function counts = local_label_counts(labels)
    u = unique(labels);
    n = zeros(numel(u), 1);

    for k = 1:numel(u)
        n(k) = sum(labels == u(k));
    end

    counts = table(u, n, 'VariableNames', {'label', 'count'});
end
