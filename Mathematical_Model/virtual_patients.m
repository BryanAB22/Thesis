function [nSeptic,nNonSeptic,nAseptic,labels,parSamples] = virtual_patients_2(sens_pars,N,rel_bound)

    time    = [0 5000];
    outFile = "virtual_patients_50.csv";

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

    labels     = strings(N,1);
    parSamples = nan(N, numel(sens_pars));
    par_89     = nan(N,1);

    pe_0       = nan(N,1);
    tnf_0      = nan(N,1);
    il10_0     = nan(N,1);
    il8_0      = nan(N,1);
    il6_0      = nan(N,1);
    mr_0       = nan(N,1);
    temp_0     = nan(N,1);

    nSeptic    = 0;
    nNonSeptic = 0;
    nAseptic   = 0;

    epsThr    = 1e-3;
    pert_frac = 0.15;

    max_retries = 1;

    for i = 1:N

        ok = false;
        attempt = 0;

        while ~ok && attempt < max_retries
            attempt = attempt + 1;

            pars = basePars;
            Init_i = Init;

            tmpSample = nan(1, numel(sens_pars));
            for k = 1:numel(sens_pars)
                j  = sens_pars(k);
                p0 = pars(j);

                pars(j) = p0 * (1 + rel_bound * (2*rand - 1));
                pars(j) = max(pars(j), 0);

                tmpSample(k) = pars(j);
            end

           
            % uniformly [0,2]
            pars(89) = 2 * rand;
            tmpPar89 = pars(89);


            idx89 = find(sens_pars == 89, 1);
            if ~isempty(idx89)
                tmpSample(idx89) = pars(89);
            end

            % Pe_0[0,2]
            Init_i(pe_idx) = 2 * rand;

            % TNF_0, IL10_0, IL8_0, IL6_0, MR_0 within ±15% of baseline
            Init_i(tnf_idx)  = max(0, Init(tnf_idx)  * (1 + pert_frac * (2*rand - 1)));
            Init_i(il10_idx) = max(0, Init(il10_idx) * (1 + pert_frac * (2*rand - 1)));
            Init_i(il8_idx)  = max(0, Init(il8_idx)  * (1 + pert_frac * (2*rand - 1)));
            Init_i(il6_idx)  = max(0, Init(il6_idx)  * (1 + pert_frac * (2*rand - 1)));
            Init_i(mr_idx)   = max(0, Init(mr_idx)   * (1 + pert_frac * (2*rand - 1)));

            % temp_0 uniformly from [36.5, 37.5]
            Init_i(temp_idx) = 36.5 + (37.5 - 36.5) * rand;

            try
                [tSol, sol] = modelDriver(pars, Init_i, time);
            catch
                continue;
            end

            if isempty(sol) || isempty(tSol)
                continue;
            end

            ok = true;

            parSamples(i,:) = tmpSample;
            par_89(i)       = tmpPar89;

            pe_0(i)         = Init_i(pe_idx);
            tnf_0(i)        = Init_i(tnf_idx);
            il10_0(i)       = Init_i(il10_idx);
            il8_0(i)        = Init_i(il8_idx);
            il6_0(i)        = Init_i(il6_idx);
            mr_0(i)         = Init_i(mr_idx);
            temp_0(i)       = Init_i(temp_idx);

            Ma     = sol(end, ma_idx);
            Pe     = sol(end, pe_idx);
            Damage = sol(end, damage_idx);

            if (Ma <= epsThr) && (Pe <= epsThr) && (Damage <= epsThr)
                labels(i) = "non_septic";
                nNonSeptic = nNonSeptic + 1;

            elseif (Pe <= epsThr) && (Ma > epsThr) && (Damage > epsThr)
                labels(i) = "aseptic";
                nAseptic = nAseptic + 1;

            elseif (Pe > epsThr) && (Ma > epsThr) && (Damage > epsThr)
                labels(i) = "septic";
                nSeptic = nSeptic + 1;

            else
                labels(i) = "failed";
            end
        end

        if ~ok
            labels(i) = "failed";
        end
    end

    T = array2table(parSamples);
    T.Properties.VariableNames = "par_" + string(sens_pars);

    T.kpg    = par_89;
    T.Pe_0   = pe_0;
    T.TNF_0  = tnf_0;
    T.IL10_0 = il10_0;
    T.IL8_0  = il8_0;
    T.IL6_0  = il6_0;
    T.MR_0   = mr_0;
    T.temp_0 = temp_0;
    T.label  = labels;

    writetable(T, outFile);

end