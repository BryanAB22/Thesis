function SSE = cost_function_2(delta, fit_idx, base_pars, Init, fitData, tspan, useLogFit, perturbPercent, init_idx)
% cost_function_2
%
% Objective used by Bayesian optimization.
%
% IMPORTANT CHANGES:
%   1) The fitted variables are additive changes, not multipliers.
%          pars_new(fit_idx) = base_pars(fit_idx) + delta_par
%          Init_new(init_idx) = Init(init_idx) + delta_init
%
%   2) The cost is the sum of per-state averaged UNWEIGHTED squared residuals:
%
%          SSE = (1/N_HR)   * sum_i (HR_model_i   - HR_obs_i)^2   +
%                (1/N_TEMP) * sum_i (TEMP_model_i - TEMP_obs_i)^2 +
%                (1/N_TNF)  * sum_i (TNF_model_i  - TNF_obs_i)^2  +
%                (1/N_IL6)  * sum_i (IL6_model_i  - IL6_obs_i)^2  +
%                (1/N_IL8)  * sum_i (IL8_model_i  - IL8_obs_i)^2
%
%      The residuals are NOT divided by SD.
%      Only finite observed data points are counted in each N.
%      There is NO final division by the number of states.
%
% Inputs:
%   delta          additive changes for fitted parameters followed by fitted ICs
%   fit_idx        parameter indices being fit
%   base_pars      starting parameter vector
%   Init           starting initial-condition vector
%   fitData        struct containing observed data for each measured state
%   tspan          model evaluation times; observed data times are added automatically
%   useLogFit      true = log residuals, false = linear residuals
%   perturbPercent used only to define a safety clamp for additive delta size
%   init_idx       initial-condition indices being fit, e.g. Init(7) = PE(0)

    if nargin < 7 || isempty(useLogFit)
        useLogFit = false;
    end

    if nargin < 8 || isempty(perturbPercent)
        perturbPercent = 30;
    end

    if nargin < 9 || isempty(init_idx)
        init_idx = [];
    end

    badCost = 1e20;

    delta = delta(:);
    fit_idx = fit_idx(:);
    init_idx = init_idx(:);
    base_pars = base_pars(:);
    Init = Init(:);

    nParFit  = numel(fit_idx);
    nInitFit = numel(init_idx);
    nDeltaExpected = nParFit + nInitFit;

    if numel(delta) ~= nDeltaExpected
        error('delta has %d entries, but expected %d = %d parameters + %d initial conditions.', ...
              numel(delta), nDeltaExpected, nParFit, nInitFit);
    end

    % Convert user setting to a fraction only for the additive bound size.
    % Example: perturbPercent = 80 means abs(delta) <= 0.80*abs(starting value).
    if perturbPercent > 1
        perturbFraction = perturbPercent/100;
    else
        perturbFraction = perturbPercent;
    end
    perturbFraction = abs(perturbFraction);

    % Safety clamp: this is additive, not multiplicative.
    startVals = [base_pars(fit_idx); Init(init_idx)];
    deltaMax = perturbFraction .* abs(startVals);

    % If a selected value is exactly zero, still allow a small positive search range.
    zeroScale = 1;
    zeroMask = deltaMax == 0;
    deltaMax(zeroMask) = perturbFraction * zeroScale;

    lowerDelta = -deltaMax;
    upperDelta =  deltaMax;

    % Most ODE parameters and initial conditions are nonnegative. Prevent the
    % optimizer from making a nonnegative starting value negative.
    nonnegativeStart = startVals >= 0;
    lowerDelta(nonnegativeStart) = max(lowerDelta(nonnegativeStart), -0.999999*startVals(nonnegativeStart));

    delta = min(max(delta, lowerDelta), upperDelta);

    delta_par  = delta(1:nParFit);
    delta_init = delta(nParFit+1:end);

    pars_new = base_pars;
    if nParFit > 0
        pars_new(fit_idx) = base_pars(fit_idx) + delta_par;
    end

    Init_new = Init;
    if nInitFit > 0
        Init_new(init_idx) = Init(init_idx) + delta_init;
    end

    if any(~isfinite(pars_new)) || any(~isfinite(Init_new)) || any(Init_new < 0)
        SSE = badCost;
        return
    end

    % Build the model output times from the requested tspan plus all actual
    % observed data times. This makes the fit compare only at CSV times.
    fields = fieldnames(fitData);
    tEval = tspan(:);
    for i = 1:numel(fields)
        obs = fitData.(fields{i});
        td_all = obs.time(:);
        yd_all = obs.mean(:);
        validTime = isfinite(td_all) & isfinite(yd_all);
        tEval = [tEval; td_all(validTime)]; %#ok<AGROW>
    end
    tEval = unique(tEval(isfinite(tEval)), 'sorted');

    if numel(tEval) < 2
        SSE = badCost;
        return
    end

    % Solve the ODE directly at the requested evaluation times. This avoids
    % fitting at every internal ODE solver step and avoids interpolation here.
    try
        [t, sol] = solveModelAtTimes(pars_new, Init_new, tEval);
    catch
        SSE = badCost;
        return
    end

    if isempty(t) || isempty(sol) || any(~isfinite(sol(:)))
        SSE = badCost;
        return
    end

    [tUnique, ia] = unique(t(:), 'stable');
    solUnique = sol(ia, :);

    stateCosts = [];
    epsLog = 1e-8;
    tol = 1e-8;

    for i = 1:numel(fields)
        obs = fitData.(fields{i});

        td = obs.time(:);
        yd = obs.mean(:);
        stateIdx = obs.state;

        if stateIdx > size(solUnique,2)
            SSE = badCost;
            return
        end

        valid = isfinite(td) & isfinite(yd);
        if ~any(valid)
            continue
        end

        td = td(valid);
        yd = yd(valid);
        ymodel = nan(size(td));

        for q = 1:numel(td)
            matchIdx = find(abs(tUnique - td(q)) < tol, 1);

            if isempty(matchIdx)
                SSE = badCost;
                return
            end

            ymodel(q) = solUnique(matchIdx, stateIdx);
        end

        ymodel = ymodel(:);

        if useLogFit
            validLog = isfinite(ymodel) & ymodel > 0 & yd > 0;
            if ~any(validLog)
                SSE = badCost;
                return
            end

            ymodel = ymodel(validLog);
            yd     = yd(validLog);

            % Unweighted log residual: no SD division.
            r = log(max(ymodel,epsLog)) - log(max(yd,epsLog));
        else
            % Unweighted linear residual: no SD division.
            r = ymodel - yd;
        end

        if any(~isfinite(r))
            SSE = badCost;
            return
        end

        % This is exactly (1/N_state) * sum(unweighted r_i^2) for this state.
        N_state = numel(r);
        stateCosts = [stateCosts; sum(r(:).^2) / N_state]; %#ok<AGROW>
    end

    if isempty(stateCosts)
        SSE = badCost;
    else
        % Sum the averaged state costs. Do NOT divide by number of states.
        SSE = sum(stateCosts);
    end
end

function [t, sol] = solveModelAtTimes(pars_new, Init_new, tEval)
    global ODE_TOL

    if isempty(ODE_TOL)
        ODE_TOL = 1e-8;
    end

    options = odeset('RelTol', ODE_TOL, ...
                     'AbsTol', ODE_TOL, ...
                     'NonNegative', 1:numel(Init_new));

    [t, sol] = ode23s(@(tt, yy) model(tt, yy, pars_new), tEval, Init_new, options);
end
