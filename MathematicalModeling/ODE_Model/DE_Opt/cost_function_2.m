function SSE = cost_function_2(delta, fit_idx, base_pars, Init, fitData, tspan, useLogFit, perturbPercent, init_idx)
% cost_function_2
%
% Objective used by Differential Evolution or Bayesian optimization.
%
% IMPORTANT CHANGES:
%   1) The fitted variables are additive changes, not multipliers.
%          pars_new(fit_idx) = base_pars(fit_idx) + delta_par
%          Init_new(init_idx) = Init(init_idx) + delta_init
%
%   2) For linear fitting, each state's residuals are standardized by
%      that state's observed-data standard deviation:
%
%          r_state = (model - observed) / std(observed)
%
%      The cost is the sum of the mean standardized squared errors:
%
%          SSE = mean(r_HR.^2) + mean(r_TEMP.^2) + mean(r_TNF.^2) +
%                mean(r_IL6.^2) + mean(r_IL8.^2)
%
%      This makes the five measured states comparable despite different units
%      and numerical scales. If a state's observed SD is zero or invalid, a
%      safe fallback scale based on its observed mean is used.
%
%   3) For optional log fitting, log1p residuals are used so zero-valued
%      observations remain in the objective:
%
%          r_state = log1p(model) - log1p(observed)
%
%      Only finite observed data points are counted in each N. There is no
%      final division by the number of states.
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
    tol = 1e-8;
    minimumScale = 1e-8;

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
            % All five fitted states are expected to be nonnegative. Reject a
            % candidate solution that produces a negative fitted-state value.
            if any(ymodel < 0) || any(yd < 0)
                SSE = badCost;
                return
            end

            % log1p keeps zero-valued observations in the objective and makes
            % the residual reflect proportional differences more strongly.
            r = log1p(ymodel) - log1p(yd);
        else
            % Standardize this state's residuals using the spread of its own
            % observed data. This makes HR, temperature, TNF, IL-6, and IL-8
            % contribute on comparable dimensionless scales.
            scaleValue = std(yd, 0);

            % Safe fallback for a constant or nearly constant observed state.
            if ~isfinite(scaleValue) || scaleValue < minimumScale
                scaleValue = max(abs(mean(yd)), 1);
            end

            r = (ymodel - yd) ./ scaleValue;
        end

        if any(~isfinite(r))
            SSE = badCost;
            return
        end

        % Per-state mean squared residual. In linear mode, r is standardized;
        % in log mode, r is a log1p difference.
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
