function SSE = cost_function_2(theta, fit_idx, base_pars, Init, fitData, tspan, useLogFit, perturbPercent)
% cost_function_2
%
% theta is now a DIRECT fractional percent change, not a log value.
% Example:
%   theta =  0.30  means parameter = base_parameter*(1 + 0.30)  = +30%
%   theta = -0.30  means parameter = base_parameter*(1 - 0.30)  = -30%
%
% The parameters are hard-limited to +/- perturbPercent.
% There is NO penalty term added to the SSE.

    if nargin < 7 || isempty(useLogFit)
        useLogFit = false;
    end

    if nargin < 8 || isempty(perturbPercent)
        perturbPercent = 30;   % default: allow +/-30 percent
    end

    badCost = 1e20;            % returned only when the model solve or data match fails

    theta = theta(:);
    fit_idx = fit_idx(:);
    base_pars = base_pars(:);

    % Let the user write 30 for 30%, or 0.30 for 30%.
    if perturbPercent > 1
        perturbFraction = perturbPercent/100;
    else
        perturbFraction = perturbPercent;
    end
    perturbFraction = abs(perturbFraction);

    if perturbFraction >= 1
        error('perturbPercent must be less than 100%% so fitted positive parameters do not cross zero.');
    end

    % Hard clamp: parameters are only allowed to move inside +/- perturbPercent.
    % No penalty is used.
    theta = max(min(theta, perturbFraction), -perturbFraction);

    % Direct multiplier from percent change. No exp(), no log().
    multiplier = 1 + theta;

    pars_new = base_pars;
    pars_new(fit_idx) = base_pars(fit_idx).*multiplier;

    if any(~isfinite(pars_new))
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
        [t, sol] = solveModelAtTimes(pars_new, Init, tEval);
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

    residuals = [];
    epsLog = 1e-8;
    tol = 1e-8;

    for i = 1:numel(fields)
        obs = fitData.(fields{i});

        td = obs.time(:);
        yd = obs.mean(:);
        sd = obs.sd(:);
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
        sd = sd(valid);
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
            sd     = sd(validLog);

            sigma = abs(sd ./ yd);
            sigma(~isfinite(sigma) | sigma <= 0) = 0.10;
            sigma = max(sigma, 0.05);

            r = (log(max(ymodel,epsLog)) - log(max(yd,epsLog))) ./ sigma;
        else
            sigma = sd;

            fallback = 0.10*max(abs(yd), 1);
            sigma(~isfinite(sigma) | sigma <= 0) = fallback(~isfinite(sigma) | sigma <= 0);
            sigma = max(sigma, 1e-8);

            r = (ymodel - yd) ./ sigma;
        end

        if any(~isfinite(r))
            SSE = badCost;
            return
        end

        residuals = [residuals; r(:)]; %#ok<AGROW>
    end

    if isempty(residuals)
        SSE = badCost;
    else
        SSE = sum(residuals.^2);   % no penalty term
    end
end

function [t, sol] = solveModelAtTimes(pars_new, Init, tEval)
    global ODE_TOL

    if isempty(ODE_TOL)
        ODE_TOL = 1e-8;
    end

    options = odeset('RelTol', ODE_TOL, ...
                     'AbsTol', ODE_TOL, ...
                     'NonNegative', 1:numel(Init));

    [t, sol] = ode23s(@(tt, yy) model(tt, yy, pars_new), tEval, Init, options);
end
