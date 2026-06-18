function [t,sol] = rlModelStep(pars,state,startTime,endTime,doseRate)
% rlModelStep
%
% Integrate the existing MATLAB sepsis model over one RL decision interval.
% Python/PyTorch calls this function through the MATLAB Engine API.
%
% doseRate is the antibiotic administration input u(t) over this interval.
% The antimicrobial intervention is kept enabled even when doseRate = 0 so
% residual Abx concentration continues to clear and exert PK/PD killing.

if endTime <= startTime
    error('endTime must be greater than startTime.');
end

if size(state,1) > 1
    state = state(:)';
end

intervention.pathogen.enabled   = true;
intervention.pathogen.startTime = startTime;
intervention.pathogen.endTime   = endTime;
intervention.pathogen.doseRate  = doseRate;

intervention.nitricOxide.enabled        = false;
intervention.nitricOxide.startTime      = inf;
intervention.nitricOxide.endTime        = -inf;
intervention.nitricOxide.inhibitionRate = 0;

time = linspace(startTime,endTime,8)';
previousWarningState = warning('off','all');
cleanupWarningState = onCleanup(@() warning(previousWarningState));
[t,sol] = modelDriver(pars,state,time,intervention);
end
