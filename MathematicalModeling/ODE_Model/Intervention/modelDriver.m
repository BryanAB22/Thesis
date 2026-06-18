% modelDriver.m
%
% Runs the ODE model with optional PK/PD antimicrobial and
% nitric-oxide interventions.

function [t,sol] = modelDriver(pars,Init,time,intervention)

global ODE_TOL

if nargin < 4 || isempty(intervention)
    intervention.pathogen.enabled = false;
    intervention.nitricOxide.enabled = false;
end

if isempty(time) || numel(time) < 2
    error('time must contain at least a start time and an end time.');
end

if time(end) <= time(1)
    error('The final simulation time must be greater than the initial time.');
end

options = odeset( ...
    'RelTol',ODE_TOL, ...
    'AbsTol',ODE_TOL, ...
    'NonNegative',1:numel(Init));

% Include every intervention switch time in tspan so ode23s resolves the
% discontinuities at the treatment boundaries.
tspan = time(:);
switchTimes = [];

if intervention.pathogen.enabled
    switchTimes = [switchTimes; ...
        intervention.pathogen.startTime; ...
        intervention.pathogen.endTime]; %#ok<AGROW>
end

if intervention.nitricOxide.enabled
    switchTimes = [switchTimes; ...
        intervention.nitricOxide.startTime; ...
        intervention.nitricOxide.endTime]; %#ok<AGROW>
end

switchTimes = switchTimes( ...
    switchTimes >= time(1) & switchTimes <= time(end));

tspan = unique([tspan; switchTimes]);

[t,sol] = ode23s( ...
    @(t,y) model(t,y,pars,intervention), ...
    tspan, ...
    Init, ...
    options);
