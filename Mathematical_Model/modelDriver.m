%modelDriver.m

function [t,sol] = modelDriver(pars,Init,time)

global ODE_TOL

% options = odeset('RelTol',ODE_TOL,'AbsTol',ODE_TOL);
options = odeset('RelTol',ODE_TOL,'AbsTol',ODE_TOL, 'NonNegative',1:17); 

% sol = dde23(@model,0.4,Init,[time(1) time(end)],options,pars); 
[t,sol] = ode23s(@(t,sol) model(t,sol,pars),[time(1) time(end)],Init,options); 

% ode23s, ode113, ode78, ode89