clear; clc; close all;

% ============================================================
% Files
% ============================================================
% This is the CSV you already use in run_2.m.
% Change the name here if your CSV has a different filename.
dataCsv = "placebo_plotted_data(Survivor_Data).csv";

% Output CSV created by the virtual-patient generator.
outFile = "virtual_patients_500.csv";

% ============================================================
% Virtual-patient settings
% ============================================================
N         = 500;       % number of virtual patients
rel_bound = 0.25;     

% Parameters to perturb around baseline.
% pars(89) = kpg is always sampled from [0, 2] for the scatter plot.
% Start simple with [89]. Add more parameter indices later if needed.
sens_pars = [89, 76, 9 ,91,98,100,93,39,102,95,7];

% ============================================================
% Generate CSV and plot
% ============================================================
[counts, T] = virtual_patients_generator(sens_pars, N, rel_bound, outFile, dataCsv); %#ok<NASGU,ASGLU>
plot_virtual_patients(outFile);
