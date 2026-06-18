% run_antimicrobial_no_intervention.m
%
% Compare four cases:
%   1. No treatment
%   2. Antimicrobial only
%   3. Nitric-oxide inhibition only
%   4. Combined antimicrobial + nitric-oxide inhibition
%
% State 5  = activated macrophages (Ma)
% State 7  = pathogen (Pe)
% State 15 = nitric oxide (NO)
% State 17 = damage (D)
% State 18 = antibiotic concentration (Abx)

clear;
clc;
close all;

%% Load model parameters and initial conditions
data = struct();
[pars,Init] = load_pars_Init_Copeland_Edited(data);
kabx = pars(115);
kabxpe = pars(116);
xabxpe = pars(117);

%% Simulation time
time = (0:0.1:300)';

%% Example treatment settings
% These are MODEL-EXPLORATION values, not clinical doses.
% In your previous trajectory, damage saturated very early, so an
% intervention beginning at t = 24 is likely too late to reverse damage.
pathogenStart = 3.0;
pathogenEnd   = 72.0;
abxDoseRate   = 7.0*2;    % concentration/time while antibiotic is given

noStart = 3.0;
noEnd   = 120.0;
noRate  = 0.25;         % added NO clearance rate, 1/time

%% 1. No treatment
noTreatment.pathogen.enabled   = false;
noTreatment.pathogen.startTime = inf;
noTreatment.pathogen.endTime   = -inf;
noTreatment.pathogen.doseRate  = 0;

noTreatment.nitricOxide.enabled        = false;
noTreatment.nitricOxide.startTime      = inf;
noTreatment.nitricOxide.endTime        = -inf;
noTreatment.nitricOxide.inhibitionRate = 0;

%% 2. Antimicrobial only
antimicrobialOnly.pathogen.enabled   = true;
antimicrobialOnly.pathogen.startTime = pathogenStart;
antimicrobialOnly.pathogen.endTime   = pathogenEnd;
antimicrobialOnly.pathogen.doseRate  = abxDoseRate;

antimicrobialOnly.nitricOxide.enabled        = false;
antimicrobialOnly.nitricOxide.startTime      = inf;
antimicrobialOnly.nitricOxide.endTime        = -inf;
antimicrobialOnly.nitricOxide.inhibitionRate = 0;

%% 3. Nitric-oxide inhibition only
noOnly.pathogen.enabled   = false;
noOnly.pathogen.startTime = inf;
noOnly.pathogen.endTime   = -inf;
noOnly.pathogen.doseRate  = 0;

noOnly.nitricOxide.enabled        = true;
noOnly.nitricOxide.startTime      = noStart;
noOnly.nitricOxide.endTime        = noEnd;
noOnly.nitricOxide.inhibitionRate = noRate;

%% 4. Combined treatment
combined.pathogen.enabled   = true;
combined.pathogen.startTime = pathogenStart;
combined.pathogen.endTime   = pathogenEnd;
combined.pathogen.doseRate  = abxDoseRate;

combined.nitricOxide.enabled        = true;
combined.nitricOxide.startTime      = noStart;
combined.nitricOxide.endTime        = noEnd;
combined.nitricOxide.inhibitionRate = noRate;

%% Run all four scenarios
scenarioNames = { ...
    'No treatment'; ...
    'Antimicrobial only'; ...
    'NO inhibition only'; ...
    'Combined treatment'};

interventions = { ...
    noTreatment; ...
    antimicrobialOnly; ...
    noOnly; ...
    combined};

numberOfScenarios = numel(interventions);
tCell = cell(numberOfScenarios,1);
solCell = cell(numberOfScenarios,1);

for scenarioNumber = 1:numberOfScenarios
    [tCell{scenarioNumber},solCell{scenarioNumber}] = modelDriver( ...
        pars,Init,time,interventions{scenarioNumber});
end

%% State indices
maIndex = 5;
peIndex = 7;
noIndex = 15;
damageIndex = 17;
abxIndex = 18;

%% User-editable state-classification thresholds
% These are numerical model thresholds, not clinical criteria.
healthyPeThreshold = 0.01;
healthyMaThreshold = 0.01;
healthyDThreshold  = 0.01;

%% Calculate outcomes
peakPe = zeros(numberOfScenarios,1);
peAUC = zeros(numberOfScenarios,1);
peakNO = zeros(numberOfScenarios,1);
noAUC = zeros(numberOfScenarios,1);
peakMa = zeros(numberOfScenarios,1);
peakD = zeros(numberOfScenarios,1);
peakAbx = zeros(numberOfScenarios,1);
finalPe = zeros(numberOfScenarios,1);
finalNO = zeros(numberOfScenarios,1);
finalMa = zeros(numberOfScenarios,1);
finalD = zeros(numberOfScenarios,1);
finalAbx = zeros(numberOfScenarios,1);
classification = cell(numberOfScenarios,1);

for scenarioNumber = 1:numberOfScenarios
    tNow = tCell{scenarioNumber};
    solNow = solCell{scenarioNumber};

    peakPe(scenarioNumber) = max(solNow(:,peIndex));
    peAUC(scenarioNumber) = trapz(tNow,solNow(:,peIndex));
    peakNO(scenarioNumber) = max(solNow(:,noIndex));
    noAUC(scenarioNumber) = trapz(tNow,solNow(:,noIndex));
    peakMa(scenarioNumber) = max(solNow(:,maIndex));
    peakD(scenarioNumber) = max(solNow(:,damageIndex));
    peakAbx(scenarioNumber) = max(solNow(:,abxIndex));

    finalPe(scenarioNumber) = solNow(end,peIndex);
    finalNO(scenarioNumber) = solNow(end,noIndex);
    finalMa(scenarioNumber) = solNow(end,maIndex);
    finalD(scenarioNumber) = solNow(end,damageIndex);
    finalAbx(scenarioNumber) = solNow(end,abxIndex);

    pathogenIsLow = finalPe(scenarioNumber) <= healthyPeThreshold;
    macrophagesAreLow = finalMa(scenarioNumber) <= healthyMaThreshold;
    damageIsLow = finalD(scenarioNumber) <= healthyDThreshold;

    if pathogenIsLow && macrophagesAreLow && damageIsLow
        classification{scenarioNumber} = 'Healthy';
    elseif pathogenIsLow
        classification{scenarioNumber} = 'Aseptic inflammatory';
    else
        classification{scenarioNumber} = 'Septic/infected';
    end
end

resultsTable = table( ...
    string(scenarioNames(:)), ...
    peakPe,peAUC,peakNO,noAUC,peakMa,peakD,peakAbx, ...
    finalPe,finalNO,finalMa,finalD,finalAbx,string(classification), ...
    'VariableNames',{ ...
        'Scenario','PeakPe','PeAUC','PeakNO','NOAUC','PeakMa','PeakD', ...
        'PeakAbx','FinalPe','FinalNO','FinalMa','FinalD','FinalAbx', ...
        'FinalState'});

disp(resultsTable);

fprintf('\n================ INTERVENTION SETTINGS ================\n');
fprintf(['Antimicrobial PK/PD: start = %.3f, end = %.3f, ' ...
    'doseRate = %.6f, kabx = %.6f, kabxpe = %.6f, xabxpe = %.6f\n'], ...
    pathogenStart,pathogenEnd,abxDoseRate,kabx,kabxpe,xabxpe);
fprintf('NO inhibitor:  start = %.3f, end = %.3f, rate = %.6f 1/time\n', ...
    noStart,noEnd,noRate);
fprintf('Healthy thresholds: Pe <= %.4g, Ma <= %.4g, D <= %.4g\n', ...
    healthyPeThreshold,healthyMaThreshold,healthyDThreshold);
fprintf('=======================================================\n\n');

%% Plot Pe, NO, Ma, D, and Abx
figure('Name','Antimicrobial and NO Intervention','Color','w');
tiledlayout(3,2,'TileSpacing','compact','Padding','compact');

stateIndices = [peIndex,noIndex,maIndex,damageIndex,abxIndex];
yLabels = {'Pathogen, Pe','Nitric oxide, NO', ...
    'Activated macrophages, Ma','Damage, D','Antibiotic, Abx'};
titles = {'Pathogen response','Nitric-oxide response', ...
    'Activated-macrophage response','Damage response', ...
    'Antibiotic PK response'};

for plotNumber = 1:numel(stateIndices)
    nexttile;
    hold on;

    for scenarioNumber = 1:numberOfScenarios
        if scenarioNumber == 1
            lineStyle = '-';
        elseif scenarioNumber == 2
            lineStyle = '--';
        elseif scenarioNumber == 3
            lineStyle = ':';
        else
            lineStyle = '-.';
        end

        plot(tCell{scenarioNumber}, ...
            solCell{scenarioNumber}(:,stateIndices(plotNumber)), ...
            lineStyle,'LineWidth',2);
    end

    xline(pathogenStart,':');
    xline(noStart,':');

    xlabel('Time');
    ylabel(yLabels{plotNumber});
    title(titles{plotNumber});
    grid on;

    if plotNumber == 1
        legend(scenarioNames,'Location','best');
    end
end

sgtitle(sprintf([ ...
    'Antimicrobial PK/PD max kill = %.3f; NO-inhibition rate = %.3f'], ...
    kabxpe,noRate));

%% Save outputs
scriptFolder = fileparts(mfilename('fullpath'));
figureFile = fullfile(scriptFolder, ...
    'antimicrobial_no_intervention_comparison.png');
resultsFile = fullfile(scriptFolder, ...
    'antimicrobial_no_intervention_results.csv');

set(gcf,'PaperPositionMode','auto');
print(gcf,figureFile,'-dpng','-r300');
writetable(resultsTable,resultsFile);

fprintf('Saved figure: %s\n',figureFile);
fprintf('Saved results: %s\n',resultsFile);
