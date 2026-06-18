function plot_virtual_patients(csvFile)
%PLOT_VIRTUAL_PATIENTS Plot virtual patients by kpg and Pe_0.
%
% Usage:
%   plot_virtual_patients("virtual_patients_50.csv")

    if nargin < 1 || isempty(csvFile)
        csvFile = "virtual_patients_500.csv";
    end

    T = readtable(csvFile);

    kpg = T.kpg;
    Pe0 = T.Pe_0;
    lab = string(T.label);

    idx_non          = lab == "non_septic";
    idx_septic       = lab == "septic";
    idx_aseptic      = lab == "aseptic";
    idx_unclassified = lab == "unclassified";
    idx_failed       = lab == "failed" | isnan(kpg) | isnan(Pe0);

    figure('Color', 'w');
    hold on;
    grid on;
    box on;

    % Same visual idea as VP_graph.m, but with all labels included.
    scatter(kpg(idx_non),          Pe0(idx_non),          45, [0.60 0.85 1.00], 'filled');
    scatter(kpg(idx_septic),       Pe0(idx_septic),       45, 'r',              'filled');
    scatter(kpg(idx_aseptic),      Pe0(idx_aseptic),      45, [0.70 0.40 0.90], 'filled');
    scatter(kpg(idx_unclassified), Pe0(idx_unclassified), 45, [0.50 0.50 0.50], 'filled');
    scatter(kpg(idx_failed),       Pe0(idx_failed),       45, [1.00 0.50 0.00], 'filled');

    xlabel('K_{pg}', 'Interpreter', 'tex');
    ylabel('Pe_0', 'Interpreter', 'tex');
    title('Virtual Patients');

    legend({'Non-septic', 'Septic', 'Aseptic', 'Unclassified', 'Failed'}, ...
           'Location', 'best');

    xlim([0 2]);
    ylim([0 2]);
    set(gca, 'FontSize', 12);

    hold off;
end
