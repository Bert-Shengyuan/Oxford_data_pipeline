%% TCELL STRUCT NEURON COUNT THRESHOLD CHECK
% =========================================================================
% PURPOSE: Companion to data_size_check.m / analyze_neuron_count_thresholds.m.
% Instead of counting neurons from per-session '*_analysis_results.mat'
% spike_data arrays, this script counts neurons directly from the curated
% unit table in tcell_cellmetrics_struct.mat (one row per neuron), split
% by sorting-quality label (Good; Good+MUA).
%
% OUTPUTS:
%   1. Two per-neuron txt files (one line per neuron: session, region):
%        neurons_good.txt      - bc_bc_unitType == 'Good'
%        neurons_good_mua.txt  - bc_bc_unitType == 'Good' or 'MUA'
%   2. For each of the two label sets, a session x region neuron-count
%      matrix and the same three figures as analyze_neuron_count_thresholds.m
%      (availability heatmap, availability curves, count distributions),
%      with the availability-curve y-axis showing the raw number of
%      sessions meeting the threshold rather than a percentage.
% =========================================================================

clear; clc; close all;

%% CONFIGURATION
BASE_DIR = '/Users/shengyuancai/Downloads/Oxford_dataset';
TCELL_MAT = fullfile(BASE_DIR, 'tcell_cellmetrics_struct.mat');
OUTPUT_DIR = fullfile(BASE_DIR, 'tcell_threshold_analysis_results');

% Same sweep used in data_size_check.m for direct comparability
THRESHOLD_RANGE = [50, 60, 70, 80, 90, 95, 100, 120, 150, 170, 190, 200];

% Fixed display-region order (matches the region_mapping used in
% analyze_neuron_count_thresholds.m); brainRegion_names2 already stores
% these exact display names. 'other' is excluded - it is not a mapped
% brain region.
ALL_REGIONS = {'M1 Ctx', 'preM Ctx', 'OFC', 'mPFC', 'motorThal', ...
    'sensorThal', 'interThal', 'MDThal', 'Pulvinar', 'Striatum', ...
    'Hippocampus', 'Olf area', 'Hypothalamus'};

if ~exist(OUTPUT_DIR, 'dir')
    mkdir(OUTPUT_DIR);
end

%% LOAD TCELL STRUCT
fprintf('Loading %s ...\n', TCELL_MAT);
loaded = load(TCELL_MAT);
tcell_struct = loaded.tcell_struct;

unit_type = to_cellstr(tcell_struct.bc_bc_unitType);
region    = to_cellstr(tcell_struct.brainRegion_names2);
session   = to_cellstr(tcell_struct.sessiondate);

n_neurons = numel(unit_type);
assert(numel(region) == n_neurons && numel(session) == n_neurons, ...
    'bc_bc_unitType, brainRegion_names2 and sessiondate must have matching lengths.');

fprintf('Total neurons in tcell_struct: %d\n\n', n_neurons);

unit_type_upper = upper(strtrim(unit_type));
is_good = strcmp(unit_type_upper, 'GOOD');
is_mua  = strcmp(unit_type_upper, 'MUA');

good_idx     = is_good;
good_mua_idx = is_good | is_mua;

%% WRITE PER-NEURON TXT FILES (one line per neuron: session, region)
good_txt = fullfile(OUTPUT_DIR, 'neurons_good.txt');
good_mua_txt = fullfile(OUTPUT_DIR, 'neurons_good_mua.txt');

write_neuron_txt(good_txt, session(good_idx), region(good_idx));
write_neuron_txt(good_mua_txt, session(good_mua_idx), region(good_mua_idx));

fprintf('Wrote %d "Good" neurons to: %s\n', sum(good_idx), good_txt);
fprintf('Wrote %d "Good"+"MUA" neurons to: %s\n\n', sum(good_mua_idx), good_mua_txt);

%% RUN THRESHOLD ANALYSIS FOR EACH NEURON-QUALITY CRITERION
criteria = {
    'Good',     good_idx
    'Good_MUA', good_mua_idx
};

for c = 1:size(criteria, 1)
    label = criteria{c, 1};
    idx = criteria{c, 2};

    fprintf('==========================================================\n');
    fprintf('THRESHOLD ANALYSIS: %s neurons (n=%d)\n', label, sum(idx));
    fprintf('==========================================================\n');

    run_threshold_analysis(session(idx), region(idx), ALL_REGIONS, ...
        THRESHOLD_RANGE, fullfile(OUTPUT_DIR, label), label);
end

fprintf('\nAll outputs saved to: %s\n', OUTPUT_DIR);

%% ========================================================================
%  LOCAL FUNCTIONS
%  ========================================================================

function c = to_cellstr(x)
% Normalize a MATLAB cell/string/char field into a column cellstr.
if iscell(x)
    c = cellstr(x(:));
elseif isstring(x)
    c = cellstr(x(:));
elseif ischar(x)
    c = cellstr(x);
else
    error('to_cellstr:unsupportedType', ...
        'Unsupported type for text field: %s', class(x));
end
end

function write_neuron_txt(filepath, sessions, regions)
% One line per neuron: <session>\t<region>
fid = fopen(filepath, 'w');
assert(fid ~= -1, 'Could not open file for writing: %s', filepath);
cleanup = onCleanup(@() fclose(fid));

fprintf(fid, 'session\tregion\n');
for i = 1:numel(sessions)
    fprintf(fid, '%s\t%s\n', sessions{i}, regions{i});
end
end

function run_threshold_analysis(sessions, regions, all_regions, thresholds, output_dir, label)
% Build a [session x region] neuron-count matrix from per-neuron
% session/region labels, then reproduce the three
% analyze_neuron_count_thresholds.m figures from it.

if ~exist(output_dir, 'dir')
    mkdir(output_dir);
end

n_regions = numel(all_regions);
session_names = sort(unique(sessions));
n_sessions = numel(session_names);

%% Neuron count matrix: [n_sessions x n_regions]
neuron_count_matrix = zeros(n_sessions, n_regions);
for s = 1:n_sessions
    sess_mask = strcmp(sessions, session_names{s});
    for r = 1:n_regions
        neuron_count_matrix(s, r) = sum(sess_mask & strcmp(regions, all_regions{r}));
    end
end

%% Threshold availability (raw session counts + percentage)
n_thresholds = numel(thresholds);
availability_matrix = zeros(n_thresholds, n_regions);  % # sessions meeting threshold
percentage_matrix = zeros(n_thresholds, n_regions);    % % of sessions meeting threshold

for t = 1:n_thresholds
    for r = 1:n_regions
        availability_matrix(t, r) = sum(neuron_count_matrix(:, r) >= thresholds(t));
        percentage_matrix(t, r) = 100 * availability_matrix(t, r) / n_sessions;
    end
end

%% Summary statistics table
neuron_stats_table = table();
neuron_stats_table.Region = all_regions';
for r = 1:n_regions
    counts = neuron_count_matrix(:, r);
    counts_valid = counts(counts > 0);
    if isempty(counts_valid)
        counts_valid = 0;
    end
    neuron_stats_table.N_Sessions(r) = sum(counts > 0);
    neuron_stats_table.Mean(r) = mean(counts_valid);
    neuron_stats_table.Median(r) = median(counts_valid);
    neuron_stats_table.Min(r) = min(counts_valid);
    neuron_stats_table.Max(r) = max(counts_valid);
    neuron_stats_table.StdDev(r) = std(counts_valid);
end

threshold_table = array2table(availability_matrix, ...
    'VariableNames', matlab.lang.makeValidName(all_regions), ...
    'RowNames', arrayfun(@(x) sprintf('n>=%d', x), thresholds, 'UniformOutput', false));

fprintf('Sessions: %d | Regions: %d\n', n_sessions, n_regions);
disp(neuron_stats_table);

%% Figure 1: Heatmap of session availability counts
figure('Position', [100, 100, 1400, 800], 'Color', 'w');

imagesc(availability_matrix);
colormap(flipud(hot));
colorbar('FontSize', 14, 'FontWeight', 'bold');

xlabel('Brain Region', 'FontSize', 18, 'FontWeight', 'bold');
ylabel('Neuron Count Threshold', 'FontSize', 18, 'FontWeight', 'normal');
title({sprintf('%s neurons - Total Sessions: N=%d', strrep(label, '_', '+'), n_sessions)}, ...
      'FontSize', 20, 'FontWeight', 'bold');

set(gca, 'XTick', 1:n_regions, 'XTickLabel', all_regions, ...
    'XTickLabelRotation', 45, 'FontSize', 12, 'FontWeight', 'normal');
set(gca, 'YTick', 1:n_thresholds, 'YTickLabel', thresholds, ...
    'FontSize', 14, 'FontWeight', 'normal');

cmap = colormap(gca);
clim_vals = get(gca, 'CLim');
cdata_range = max(clim_vals(2) - clim_vals(1), eps);
for t = 1:n_thresholds
    for r = 1:n_regions
        value = availability_matrix(t, r);
        cmap_idx = max(1, min(size(cmap, 1), ...
            round(1 + (value - clim_vals(1)) / cdata_range * (size(cmap, 1) - 1))));
        box_color = cmap(cmap_idx, :);
        box_luminance = 0.299*box_color(1) + 0.587*box_color(2) + 0.114*box_color(3);
        if box_luminance < 0.5
            text_color = 'white';
        else
            text_color = 'black';
        end
        text(r, t, sprintf('%.0f', value), ...
            'HorizontalAlignment', 'center', 'FontSize', 10, ...
            'FontWeight', 'bold', 'Color', text_color);
    end
end

saveas(gcf, fullfile(output_dir, 'threshold_availability_heatmap.png'));

%% Figure 2: Availability curves - Y AXIS = raw session number (not %)
figure('Position', [150, 150, 1200, 700], 'Color', 'w');

hold on;
colors = lines(n_regions);

for r = 1:n_regions
    plot(thresholds, availability_matrix(:, r), '-o', ...
        'LineWidth', 2.5, 'MarkerSize', 8, ...
        'Color', colors(r, :), 'MarkerFaceColor', colors(r, :), ...
        'DisplayName', all_regions{r});
end

% Reference line at half of the total session count
plot(thresholds, (n_sessions/2)*ones(size(thresholds)), '--k', 'LineWidth', 2, ...
    'DisplayName', sprintf('50%% of sessions (N=%d)', n_sessions));

hold off;

xlabel('Minimum Neuron Count Threshold', 'FontSize', 18, 'FontWeight', 'bold');
ylabel('Number of Sessions Available', 'FontSize', 18, 'FontWeight', 'bold');
title(sprintf('%s neurons - Impact of Neuron Count Criteria on Session Availability', ...
    strrep(label, '_', '+')), 'FontSize', 20, 'FontWeight', 'bold');
legend('Location', 'eastoutside', 'FontSize', 12);
grid on;
set(gca, 'FontSize', 14, 'FontWeight', 'bold');
xlim([min(thresholds)-5, max(thresholds)+5]);
ylim([0, n_sessions*1.05]);

saveas(gcf, fullfile(output_dir, 'threshold_availability_curves.png'));

%% Figure 3: Neuron count distributions per region
figure('Position', [200, 200, 1600, 900], 'Color', 'w');

for r = 1:n_regions
    subplot(ceil(n_regions/4), 4, r);

    counts = neuron_count_matrix(:, r);
    counts_valid = counts(counts > 0);

    histogram(counts_valid, 'BinWidth', 10, 'FaceColor', colors(r, :), ...
        'EdgeColor', 'black', 'LineWidth', 1.5);

    hold on;
    yl = ylim;
    plot([50 50], yl, '--r', 'LineWidth', 2);  % Current pipeline threshold (n=50)
    hold off;

    title(sprintf('%s (n=%d)', all_regions{r}, length(counts_valid)), ...
        'FontSize', 14, 'FontWeight', 'bold');
    xlabel('Neuron Count', 'FontSize', 12);
    ylabel('Sessions', 'FontSize', 12);
    set(gca, 'FontSize', 11);
    grid on;
end

sgtitle(sprintf('%s Neurons: Count Distributions Across Regions', ...
    strrep(label, '_', '+')), 'FontSize', 22, 'FontWeight', 'bold');

saveas(gcf, fullfile(output_dir, 'neuron_count_distributions.png'));

%% Save data tables
writetable(neuron_stats_table, fullfile(output_dir, 'neuron_count_statistics.csv'));
writetable(threshold_table, fullfile(output_dir, 'threshold_availability.csv'), ...
    'WriteRowNames', true);

session_region_table = array2table(neuron_count_matrix, ...
    'VariableNames', matlab.lang.makeValidName(all_regions), ...
    'RowNames', session_names);
writetable(session_region_table, fullfile(output_dir, 'session_neuron_counts.csv'), ...
    'WriteRowNames', true);

fprintf('Saved figures/tables to: %s\n\n', output_dir);

end
