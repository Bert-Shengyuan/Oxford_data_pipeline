function threshold_results = analyze_neuron_count_thresholds(base_dir, varargin)
% ANALYZE_NEURON_COUNT_THRESHOLDS Systematic analysis of session availability 
% across varying neuron count thresholds per region
%
% PURPOSE:
%   This function addresses a fundamental experimental design question: how
%   does the minimum neuron count criterion affect the number of viable
%   sessions available for cross-regional analysis? The current pipeline
%   employs n=50 neurons per region, but the optimality of this threshold
%   requires empirical validation across the entire dataset.
%
% THEORETICAL FRAMEWORK:
%   The threshold selection represents a classical statistical trade-off:
%   - Higher thresholds (e.g., n≥80): Enhanced statistical power per session,
%     reduced sampling variance, but potentially inadequate sample size
%   - Lower thresholds (e.g., n≥30): Increased session count, broader
%     coverage, but elevated measurement uncertainty per session
%
% USAGE:
%   results = analyze_neuron_count_thresholds(base_dir)
%   results = analyze_neuron_count_thresholds(base_dir, 'ThresholdRange', [30 40 50 60 70 80 100])
%   results = analyze_neuron_count_thresholds(base_dir, 'SaveResults', true)
%
% INPUT:
%   base_dir       - Path to Oxford dataset directory containing 
%                    'session_analysis_results' subdirectory
%
% OPTIONAL PARAMETERS (Name-Value pairs):
%   'ThresholdRange' - Vector of neuron count thresholds to test
%                      Default: [30 40 50 60 70 80 100 120 150]
%   'SaveResults'    - Boolean flag to save output tables and figures
%                      Default: true
%   'OutputDir'      - Directory for saving results
%                      Default: [base_dir '/threshold_analysis_results']
%   'Verbose'        - Display detailed progress information
%                      Default: true
%
% OUTPUT:
%   threshold_results - Structure containing:
%       .neuron_counts     - [n_sessions × n_regions] matrix of neuron counts
%       .session_names     - Cell array of session identifiers
%       .region_names      - Cell array of region identifiers  
%       .thresholds        - Vector of tested threshold values
%       .availability      - [n_thresholds × n_regions] session count matrix
%       .percentage        - [n_thresholds × n_regions] percentage matrix
%       .summary_table     - Formatted table for publication
%
% EXAMPLE:
%   base_dir = '/Users/shengyuancai/Downloads/Oxford_dataset';
%   results = analyze_neuron_count_thresholds(base_dir, ...
%       'ThresholdRange', [30 50 70 100], ...
%       'SaveResults', true);
%
% METHODOLOGY:
%   1. Systematic file enumeration: Identify all *_analysis_results.mat files
%   2. Data extraction: Parse region_data.regions.REGIONNAME.spike_data
%   3. Neuron counting: Determine dimensions [trial × neuron × time]
%   4. Threshold evaluation: For each (threshold, region) pair, compute
%      the cardinality of sessions satisfying the minimum neuron criterion
%   5. Statistical reporting: Generate comprehensive tables and visualizations
%
% AUTHOR: Senior Computational Neuroscience Laboratory
% DATE: December 2025
% VERSION: 1.0

%% Parse Input Arguments
p = inputParser;
addRequired(p, 'base_dir', @ischar);
addParameter(p, 'ThresholdRange', [30 40 50 60 70 80 100 120 150], @isnumeric);
addParameter(p, 'SaveResults', true, @islogical);
addParameter(p, 'OutputDir', '', @ischar);
addParameter(p, 'Verbose', true, @islogical);
parse(p, base_dir, varargin{:});

thresholds = p.Results.ThresholdRange;
save_results = p.Results.SaveResults;
verbose = p.Results.Verbose;

% Determine output directory
if isempty(p.Results.OutputDir)
    output_dir = fullfile(base_dir, 'threshold_analysis_results');
else
    output_dir = p.Results.OutputDir;
end

if save_results && ~exist(output_dir, 'dir')
    mkdir(output_dir);
end

%% Identify All Analysis Results Files
results_dir = fullfile(base_dir, 'sessions_spont_miss_long_results');
if ~exist(results_dir, 'dir')
    error('Session analysis results directory not found: %s', results_dir);
end

% Systematic file enumeration
mat_files = dir(fullfile(results_dir, '*_analysis_results.mat'));
n_sessions = length(mat_files);

if verbose
    fprintf('\n========================================\n');
    fprintf('NEURON COUNT THRESHOLD ANALYSIS\n');
    fprintf('========================================\n\n');
    fprintf('Analysis directory: %s\n', results_dir);
    fprintf('Sessions identified: %d\n', n_sessions);
    fprintf('Thresholds to test: %s\n', mat2str(thresholds));
    fprintf('\n');
end

if n_sessions == 0
    error('No analysis results files found in: %s', results_dir);
end

%% Extract Neuron Counts from All Sessions
if verbose
    fprintf('Phase I: Extracting neuron counts from session data...\n');
    fprintf('---------------------------------------------------\n');
end

% Initialize data structures
session_names = cell(n_sessions, 1);
all_regions = {};  % Will collect unique region names
neuron_count_data = cell(n_sessions, 1);  % Store as cell initially

for i = 1:n_sessions
    session_file = fullfile(results_dir, mat_files(i).name);
    
    % Extract session name (remove '_analysis_results.mat' suffix)
    [~, session_name, ~] = fileparts(mat_files(i).name);
    session_names{i} = strrep(session_name, '_analysis_results', '');
    
    if verbose
        fprintf('  [%3d/%3d] Processing: %s\n', i, n_sessions, session_names{i});
    end
    
    try
        % Load session data (handle both old and new MAT formats)
        try
            data = load(session_file);
        catch
            % If standard load fails, might be v7.3 format
            warning('Standard load failed for %s, attempting -v7.3 load', session_names{i});
            data = load(session_file, '-mat');
        end
        
        % Extract region data
        if ~isfield(data, 'region_data') || ~isfield(data.region_data, 'regions')
            warning('Session %s: region_data structure not found', session_names{i});
            continue;
        end
        
        regions = fieldnames(data.region_data.regions);
        session_neuron_counts = struct();
        
        % Iterate through all regions in this session
        for r = 1:length(regions)
            region_name = regions{r};
            region_data = data.region_data.regions.(region_name);
            
            % Extract spike_data dimensions
            % CRITICAL: spike_data format is [time * neurons * trial]
            if isfield(region_data, 'spike_data')
                spike_data = region_data.spike_data;
                n_neurons = size(spike_data, 2);  % First dimension = neurons
                
                session_neuron_counts.(region_name) = n_neurons;
                
                % Accumulate unique region names
                if ~ismember(region_name, all_regions)
                    all_regions{end+1} = region_name;
                end
            end
        end
        
        neuron_count_data{i} = session_neuron_counts;
        
    catch ME
        warning('Error processing session %s: %s', session_names{i}, ME.message);
        continue;
    end
end

% Sort region names alphabetically for consistent presentation
all_regions = sort(all_regions);
n_regions = length(all_regions);

if verbose
    fprintf('\nData extraction complete.\n');
    fprintf('Total regions identified: %d\n', n_regions);
    fprintf('Regions: %s\n\n', strjoin(all_regions, ', '));
end

%% Construct Neuron Count Matrix
% Convert cell array to structured matrix: [n_sessions × n_regions]
neuron_count_matrix = zeros(n_sessions, n_regions);

for i = 1:n_sessions
    if ~isempty(neuron_count_data{i})
        for r = 1:n_regions
            region_name = all_regions{r};
            if isfield(neuron_count_data{i}, region_name)
                neuron_count_matrix(i, r) = neuron_count_data{i}.(region_name);
            else
                neuron_count_matrix(i, r) = 0;  % Region not present in this session
            end
        end
    end
end

%% Threshold Availability Analysis
if verbose
    fprintf('Phase II: Evaluating session availability across thresholds...\n');
    fprintf('------------------------------------------------------------\n');
end

n_thresholds = length(thresholds);
availability_matrix = zeros(n_thresholds, n_regions);
percentage_matrix = zeros(n_thresholds, n_regions);

for t = 1:n_thresholds
    threshold = thresholds(t);
    
    for r = 1:n_regions
        % Count sessions where this region meets the threshold
        n_available = sum(neuron_count_matrix(:, r) >= threshold);
        availability_matrix(t, r) = n_available;
        percentage_matrix(t, r) = (n_available / n_sessions) * 100;
    end
    
    if verbose
        fprintf('  Threshold n≥%3d: Mean availability = %.1f%% (range: %.1f%% - %.1f%%)\n', ...
            threshold, mean(percentage_matrix(t, :)), ...
            min(percentage_matrix(t, :)), max(percentage_matrix(t, :)));
    end
end

%% Generate Summary Tables
% Table 1: Neuron count statistics per region
neuron_stats_table = table();
neuron_stats_table.Region = all_regions';

for r = 1:n_regions
    counts = neuron_count_matrix(:, r);
    counts_valid = counts(counts > 0);  % Exclude sessions where region absent
    
    neuron_stats_table.N_Sessions(r) = length(counts_valid);
    neuron_stats_table.Mean(r) = mean(counts_valid);
    neuron_stats_table.Median(r) = median(counts_valid);
    neuron_stats_table.Min(r) = min(counts_valid);
    neuron_stats_table.Max(r) = max(counts_valid);
    neuron_stats_table.StdDev(r) = std(counts_valid);
end

% Table 2: Session availability across thresholds
threshold_table = array2table(availability_matrix, ...
    'VariableNames', matlab.lang.makeValidName(all_regions), ...
    'RowNames', arrayfun(@(x) sprintf('n≥%d', x), thresholds, 'UniformOutput', false));

%% Generate Visualizations
if save_results
    
    % Figure 1: Heatmap of session availability percentages
    figure('Position', [100, 100, 1400, 800], 'Color', 'w');
    
    imagesc(availability_matrix);
    colormap(flipud(hot));
    colorbar('FontSize', 14, 'FontWeight', 'bold');
    
    % Annotations
    xlabel('Brain Region', 'FontSize', 18, 'FontWeight', 'bold');
    ylabel('Neuron Count Threshold', 'FontSize', 18, 'FontWeight', 'normal');
    title({sprintf('Total Sessions: N=%d', n_sessions)}, ...
          'FontSize', 20, 'FontWeight', 'bold');
    
    % Axis formatting
    set(gca, 'XTick', 1:n_regions, 'XTickLabel', all_regions, ...
        'XTickLabelRotation', 45, 'FontSize', 12, 'FontWeight', 'normal');
    set(gca, 'YTick', 1:n_thresholds, 'YTickLabel', thresholds, ...
        'FontSize', 14, 'FontWeight', 'normal');
    
    % Add percentage text annotations
    for t = 1:n_thresholds
        for r = 1:n_regions
            text(r, t, sprintf('%.0f', availability_matrix(t, r)), ...
                'HorizontalAlignment', 'center', 'FontSize', 10, ...
                'FontWeight', 'bold', 'Color', 'black');
        end
    end
    
    saveas(gcf, fullfile(output_dir, 'threshold_availability_heatmap.png'));
    %saveas(gcf, fullfile(output_dir, 'threshold_availability_heatmap.fig'));
    
    % Figure 2: Line plot showing availability curves
    figure('Position', [150, 150, 1200, 700], 'Color', 'w');
    
    hold on;
    colors = lines(n_regions);
    
    for r = 1:n_regions
        plot(thresholds, percentage_matrix(:, r), '-o', ...
            'LineWidth', 2.5, 'MarkerSize', 8, ...
            'Color', colors(r, :), 'MarkerFaceColor', colors(r, :), ...
            'DisplayName', all_regions{r});
    end
    
    % Reference line at 50% availability
    plot(thresholds, 50*ones(size(thresholds)), '--k', 'LineWidth', 2, ...
        'DisplayName', '50% Threshold');
    
    hold off;
    
    xlabel('Minimum Neuron Count Threshold', 'FontSize', 18, 'FontWeight', 'bold');
    ylabel('Session Availability (%)', 'FontSize', 18, 'FontWeight', 'bold');
    title('Impact of Neuron Count Criteria on Experimental Coverage', ...
        'FontSize', 20, 'FontWeight', 'bold');
    legend('Location', 'eastoutside', 'FontSize', 12);
    grid on;
    set(gca, 'FontSize', 14, 'FontWeight', 'bold');
    xlim([min(thresholds)-5, max(thresholds)+5]);
    ylim([0, 105]);
    
    saveas(gcf, fullfile(output_dir, 'threshold_availability_curves.png'));
    %saveas(gcf, fullfile(output_dir, 'threshold_availability_curves.fig'));
    
    % Figure 3: Neuron count distributions per region
    figure('Position', [200, 200, 1600, 900], 'Color', 'w');
    
    for r = 1:n_regions
        subplot(ceil(n_regions/4), 4, r);
        
        counts = neuron_count_matrix(:, r);
        counts_valid = counts(counts > 0);
        
        histogram(counts_valid, 'BinWidth', 10, 'FaceColor', colors(r, :), ...
            'EdgeColor', 'black', 'LineWidth', 1.5);
        
        % Add threshold reference lines
        hold on;
        yl = ylim;
        plot([50 50], yl, '--r', 'LineWidth', 2);  % Current threshold
        hold off;
        
        title(sprintf('%s (n=%d)', all_regions{r}, length(counts_valid)), ...
            'FontSize', 14, 'FontWeight', 'bold');
        xlabel('Neuron Count', 'FontSize', 12);
        ylabel('Sessions', 'FontSize', 12);
        set(gca, 'FontSize', 11);
        grid on;
    end
    
    sgtitle('Neuron Count Distributions Across Regions', ...
        'FontSize', 22, 'FontWeight', 'bold');
    
    saveas(gcf, fullfile(output_dir, 'neuron_count_distributions.png'));
    %saveas(gcf, fullfile(output_dir, 'neuron_count_distributions.fig'));
end

%% Save Data Tables
if save_results
    writetable(neuron_stats_table, fullfile(output_dir, 'neuron_count_statistics.csv'));
    writetable(threshold_table, fullfile(output_dir, 'threshold_availability.csv'), ...
        'WriteRowNames', true);
    
    % Save detailed session-by-region matrix
    session_region_table = array2table(neuron_count_matrix, ...
        'VariableNames', matlab.lang.makeValidName(all_regions), ...
        'RowNames', session_names);
    writetable(session_region_table, fullfile(output_dir, 'session_neuron_counts.csv'), ...
        'WriteRowNames', true);
end

%% Compile Output Structure
threshold_results = struct();
threshold_results.neuron_counts = neuron_count_matrix;
threshold_results.session_names = session_names;
threshold_results.region_names = all_regions;
threshold_results.thresholds = thresholds;
threshold_results.availability = availability_matrix;
threshold_results.percentage = percentage_matrix;
threshold_results.statistics_table = neuron_stats_table;
threshold_results.threshold_table = threshold_table;

%% Generate Interpretive Report
if verbose
    fprintf('\n========================================\n');
    fprintf('THRESHOLD ANALYSIS SUMMARY\n');
    fprintf('========================================\n\n');
    
    fprintf('DESCRIPTIVE STATISTICS (Neuron Counts per Region):\n');
    fprintf('--------------------------------------------------\n');
    disp(neuron_stats_table);
    fprintf('\n');
    
    fprintf('SESSION AVAILABILITY MATRIX (Count):\n');
    fprintf('-----------------------------------\n');
    disp(threshold_table);
    fprintf('\n');
    
    fprintf('CRITICAL INSIGHTS:\n');
    fprintf('-----------------\n');
    
    % Identify optimal threshold (maximizing coverage while maintaining >30 sessions)
    mean_availability = mean(availability_matrix, 2);
    suitable_thresholds = thresholds(mean_availability >= 30);
    
    if ~isempty(suitable_thresholds)
        optimal_threshold = max(suitable_thresholds);
        fprintf('• Recommended threshold: n≥%d (maintains ≥30 sessions on average)\n', ...
            optimal_threshold);
    else
        fprintf('• WARNING: No threshold maintains ≥30 sessions across all regions\n');
    end
    
    % Current threshold evaluation
    current_idx = find(thresholds == 50);
    if ~isempty(current_idx)
        current_availability = mean(percentage_matrix(current_idx, :));
        fprintf('• Current threshold (n=50): %.1f%% average availability\n', ...
            current_availability);
    end
    
    % Identify regions with limited coverage
    low_coverage_regions = all_regions(neuron_stats_table.Mean < 50);
    if ~isempty(low_coverage_regions)
        fprintf('• Regions with limited neuron sampling: %s\n', ...
            strjoin(low_coverage_regions, ', '));
    end
    
    fprintf('\n');
end

if save_results && verbose
    fprintf('Results saved to: %s\n', output_dir);
    fprintf('  • threshold_availability_heatmap.png\n');
    fprintf('  • threshold_availability_curves.png\n');
    fprintf('  • neuron_count_distributions.png\n');
    fprintf('  • neuron_count_statistics.csv\n');
    fprintf('  • threshold_availability.csv\n');
    fprintf('  • session_neuron_counts.csv\n');
    fprintf('\n');
end

end
