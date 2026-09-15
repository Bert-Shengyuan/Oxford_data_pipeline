%% NEURON COUNT THRESHOLD ANALYSIS - DEMONSTRATION SCRIPT
% =========================================================================
% PURPOSE: This script demonstrates systematic evaluation of how minimum
% neuron count criteria affect experimental session availability across
% brain regions in the Oxford neurophysiology dataset.
%
% SCIENTIFIC CONTEXT:
% The selection of neuron count thresholds represents a fundamental 
% experimental design decision with cascading implications for:
%   1. Statistical Power: Higher n_neuron → reduced variance, enhanced
%      signal detection capability
%   2. Sample Size: Stringent thresholds → fewer qualifying sessions,
%      potentially inadequate for population-level inference
%   3. Regional Coverage: Different regions exhibit heterogeneous neuron
%      yields, necessitating threshold optimization
%
% RECOMMENDED WORKFLOW:
%   Step 1: Execute comprehensive threshold sweep (30-150 neurons)
%   Step 2: Examine regional neuron count distributions
%   Step 3: Identify optimal threshold balancing power and coverage
%   Step 4: Validate current pipeline threshold (n=50) empirically
%
% AUTHOR: Senior Computational Neuroscience Laboratory
% DATE: December 2025
% =========================================================================

clear; clc; close all;

%% CONFIGURATION
% Specify the base directory containing your Oxford dataset
% CRITICAL: Modify this path to match your local system configuration
BASE_DIR = '/Users/shengyuancai/Downloads/Oxford_dataset';

% Define threshold range for systematic evaluation
% RATIONALE: 
%   - Lower bound (30): Minimum for reliable PCA/CCA (10-15 components)
%   - Current practice (50): Established in existing pipeline
%   - Upper bound (150): Theoretical maximum for single-region analyses
THRESHOLD_RANGE = [30, 40, 50, 60, 70, 80, 90,95,100, 120, 150,170,190,200];

% Output configuration
SAVE_RESULTS = true;
VERBOSE_OUTPUT = true;

%% EXECUTE THRESHOLD ANALYSIS
fprintf('==========================================================\n');
fprintf('OXFORD DATASET: NEURON COUNT THRESHOLD ANALYSIS\n');
fprintf('==========================================================\n\n');

fprintf('Initiating systematic threshold evaluation...\n');
fprintf('Base Directory: %s\n', BASE_DIR);
fprintf('Threshold Range: [%s]\n', num2str(THRESHOLD_RANGE));
fprintf('\n');

% Execute analysis function
results = analyze_neuron_count_thresholds(BASE_DIR, ...
    'ThresholdRange', THRESHOLD_RANGE, ...
    'SaveResults', SAVE_RESULTS, ...
    'Verbose', VERBOSE_OUTPUT);

%% POST-ANALYSIS: DETAILED EXAMINATION
fprintf('\n==========================================================\n');
fprintf('DETAILED POST-ANALYSIS EXAMINATION\n');
fprintf('==========================================================\n\n');

%% Analysis 1: Identify Regions with Consistent High Yields
fprintf('ANALYSIS 1: Regions with Robust Neuron Sampling\n');
fprintf('------------------------------------------------\n');
fprintf('Criterion: Mean neuron count ≥70 AND availability ≥80%% at n=50 threshold\n\n');

mean_counts = results.statistics_table.Mean;
region_names = results.statistics_table.Region;

% Find current threshold index (n=50)
current_threshold_idx = find(results.thresholds == 50);
if ~isempty(current_threshold_idx)
    availability_at_50 = results.percentage(current_threshold_idx, :);
    
    % Identify high-yield regions
    high_yield_regions = find(mean_counts >= 70 & availability_at_50' >= 80);
    
    if ~isempty(high_yield_regions)
        fprintf('High-yield regions identified:\n');
        for i = 1:length(high_yield_regions)
            idx = high_yield_regions(i);
            fprintf('  • %s: μ=%.1f neurons, %.0f%% availability\n', ...
                region_names{idx}, mean_counts(idx), availability_at_50(idx));
        end
    else
        fprintf('No regions meet high-yield criteria.\n');
    end
else
    fprintf('Current threshold (n=50) not found in analysis range.\n');
end

fprintf('\n');

%% Analysis 2: Evaluate Trade-offs for Alternative Thresholds
fprintf('ANALYSIS 2: Threshold Trade-off Quantification\n');
fprintf('-----------------------------------------------\n');

fprintf('Comparing alternative thresholds to current standard (n=50):\n\n');

% Calculate mean session availability for each threshold
mean_availability = mean(results.availability, 2);
mean_percentage = mean(results.percentage, 2);

% Create comparison table
comparison_table = table(results.thresholds', mean_availability, mean_percentage, ...
    'VariableNames', {'Threshold', 'Mean_Sessions', 'Mean_Percentage'});

comparison_table.Relative_to_50 = ...
    (comparison_table.Mean_Sessions / comparison_table.Mean_Sessions(results.thresholds == 50) - 1) * 100;

disp(comparison_table);
fprintf('\n');

% Identify threshold yielding maximum sessions while maintaining >70%% mean availability
suitable_idx = find(mean_percentage >= 70);
if ~isempty(suitable_idx)
    [~, best_idx] = max(mean_availability(suitable_idx));
    optimal_threshold = results.thresholds(suitable_idx(best_idx));
    
    fprintf('RECOMMENDATION: Optimal threshold = n≥%d neurons\n', optimal_threshold);
    fprintf('  • Maintains ≥70%% mean availability across regions\n');
    fprintf('  • Maximizes experimental coverage\n');
    
    if optimal_threshold ~= 50
        fprintf('\n');
        fprintf('CONSIDERATION: Current threshold (n=50) may benefit from adjustment.\n');
        fprintf('  Current: %.0f%% mean availability\n', mean_percentage(results.thresholds == 50));
        fprintf('  Optimal: %.0f%% mean availability\n', mean_percentage(results.thresholds == optimal_threshold));
        fprintf('  Difference: %+.0f sessions on average\n', ...
            mean_availability(results.thresholds == optimal_threshold) - ...
            mean_availability(results.thresholds == 50));
    end
else
    fprintf('WARNING: No threshold maintains ≥70%% mean availability.\n');
    fprintf('Consider: (1) Lower threshold, (2) Exclude low-yield regions, or\n');
    fprintf('          (3) Collect additional experimental sessions.\n');
end

fprintf('\n');

%% Analysis 3: Region-Specific Threshold Recommendations
fprintf('ANALYSIS 3: Region-Specific Threshold Recommendations\n');
fprintf('-----------------------------------------------------\n');
fprintf('Identifying individualized thresholds per region (≥80%% availability):\n\n');

for r = 1:length(region_names)
    region_availability = results.percentage(:, r);
    
    % Find maximum threshold maintaining ≥80% availability
    suitable_thresholds = results.thresholds(region_availability >= 80);
    
    if ~isempty(suitable_thresholds)
        max_threshold = max(suitable_thresholds);
        fprintf('  %8s: Can support n≥%3d (%.0f%% availability)\n', ...
            region_names{r}, max_threshold, ...
            region_availability(results.thresholds == max_threshold));
    else
        fprintf('  %8s: Cannot maintain 80%% availability at any threshold\n', ...
            region_names{r});
    end
end

fprintf('\n');

%% Analysis 4: Pairwise Region Coverage Matrix
fprintf('ANALYSIS 4: Pairwise Cross-Regional Analysis Coverage\n');
fprintf('------------------------------------------------------\n');
fprintf('Computing joint session availability for region pairs (n≥50):\n\n');

if ~isempty(current_threshold_idx)
    n_regions = length(region_names);
    pairwise_coverage = zeros(n_regions, n_regions);
    
    % For each region pair, count sessions where BOTH regions meet threshold
    for i = 1:n_regions
        for j = 1:n_regions
            if i ~= j
                % Sessions where both regions have ≥50 neurons
                both_sufficient = (results.neuron_counts(:, i) >= 50) & ...
                                  (results.neuron_counts(:, j) >= 50);
                pairwise_coverage(i, j) = sum(both_sufficient);
            end
        end
    end
    
    % Display as table
    pairwise_table = array2table(pairwise_coverage, ...
        'VariableNames', matlab.lang.makeValidName(region_names), ...
        'RowNames', region_names);
    
    fprintf('Pairwise session counts (both regions ≥50 neurons):\n');
    disp(pairwise_table);
    
    % Identify pairs with <20 joint sessions
    [row, col] = find(pairwise_coverage < 20 & pairwise_coverage > 0);
    if ~isempty(row)
        fprintf('\nWARNING: Region pairs with limited coverage (<20 sessions):\n');
        for k = 1:length(row)
            fprintf('  • %s ↔ %s: %d sessions\n', ...
                region_names{row(k)}, region_names{col(k)}, ...
                pairwise_coverage(row(k), col(k)));
        end
    end
else
    fprintf('Threshold n=50 not available in analysis range.\n');
end

fprintf('\n');

%% FINAL SUMMARY AND RECOMMENDATIONS
fprintf('==========================================================\n');
fprintf('EXECUTIVE SUMMARY AND RECOMMENDATIONS\n');
fprintf('==========================================================\n\n');

fprintf('DATASET CHARACTERISTICS:\n');
fprintf('  • Total sessions analyzed: %d\n', length(results.session_names));
fprintf('  • Regions identified: %d\n', length(region_names));
fprintf('  • Mean neuron count (across all regions/sessions): %.1f\n', ...
    mean(results.neuron_counts(results.neuron_counts > 0)));
fprintf('  • Median neuron count: %.1f\n', ...
    median(results.neuron_counts(results.neuron_counts > 0)));
fprintf('\n');

fprintf('CURRENT THRESHOLD EVALUATION (n=50):\n');
if ~isempty(current_threshold_idx)
    fprintf('  • Mean session availability: %.1f%%\n', mean_percentage(current_threshold_idx));
    fprintf('  • Minimum regional availability: %.1f%%\n', ...
        min(results.percentage(current_threshold_idx, :)));
    fprintf('  • Maximum regional availability: %.1f%%\n', ...
        max(results.percentage(current_threshold_idx, :)));
else
    fprintf('  • Not evaluated in current threshold range\n');
end
fprintf('\n');

fprintf('METHODOLOGICAL RECOMMENDATIONS:\n');
fprintf('  1. THRESHOLD SELECTION: Balance statistical power (higher n)\n');
fprintf('     with experimental coverage (sufficient sessions)\n');
fprintf('  2. REGIONAL HETEROGENEITY: Consider region-specific thresholds\n');
fprintf('     for analyses not requiring strict parity\n');
fprintf('  3. PAIRWISE ANALYSES: Verify joint availability for CCA studies\n');
fprintf('  4. POWER ANALYSIS: Conduct post-hoc power calculations to validate\n');
fprintf('     that selected threshold provides adequate sensitivity\n');
fprintf('\n');

fprintf('NEXT STEPS:\n');
fprintf('  □ Review generated visualizations in: %s/threshold_analysis_results/\n', BASE_DIR);
fprintf('  □ Examine neuron_count_distributions.png for regional sampling patterns\n');
fprintf('  □ Consult threshold_availability_heatmap.png for comprehensive overview\n');
fprintf('  □ Consider adjusting pipeline threshold based on empirical findings\n');
fprintf('  □ Document threshold selection rationale in methods section\n');
fprintf('\n');

fprintf('==========================================================\n');
fprintf('Analysis complete. Results saved.\n');
fprintf('==========================================================\n');

%% OPTIONAL: Export Results to Workspace
fprintf('\nResults structure exported to workspace variable: ''results''\n');
fprintf('Access components via:\n');
fprintf('  - results.neuron_counts     : [sessions × regions] neuron count matrix\n');
fprintf('  - results.availability      : [thresholds × regions] session counts\n');
fprintf('  - results.percentage        : [thresholds × regions] availability %%\n');
fprintf('  - results.statistics_table  : Descriptive statistics per region\n');
fprintf('  - results.threshold_table   : Availability matrix as formatted table\n');
fprintf('\n');