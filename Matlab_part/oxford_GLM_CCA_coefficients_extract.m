function oxford_glm_cca_coefficients_extraction()
% OXFORD_GLM_CCA_COEFFICIENTS_EXTRACTION 
% Fits GLM to relate neuronal PSTH to CCA latent variables for Oxford dataset
%
% THEORETICAL FRAMEWORK:
% This function performs General Linear Model fitting to elucidate how individual
% neurons contribute to the canonical correlation patterns. The GLM establishes
% a mathematical relationship:
%
%   $\mathbf{L}(t) = \sum_i \beta_i \cdot \text{FR}_i(t) + \boldsymbol{\varepsilon}$
%
% where $\mathbf{L}(t)$ represents the CCA latent variable, $\beta_i$ denote the 
% GLM coefficients quantifying each neuron's contribution, and $\text{FR}_i(t)$ 
% represents the firing rate of neuron $i$.
%
% ANATOMICAL ORDERING CONVENTION:
% All region pairs follow a strict anatomical hierarchy:
%   Cortical: mPFC, ORB, MOp, MOs, OLF
%   Striatal & Limbic: STR, STRv, HIPP
%   Thalamic: MD, LP, VALVM, VPMPO, ILM
%   Hypothalamic: HY
%   Fiber tracts: fiber
%   Other: other
%
% DATA STRUCTURE (from unified session results files):
%   session_data.region_data.regions.{REGION}.spike_data  [n_trials × n_neurons × n_timepoints]
%   session_data.region_data.regions.{REGION}.selected_neurons  [1 × target_neurons]
%   session_data.cca_results.pair_results{i}.projections
%
% USAGE:
%   oxford_glm_cca_coefficients_extraction()

    fprintf('=== Oxford Dataset: GLM Analysis of CCA Latent Variables ===\n\n');
    
    %% =====================================================================
    %  CONFIGURATION SECTION
    %  =====================================================================
    
    % Base directory for Oxford dataset
    base_results_dir = '/Users/shengyuancai/Downloads/Oxford_dataset/';
    
    % Data source: Choose between task conditions
    % Options: 'sessions_cued_hit_long_results' or 'sessions_spont_short_results'
    data_source = 'sessions_spont_miss_long_results';
    
    % Full path to session results
    session_results_dir = fullfile(base_results_dir, data_source);
    
    % Analysis parameters
    component_to_analyze = 1;    % Which CCA component to analyze (1-indexed)
    min_neurons_threshold = 30;  % Minimum neurons required for reliable GLM fitting
    
    %% =====================================================================
    %  ANATOMICAL ORDERING DEFINITION
    %  =====================================================================
    
    ANATOMICAL_ORDER = {
        'mPFC', 'ORB', 'MOp', 'MOs', 'OLF', ...  % Cortical regions
        'STR', 'STRv', 'HIPP', ...               % Striatal & limbic
        'MD', 'LP', 'VALVM', 'VPMPO', 'ILM', ... % Thalamic nuclei
        'HY', ...                                 % Hypothalamic
        'fiber', ...                              % Fiber tracts
        'other'                                   % Catch-all category
    };
    
    %% =====================================================================
    %  REGION PAIRS OF INTEREST
    %  =====================================================================
    %  Focus on STR, MOs, and MOp paired with other regions.
    
    % MOs-focused pairs
    % MOs_pairs = {
    %     'mPFC_MOs', 'ORB_MOs', 'MOp_MOs', ...
    %     'MOs_STR', 'MOs_MD', 'MOs_LP'
    % };
    % 
    % % MOp-focused pairs
    % MOp_pairs = {
    %     'mPFC_MOp', 'ORB_MOp', 'MOp_MOs', ...
    %     'MOp_OLF', 'MOp_STR', 'MOp_MD', 'MOp_LP'
    % };
    
    % STR-focused pairs
    STR_pairs = {
        'mPFC_STR', 'ORB_STR', 'MOp_STR', ...
        'MOs_STR', 'OLF_STR', 'STR_STRv','STR_MD', 'STR_LP','STR_VALVM','STR_VPMPO','STR_ILM','STR_HY'
    };
    
    % Combine all unique pairs
    all_target_pairs = unique([STR_pairs]);
    
    fprintf('Configuration:\n');
    fprintf('  Data source: %s\n', data_source);
    fprintf('  Session results directory: %s\n', session_results_dir);
    fprintf('  Analyzing CCA component: %d\n', component_to_analyze);
    fprintf('  Minimum neurons threshold: %d\n', min_neurons_threshold);
    fprintf('  Target region pairs: %d\n', length(all_target_pairs));
    fprintf('  Pairs: %s\n\n', strjoin(all_target_pairs, ', '));
    
    %% =====================================================================
    %  OUTPUT DIRECTORY SETUP
    %  =====================================================================
    
    output_dir = fullfile(session_results_dir, ...
        sprintf('GLM_Analysis_Component_%d', component_to_analyze));
    if ~exist(output_dir, 'dir')
        mkdir(output_dir);
    end
    fprintf('Output directory: %s\n\n', output_dir);
    
    %% =====================================================================
    %  DISCOVER AVAILABLE SESSIONS
    %  =====================================================================
    
    % Session files follow pattern: {session}_session_results.mat or {session}_analysis_results.mat
    session_files = dir(fullfile(session_results_dir, '*_session_results.mat'));
    if isempty(session_files)
        session_files = dir(fullfile(session_results_dir, '*_analysis_results.mat'));
    end
    fprintf('Found %d sessions with results\n\n', length(session_files));
    
    % Initialize storage for all GLM results
    all_glm_results = struct();
    all_glm_results.component_analyzed = component_to_analyze;
    all_glm_results.analysis_timestamp = datestr(now);
    all_glm_results.data_source = data_source;
    all_glm_results.anatomical_order = ANATOMICAL_ORDER;
    all_glm_results.target_pairs = all_target_pairs;
    all_glm_results.region_pairs = {};
    
    %% =====================================================================
    %  PROCESS EACH SESSION
    %  =====================================================================
    
    for session_idx = 1:length(session_files)
        filename = session_files(session_idx).name;
        % Extract session name (remove suffix)
        session_name = regexprep(filename, '_(session|analysis)_results\.mat$', '');
        
        fprintf('Processing session %d/%d: %s\n', session_idx, length(session_files), session_name);
        
        try
            % Load unified session results
            session_file = fullfile(session_results_dir, filename);
            fprintf('  Loading session results...\n');
            loaded_data = load(session_file);
            
            % Handle different possible data structures
            if isfield(loaded_data, 'region_data')
                region_data = loaded_data.region_data;
            elseif isfield(loaded_data, 'session_data') && isfield(loaded_data.session_data, 'region_data')
                region_data = loaded_data.session_data.region_data;
            else
                fprintf('  Warning: region_data not found, skipping session\n');
                continue;
            end
            
            if isfield(loaded_data, 'cca_results')
                cca_results = loaded_data.cca_results;
            elseif isfield(loaded_data, 'session_data') && isfield(loaded_data.session_data, 'cca_results')
                cca_results = loaded_data.session_data.cca_results;
            else
                fprintf('  Warning: cca_results not found, skipping session\n');
                continue;
            end
            
            % Process each region pair from this session
            n_pairs = length(cca_results.pair_results);
            fprintf('  Found %d region pairs in session\n', n_pairs);
            
            for pair_idx = 1:n_pairs
                pair_result = cca_results.pair_results{pair_idx};
                region_pair_name = cca_results.region_pairs{pair_idx};
                
                % Check if this pair is in our target list (try both orderings)
                if ~ismember(region_pair_name, all_target_pairs)
                    regions_split = strsplit(region_pair_name, '_');
                    reversed_name = sprintf('%s_%s', regions_split{2}, regions_split{1});
                    if ~ismember(reversed_name, all_target_pairs)
                        continue;  % Skip pairs not in our focus list
                    end
                end
                
                % Reorder to match anatomical convention
                ordered_pair_name = reorder_pair_anatomically(region_pair_name, ANATOMICAL_ORDER);
                
                fprintf('    Analyzing pair: %s\n', ordered_pair_name);
                
                % Parse region names from the ORIGINAL pair name (as stored in CCA results)
                regions = strsplit(region_pair_name, '_');
                region1_name = pair_result.region_i;
                region2_name = pair_result.region_j;
                
                % Check if we have enough significant components
                if ~isfield(pair_result, 'significant_components') || ...
                   length(pair_result.significant_components) < component_to_analyze
                    fprintf('      Insufficient significant components, skipping\n');
                    continue;
                end
                
                % Extract the canonical projections for the specified component
                if ~isfield(pair_result, 'projections') || ...
                   ~isfield(pair_result.projections, 'components') || ...
                   pair_result.projections.n_components < component_to_analyze
                    fprintf('      Canonical projections not available, skipping\n');
                    continue;
                end
                
                comp_projections = pair_result.projections.components{component_to_analyze};
                
                % Extract latent variables (canonical variates)
                % Reshape to column vectors for GLM
                latent1 = reshape(comp_projections.region_i_trials', [], 1);
                latent2 = reshape(comp_projections.region_j_trials', [], 1);
                
                % Get firing rate data from region_data structure
                if ~isfield(region_data.regions, region1_name) || ...
                   ~isfield(region_data.regions, region2_name)
                    fprintf('      Region data not available for both regions, skipping\n');
                    continue;
                end
                
                % Extract spike_data for each region
                % Dimensions: [n_trials × n_neurons × n_timepoints]
                region1_spike_data = region_data.regions.(region1_name).spike_data;
                region2_spike_data = region_data.regions.(region2_name).spike_data;
                
                % Get the neuron indices that were used in CCA
                % These are stored in pair_result from CCA analysis
                if isfield(pair_result, 'selected_neurons_i')
                    selected_neurons1 = pair_result.selected_neurons_i;
                    selected_neurons2 = pair_result.selected_neurons_j;
                elseif isfield(region_data.regions.(region1_name), 'selected_neurons')
                    selected_neurons1 = region_data.regions.(region1_name).selected_neurons;
                    selected_neurons2 = region_data.regions.(region2_name).selected_neurons;
                else
                    fprintf('      Selected neurons indices not found, using all neurons\n');
                    selected_neurons1 = 1:size(region1_spike_data, 2);
                    selected_neurons2 = 1:size(region2_spike_data, 2);
                end
                
                % Extract only the selected neurons
                region1_trial_data = region1_spike_data(:, selected_neurons1, :);
                region2_trial_data = region2_spike_data(:, selected_neurons2, :);
                
                % Verify data dimensions
                [n_trials1, n_neurons1, n_timepoints1] = size(region1_trial_data);
                [n_trials2, n_neurons2, n_timepoints2] = size(region2_trial_data);
                
                fprintf('      Region 1 (%s): %d trials, %d neurons, %d timepoints\n', ...
                        region1_name, n_trials1, n_neurons1, n_timepoints1);
                fprintf('      Region 2 (%s): %d trials, %d neurons, %d timepoints\n', ...
                        region2_name, n_trials2, n_neurons2, n_timepoints2);
                
                % Check minimum neuron threshold
                if n_neurons1 < min_neurons_threshold || n_neurons2 < min_neurons_threshold
                    fprintf('      Insufficient neurons for reliable GLM (threshold: %d), skipping\n', ...
                            min_neurons_threshold);
                    continue;
                end
                
                % Reshape trial data for GLM: concatenate across trials
                % Target format: (n_trials * n_timepoints) × n_neurons
                neural_data1 = reshape(permute(region1_trial_data, [2, 3, 1]), n_neurons1, [])';
                neural_data2 = reshape(permute(region2_trial_data, [2, 3, 1]), n_neurons2, [])';
                
                % Verify latent variable dimensions match neural data
                n_samples_expected = size(neural_data1, 1);
                if length(latent1) ~= n_samples_expected
                    fprintf('      Dimension mismatch: latent has %d samples, neural data has %d\n', ...
                            length(latent1), n_samples_expected);
                    fprintf('      Attempting to reshape latent variables...\n');
                    
                    latent1_reshaped = reshape(latent1, [], 1);
                    latent2_reshaped = reshape(latent2, [], 1);
                    
                    if length(latent1_reshaped) == n_samples_expected
                        latent1 = latent1_reshaped;
                        latent2 = latent2_reshaped;
                        fprintf('      Successfully reshaped latent variables\n');
                    else
                        fprintf('      Cannot resolve dimension mismatch, skipping\n');
                        continue;
                    end
                end
                
                % Perform GLM fitting for both regions
                fprintf('      Fitting GLM for region 1 (%s)...\n', region1_name);
                [beta1, stats1] = fit_glm_for_region(neural_data1, latent1);
                
                fprintf('      Fitting GLM for region 2 (%s)...\n', region2_name);
                [beta2, stats2] = fit_glm_for_region(neural_data2, latent2);
                
                % Store results using anatomically ordered pair key
                pair_key = ordered_pair_name;
                
                if ~isfield(all_glm_results, pair_key)
                    % Determine which region is first in anatomical order
                    ordered_regions = strsplit(ordered_pair_name, '_');
                    all_glm_results.(pair_key) = struct();
                    all_glm_results.(pair_key).region1 = ordered_regions{1};
                    all_glm_results.(pair_key).region2 = ordered_regions{2};
                    all_glm_results.(pair_key).sessions = {};
                    all_glm_results.region_pairs{end+1} = pair_key;
                end
                
                % Determine if we need to swap regions to match anatomical order
                ordered_regions = strsplit(ordered_pair_name, '_');
                needs_swap = ~strcmp(region1_name, ordered_regions{1});
                
                % Create session result structure
                session_result = struct();
                session_result.session_name = session_name;
                
                if needs_swap
                    % Swap coefficients and stats to match anatomical order
                    session_result.region1_coefficients = beta2;
                    session_result.region2_coefficients = beta1;
                    session_result.region1_stats = stats2;
                    session_result.region2_stats = stats1;
                    session_result.selected_neurons1 = selected_neurons2;
                    session_result.selected_neurons2 = selected_neurons1;
                else
                    session_result.region1_coefficients = beta1;
                    session_result.region2_coefficients = beta2;
                    session_result.region1_stats = stats1;
                    session_result.region2_stats = stats2;
                    session_result.selected_neurons1 = selected_neurons1;
                    session_result.selected_neurons2 = selected_neurons2;
                end
                
                session_result.n_trials = n_trials1;
                session_result.n_timepoints = n_timepoints1;
                session_result.cca_R2 = pair_result.cv_results.mean_cv_R2(component_to_analyze);
                session_result.original_pair_name = region_pair_name;
                
                all_glm_results.(pair_key).sessions{end+1} = session_result;
                
                fprintf('      GLM fitting completed. R² - Region1: %.3f, Region2: %.3f\n', ...
                        session_result.region1_stats.r_squared, session_result.region2_stats.r_squared);
            end
            
        catch ME
            fprintf('  Error processing session %s: %s\n', session_name, ME.message);
            fprintf('  Stack trace:\n');
            for k = 1:length(ME.stack)
                fprintf('    %s (line %d)\n', ME.stack(k).name, ME.stack(k).line);
            end
            continue;
        end
    end
    
    %% =====================================================================
    %  SAVE GLM RESULTS
    %  =====================================================================
    
    fprintf('\nSaving GLM analysis results...\n');
    results_file = fullfile(output_dir, 'all_glm_coefficients.mat');
    save(results_file, 'all_glm_results', '-v7.3');
    fprintf('Results saved to: %s\n', results_file);
    
    %% =====================================================================
    %  GENERATE SUMMARY REPORT
    %  =====================================================================
    
    generate_glm_summary_report(all_glm_results, output_dir);
    
    fprintf('\n=== GLM Analysis Complete ===\n');
    fprintf('Total region pairs analyzed: %d\n', length(all_glm_results.region_pairs));
    fprintf('Analyzed pairs: %s\n', strjoin(all_glm_results.region_pairs, ', '));
end


%% =========================================================================
%  HELPER FUNCTIONS
%  =========================================================================

function ordered_pair = reorder_pair_anatomically(pair_name, anatomical_order)
% REORDER_PAIR_ANATOMICALLY - Ensures region pair follows anatomical convention
%
% The region appearing earlier in the anatomical hierarchy should come first.

    regions = strsplit(pair_name, '_');
    region1 = regions{1};
    region2 = regions{2};
    
    idx1 = find(strcmp(anatomical_order, region1), 1);
    idx2 = find(strcmp(anatomical_order, region2), 1);
    
    % Handle regions not in the list
    if isempty(idx1), idx1 = length(anatomical_order) + 1; end
    if isempty(idx2), idx2 = length(anatomical_order) + 1; end
    
    if idx1 <= idx2
        ordered_pair = sprintf('%s_%s', region1, region2);
    else
        ordered_pair = sprintf('%s_%s', region2, region1);
    end
end


function [beta, stats] = fit_glm_for_region(neural_data, latent)
% FIT_GLM_FOR_REGION - Fits GLM to relate neuronal activity to CCA latent variable
%
% MATHEMATICAL FORMULATION:
% We model the relationship:
%   $\mathbf{y} = \mathbf{X}\boldsymbol{\beta} + \boldsymbol{\varepsilon}$
%
% where:
%   $\mathbf{y}$ is the latent variable (response)
%   $\mathbf{X}$ is the neural activity matrix (predictors)
%   $\boldsymbol{\beta}$ are the coefficients to estimate
%   $\boldsymbol{\varepsilon}$ represents residual variance
%
% INTERPRETATION:
% The GLM coefficients $\beta_i$ quantify each neuron's contribution to the
% canonical correlation pattern. Neurons with larger absolute $|\beta_i|$ values
% contribute more strongly to the shared variance captured by CCA.
%
% Inputs:
%   neural_data - Neural activity matrix (n_samples × n_neurons)
%   latent      - CCA latent variable (n_samples × 1)
%
% Outputs:
%   beta  - GLM coefficients for each neuron (n_neurons × 1)
%   stats - Comprehensive fitting statistics

    [n_samples, n_neurons] = size(neural_data);
    
    % Use MATLAB's glmfit for robust parameter estimation
    % 'normal' distribution with 'identity' link for continuous latent variables
    [beta_with_intercept, dev, glm_stats] = glmfit(neural_data, latent, 'normal', ...
        'link', 'identity', 'constant', 'on');
    
    % Separate intercept from neuron-specific coefficients
    intercept = beta_with_intercept(1);
    beta = beta_with_intercept(2:end);
    
    % Calculate predicted values and residuals
    y_pred = glmval(beta_with_intercept, neural_data, 'identity');
    residuals = latent - y_pred;
    
    % Calculate goodness-of-fit metrics
    % $R^2 = 1 - \frac{SS_{residual}}{SS_{total}}$
    ss_total = sum((latent - mean(latent)).^2);
    ss_residual = sum(residuals.^2);
    r_squared = 1 - (ss_residual / ss_total);
    
    % Degrees of freedom analysis
    n_params = length(beta_with_intercept);
    dof = n_samples - n_params;
    
    % Compile comprehensive statistics
    stats = struct();
    stats.r_squared = r_squared;
    stats.adjusted_r_squared = 1 - (1 - r_squared) * (n_samples - 1) / dof;
    stats.intercept = intercept;
    stats.standard_errors = glm_stats.se(2:end);
    stats.t_statistics = glm_stats.t(2:end);
    stats.p_values = glm_stats.p(2:end);
    stats.mse = glm_stats.s^2;
    stats.dof = glm_stats.dfe;
    stats.deviance = dev;
    stats.coeff_covariance = glm_stats.covb(2:end, 2:end);
    
    % Normalized coefficients for cross-neuron comparison
    % $\tilde{\beta}_i = \beta_i / \sigma_\beta$
    beta_normalized = beta / std(beta);
    stats.beta_normalized = beta_normalized;
    
    % Identify statistically significant contributors (α = 0.05)
    stats.significant_neurons = find(stats.p_values < 0.05);
    stats.n_significant = length(stats.significant_neurons);
    
    % Calculate 95% confidence intervals
    % $\text{CI} = \beta \pm t_{\alpha/2, df} \cdot SE(\beta)$
    t_critical = tinv(0.975, stats.dof);
    stats.ci_lower = beta - t_critical * stats.standard_errors;
    stats.ci_upper = beta + t_critical * stats.standard_errors;
    
    % Model selection criteria
    % AIC = $n \log(RSS/n) + 2k$, BIC = $n \log(RSS/n) + k \log(n)$
    stats.aic = n_samples * log(ss_residual/n_samples) + 2 * n_params;
    stats.bic = n_samples * log(ss_residual/n_samples) + log(n_samples) * n_params;
end


function generate_glm_summary_report(all_results, output_dir)
% GENERATE_GLM_SUMMARY_REPORT - Creates comprehensive analysis summary

    report_file = fullfile(output_dir, 'glm_analysis_summary.txt');
    fid = fopen(report_file, 'w');
    
    fprintf(fid, 'Oxford Dataset: GLM Analysis Summary Report\n');
    fprintf(fid, '==========================================\n\n');
    fprintf(fid, 'Analysis Date: %s\n', datestr(now));
    fprintf(fid, 'Data Source: %s\n', all_results.data_source);
    fprintf(fid, 'CCA Component Analyzed: %d\n\n', all_results.component_analyzed);
    
    region_pairs = all_results.region_pairs;
    fprintf(fid, 'Total region pairs analyzed: %d\n', length(region_pairs));
    fprintf(fid, 'Target pairs: %s\n\n', strjoin(all_results.target_pairs, ', '));
    
    % Aggregate statistics across all pairs
    all_r2_region1 = [];
    all_r2_region2 = [];
    all_cca_r2 = [];
    
    % Process each region pair
    for i = 1:length(region_pairs)
        pair_key = region_pairs{i};
        pair_data = all_results.(pair_key);
        
        fprintf(fid, 'Region Pair: %s vs %s\n', pair_data.region1, pair_data.region2);
        fprintf(fid, '  Sessions analyzed: %d\n', length(pair_data.sessions));
        
        if isempty(pair_data.sessions)
            fprintf(fid, '  No valid sessions for this pair\n\n');
            continue;
        end
        
        % Extract statistics across sessions
        r2_values_region1 = cellfun(@(x) x.region1_stats.r_squared, pair_data.sessions);
        r2_values_region2 = cellfun(@(x) x.region2_stats.r_squared, pair_data.sessions);
        cca_r2_values = cellfun(@(x) x.cca_R2, pair_data.sessions);
        n_sig_region1 = cellfun(@(x) x.region1_stats.n_significant, pair_data.sessions);
        n_sig_region2 = cellfun(@(x) x.region2_stats.n_significant, pair_data.sessions);
        
        % Accumulate for overall summary
        all_r2_region1 = [all_r2_region1, r2_values_region1];
        all_r2_region2 = [all_r2_region2, r2_values_region2];
        all_cca_r2 = [all_cca_r2, cca_r2_values];
        
        % Write pair-specific statistics
        fprintf(fid, '  GLM R² - %s: %.3f ± %.3f (range: %.3f-%.3f)\n', ...
                pair_data.region1, mean(r2_values_region1), std(r2_values_region1), ...
                min(r2_values_region1), max(r2_values_region1));
        fprintf(fid, '  GLM R² - %s: %.3f ± %.3f (range: %.3f-%.3f)\n', ...
                pair_data.region2, mean(r2_values_region2), std(r2_values_region2), ...
                min(r2_values_region2), max(r2_values_region2));
        fprintf(fid, '  CCA R² (Component %d): %.3f ± %.3f\n', ...
                all_results.component_analyzed, mean(cca_r2_values), std(cca_r2_values));
        fprintf(fid, '  Significant neurons - %s: %.1f ± %.1f\n', ...
                pair_data.region1, mean(n_sig_region1), std(n_sig_region1));
        fprintf(fid, '  Significant neurons - %s: %.1f ± %.1f\n', ...
                pair_data.region2, mean(n_sig_region2), std(n_sig_region2));
        fprintf(fid, '\n');
    end
    
    % Overall summary statistics
    fprintf(fid, 'OVERALL SUMMARY ACROSS ALL REGION PAIRS\n');
    fprintf(fid, '=======================================\n');
    if ~isempty(all_r2_region1)
        fprintf(fid, 'Average GLM R²: %.3f ± %.3f\n', ...
                mean([all_r2_region1, all_r2_region2]), std([all_r2_region1, all_r2_region2]));
        fprintf(fid, 'Average CCA R²: %.3f ± %.3f\n', mean(all_cca_r2), std(all_cca_r2));
    end
    fprintf(fid, '\nInterpretation:\n');
    fprintf(fid, '- GLM R²: Proportion of CCA latent variance explained by neural activity\n');
    fprintf(fid, '- CCA R²: Canonical correlation strength between brain regions\n');
    fprintf(fid, '- Higher GLM R² indicates better reconstruction of CCA patterns from neurons\n');
    
    fclose(fid);
    fprintf('Summary report saved to: %s\n', report_file);
end


%% =========================================================================
%  SCRIPT EXECUTION
%  =========================================================================
% Uncomment the line below to run the analysis
% oxford_glm_cca_coefficients_extraction()