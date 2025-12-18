function oxford_glm_cca_coefficients_extraction()
% OXFORD_GLM_CCA_COEFFICIENTS_EXTRACTION 
% Fits GLM to relate neuronal PSTH to CCA latent variables for Oxford dataset
%
% THEORETICAL FRAMEWORK:
% This function performs General Linear Model fitting to elucidate how individual
% neurons contribute to the canonical correlation patterns. The GLM establishes
% a mathematical relationship: $\mathbf{L}(t) = \sum_i \beta_i \cdot \text{PSTH}_i(t) + \boldsymbol{\varepsilon}$
% where $\mathbf{L}(t)$ represents the CCA latent variable, $\beta_i$ denote the 
% GLM coefficients quantifying each neuron's contribution, and $\text{PSTH}_i(t)$ 
% represents the peri-stimulus time histogram of neuron $i$.
%
% ADAPTATION FOR OXFORD DATASET:
% This implementation is specifically designed to work with the session-based
% storage structure where CCA results and PSTH data are organized by session
% rather than by region pair directories.
%
% DATA STRUCTURE:
% Input:  session_CCA_results/{session}_CCA_results.mat
%         session_PSTH_results/{session}_PSTH_data.mat
% Output: GLM coefficients, statistics, and model diagnostics

    fprintf('=== Oxford Dataset: GLM Analysis of CCA Latent Variables ===\n\n');
    
    %% Configuration - Adapt these paths to your Oxford dataset location
    base_results_dir = '/Users/shengyuancai/Downloads/Oxford_dataset/';
    cca_results_dir = fullfile(base_results_dir, 'session_CCA_results');
    psth_results_dir = fullfile(base_results_dir, 'session_PSTH_results');
    
    % Analysis parameters
    component_to_analyze = 1;  % Which CCA component to analyze (1-indexed)
    min_neurons_threshold = 30; % Minimum neurons required for reliable GLM fitting
    
    % Create output directory for GLM results
    output_dir = fullfile(base_results_dir, sprintf('GLM_Analysis_Component_%d', component_to_analyze));
    if ~exist(output_dir, 'dir')
        mkdir(output_dir);
    end
    
    fprintf('Configuration:\n');
    fprintf('  CCA results directory: %s\n', cca_results_dir);
    fprintf('  PSTH data directory: %s\n', psth_results_dir);
    fprintf('  Analyzing CCA component: %d\n', component_to_analyze);
    fprintf('  Output directory: %s\n\n', output_dir);
    
    %% Discover all available sessions
    cca_files = dir(fullfile(cca_results_dir, '*_CCA_results.mat'));
    fprintf('Found %d sessions with CCA results\n\n', length(cca_files));
    
    % Initialize storage for all GLM results
    all_glm_results = struct();
    all_glm_results.component_analyzed = component_to_analyze;
    all_glm_results.analysis_timestamp = datestr(now);
    all_glm_results.region_pairs = {};
    
    %% Process each session
    for session_idx = 1:length(cca_files)
        % Extract session name from filename
        filename = cca_files(session_idx).name;
        session_name = strrep(filename, '_CCA_results.mat', '');
        
        fprintf('Processing session %d/%d: %s\n', session_idx, length(cca_files), session_name);
        
        try
            % Load CCA results for this session
            cca_file = fullfile(cca_results_dir, filename);
            fprintf('  Loading CCA results...\n');
            cca_data = load(cca_file);
            cca_results = cca_data.cca_results;
            
            % Load PSTH data for this session
            psth_file = fullfile(psth_results_dir, sprintf('%s_PSTH_data.mat', session_name));
            if ~exist(psth_file, 'file')
                fprintf('  Warning: PSTH file not found, skipping session\n');
                continue;
            end
            fprintf('  Loading PSTH data...\n');
            psth_data_struct = load(psth_file);
            psth_data = psth_data_struct.psth_data;
            
            % Process each region pair from this session
            n_pairs = length(cca_results.pair_results);
            fprintf('  Found %d region pairs in session\n', n_pairs);
            
            for pair_idx = 1:n_pairs
                pair_result = cca_results.pair_results{pair_idx};
                region_pair_name = cca_results.region_pairs{pair_idx};
                
                fprintf('    Analyzing pair: %s\n', region_pair_name);
                
                % Parse region names
                regions = strsplit(region_pair_name, '_');
                region1_name = regions{1};
                region2_name = regions{2};
                
                % Check if we have enough significant components
                if length(pair_result.significant_components) < component_to_analyze
                    fprintf('      Insufficient significant components, skipping\n');
                    continue;
                end
                
                % Extract the canonical projections for the specified component
                projections_results = pair_result.projections;
                if ~isfield(projections_results, 'components') || ...
                   projections_results.n_components < component_to_analyze
                    fprintf('      Canonical projections not available, skipping\n');
                    continue;
                end
                
                comp_projections = projections_results.components{component_to_analyze};
                
                % Extract latent variables (canonical variates)
                % These are the "ground truth" patterns we'll try to reconstruct via GLM
                %latent1 = comp_projections.region_i_trials'; 
                latent1 = reshape(comp_projections.region_i_trials',[],1);  % n_samples × 1
                %latent2 = comp_projections.region_j_trials';  % n_samples × 1
                latent2 = reshape(comp_projections.region_j_trials',[],1);

                
                % Get the trial data from PSTH for both regions
                if ~isfield(psth_data.regions, region1_name) || ...
                   ~isfield(psth_data.regions, region2_name)
                    fprintf('      PSTH data not available for both regions, skipping\n');
                    continue;
                end
                
                region1_trial_data = psth_data.regions.(region1_name).trial_data;
                region2_trial_data = psth_data.regions.(region2_name).trial_data;
                
                % Get the neuron indices that were actually used in CCA
                selected_neurons1 = pair_result.selected_neurons_i;
                selected_neurons2 = pair_result.selected_neurons_j;
                
                % Extract only the selected neurons
                region1_trial_data = region1_trial_data(:, selected_neurons1, :);
                region2_trial_data = region2_trial_data(:, selected_neurons2, :);
                
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
                neural_data1 = reshape(permute(region1_trial_data, [2,3,1]), n_neurons1,[] )';
                neural_data2 = reshape(permute(region2_trial_data, [2,3,1]), n_neurons2,[] )';
                
                % %Mean-center the neural data (standard preprocessing for GLM)
                % neural_data1 = neural_data1 - mean(neural_data1,1);
                % neural_data2 = neural_data2 - mean(neural_data2,1);
                
                % Verify latent variable dimensions match neural data
                n_samples_expected = size(neural_data1, 1);
                if length(latent1) ~= n_samples_expected
                    fprintf('      Dimension mismatch: latent has %d samples, neural data has %d\n', ...
                            length(latent1), n_samples_expected);
                    fprintf('      Attempting to reshape latent variables...\n');
                    
                    % Reshape latent to match expected dimensions if needed
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
                
                % Store results for this region pair
                pair_key = sprintf('%s_%s', region1_name, region2_name);
                
                if ~isfield(all_glm_results, pair_key)
                    all_glm_results.(pair_key) = struct();
                    all_glm_results.(pair_key).region1 = region1_name;
                    all_glm_results.(pair_key).region2 = region2_name;
                    all_glm_results.(pair_key).sessions = {};
                    all_glm_results.region_pairs{end+1} = pair_key;
                end
                
                % Create session result structure
                session_result = struct();
                session_result.session_name = session_name;
                session_result.region1_coefficients = beta1;
                session_result.region2_coefficients = beta2;
                session_result.region1_stats = stats1;
                session_result.region2_stats = stats2;
                session_result.selected_neurons1 = selected_neurons1;
                session_result.selected_neurons2 = selected_neurons2;
                session_result.n_trials = n_trials1;
                session_result.n_timepoints = n_timepoints1;
                session_result.cca_R2 = pair_result.cv_results.mean_cv_R2(component_to_analyze);
                
                all_glm_results.(pair_key).sessions{end+1} = session_result;
                
                fprintf('      GLM fitting completed. R² - Region1: %.3f, Region2: %.3f\n', ...
                        stats1.r_squared, stats2.r_squared);
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
    
    %% Save GLM results
    fprintf('\nSaving GLM analysis results...\n');
    results_file = fullfile(output_dir, 'all_glm_coefficients.mat');
    save(results_file, 'all_glm_results', '-v7.3');
    fprintf('Results saved to: %s\n', results_file);
    
    %% Generate summary report
    generate_glm_summary_report(all_glm_results, output_dir);
    
    fprintf('\n=== GLM Analysis Complete ===\n');
    fprintf('Total region pairs analyzed: %d\n', length(all_glm_results.region_pairs));
end

function [beta, stats] = fit_glm_for_region(neural_data, latent)
% FIT_GLM_FOR_REGION - Fits GLM to relate neuronal activity to CCA latent variable
%
% MATHEMATICAL FORMULATION:
% We model the relationship: $\mathbf{y} = \mathbf{X}\boldsymbol{\beta} + \boldsymbol{\varepsilon}$
% where $\mathbf{y}$ is the latent variable (our "response"), 
% $\mathbf{X}$ is the neural activity matrix (our "predictors"),
% $\boldsymbol{\beta}$ are the coefficients we seek to estimate,
% and $\boldsymbol{\varepsilon}$ represents residual variance.
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
    % We use 'normal' distribution with 'identity' link for continuous latent variables
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
%
% This report facilitates understanding of:
% 1. How effectively neural populations explain canonical patterns
% 2. Distribution of significant neuronal contributors
% 3. Cross-region pair comparisons

    report_file = fullfile(output_dir, 'glm_analysis_summary.txt');
    fid = fopen(report_file, 'w');
    
    fprintf(fid, 'Oxford Dataset: GLM Analysis Summary Report\n');
    fprintf(fid, '==========================================\n\n');
    fprintf(fid, 'Analysis Date: %s\n', datestr(now));
    fprintf(fid, 'CCA Component Analyzed: %d\n\n', all_results.component_analyzed);
    
    region_pairs = all_results.region_pairs;
    fprintf(fid, 'Total region pairs analyzed: %d\n\n', length(region_pairs));
    
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
        fprintf(fid, '  Significant neurons - %s: %.1f ± %.1f (%.1f%%)\n', ...
                pair_data.region1, mean(n_sig_region1), std(n_sig_region1), ...
                100 * mean(n_sig_region1) / length(pair_data.sessions{1}.selected_neurons1));
        fprintf(fid, '  Significant neurons - %s: %.1f ± %.1f (%.1f%%)\n', ...
                pair_data.region2, mean(n_sig_region2), std(n_sig_region2), ...
                100 * mean(n_sig_region2) / length(pair_data.sessions{1}.selected_neurons2));
        fprintf(fid, '\n');
    end
    
    % Overall summary statistics
    fprintf(fid, 'OVERALL SUMMARY ACROSS ALL REGION PAIRS\n');
    fprintf(fid, '=======================================\n');
    fprintf(fid, 'Average GLM R²: %.3f ± %.3f\n', ...
            mean([all_r2_region1, all_r2_region2]), std([all_r2_region1, all_r2_region2]));
    fprintf(fid, 'Average CCA R²: %.3f ± %.3f\n', mean(all_cca_r2), std(all_cca_r2));
    fprintf(fid, '\nInterpretation:\n');
    fprintf(fid, '- GLM R²: Proportion of CCA latent variance explained by neural activity\n');
    fprintf(fid, '- CCA R²: Canonical correlation strength between brain regions\n');
    fprintf(fid, '- Higher GLM R² indicates better reconstruction of CCA patterns from neurons\n');
    
    fclose(fid);
    fprintf('Summary report saved to: %s\n', report_file);
end

%%
oxford_glm_cca_coefficients_extraction()