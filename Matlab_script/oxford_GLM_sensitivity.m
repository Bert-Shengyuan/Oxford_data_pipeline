function oxford_glm_sensitivity_analysis()
% OXFORD_GLM_SENSITIVITY_ANALYSIS
% Performs sensitivity analysis on neuronal contributions for Oxford dataset
%
% THEORETICAL MOTIVATION:
% This analysis addresses a fundamental question in systems neuroscience:
% Is neural information distributed across the population, or concentrated
% in a few highly-informative neurons? We quantify this through systematic
% neuron removal experiments.
%
% METHODOLOGY:
% We employ two complementary removal strategies:
% 1. Top-ranked removal: Remove neurons with highest GLM coefficients
%    (tests concentration hypothesis)
% 2. Random removal: Remove neurons randomly
%    (establishes baseline degradation)
%
% The divergence between these strategies reveals the distribution of
% neural information. Large divergence indicates concentrated coding,
% whereas small divergence suggests distributed representation.
%
% INTERPRETATION:
% Define $\Delta R^2(\rho)$ as the drop in explained variance when removing
% proportion $\rho$ of neurons. If $\Delta R^2_{\text{toprank}}(\rho) \gg \Delta R^2_{\text{random}}(\rho)$,
% this indicates sparse, hierarchical coding. Conversely, similar degradation
% patterns suggest distributed coding across the population.
    clc;clear all;
    fprintf('=== Oxford Dataset: GLM Sensitivity Analysis ===\n\n');
    
    %% Configuration - Adapt to your Oxford dataset structure
    base_results_dir = '/Users/shengyuancai/Downloads/Oxford_dataset/';
    component_to_analyze = 1;
    glm_results_dir = fullfile(base_results_dir, sprintf('GLM_Analysis_Component_%d', component_to_analyze));
    cca_results_dir = fullfile(base_results_dir, 'session_CCA_results');
    psth_results_dir = fullfile(base_results_dir, 'session_PSTH_results');
    
    % Create output directory for sensitivity analysis
    output_dir = fullfile(glm_results_dir, 'sensitivity_analysis');
    if ~exist(output_dir, 'dir')
        mkdir(output_dir);
    end
    
    % Analysis parameters
    removal_percentages = 0:5:90;  % Neuron removal schedule: 0%, 5%, ..., 90%
    n_random_iterations = 10;       % Monte Carlo samples for random removal
    
    fprintf('Configuration:\n');
    fprintf('  GLM results directory: %s\n', glm_results_dir);
    fprintf('  Component analyzed: %d\n', component_to_analyze);
    fprintf('  Removal percentages: %s\n', mat2str(removal_percentages));
    fprintf('  Random iterations: %d\n', n_random_iterations);
    fprintf('  Output directory: %s\n\n', output_dir);
    
    %% Load GLM coefficients
    fprintf('Loading GLM results...\n');
    glm_file = fullfile(glm_results_dir, 'all_glm_coefficients.mat');
    if ~exist(glm_file, 'file')
        error('GLM results not found. Please run oxford_GLM_explain.m first.');
    end
    glm_data = load(glm_file);
    all_glm_results = glm_data.all_glm_results;
    
    %% Process each region pair
    region_pairs = all_glm_results.region_pairs;
    fprintf('Found %d region pairs with GLM results\n\n', length(region_pairs));
    region_pairs = {'MOs_ORB','MOs_STR','MOp_MOs','MOs_mPFC','LP_MOs','MD_MOs'};
    %region_pairs = {'MOp_OLF'};
    for pair_idx = 1:length(region_pairs)
        pair_key = region_pairs{pair_idx};
        pair_data = all_glm_results.(pair_key);
        
        fprintf('Processing pair %d/%d: %s vs %s\n', pair_idx, length(region_pairs), ...
                pair_data.region1, pair_data.region2);
        
        % Initialize sensitivity results for this pair
        sensitivity_results = struct();
        sensitivity_results.region1 = pair_data.region1;
        sensitivity_results.region2 = pair_data.region2;
        sensitivity_results.component = component_to_analyze;
        sensitivity_results.removal_percentages = removal_percentages;
        sensitivity_results.n_random_iterations = n_random_iterations;
        sensitivity_results.sessions = {};
        
        % Process each session for this region pair
        n_sessions = length(pair_data.sessions);
        fprintf('  Analyzing %d sessions...\n', n_sessions);
        
        for session_idx = 1:length(pair_data.sessions)
            session_data = pair_data.sessions{session_idx};
            session_name = session_data.session_name;
            
            fprintf('    Session %d/%d: %s\n', session_idx, n_sessions, session_name);
            
            try
                % Load the original neural data and latent variables
                % We need these to recompute GLM with different neuron subsets
                
                % Load CCA results
                cca_file = fullfile(cca_results_dir, sprintf('%s_CCA_results.mat', session_name));
                cca_data = load(cca_file);
                cca_results = cca_data.cca_results;
                
                % Find this specific region pair in the CCA results
                pair_name = sprintf('%s_%s', pair_data.region1, pair_data.region2);
                pair_result_idx = find(strcmp(cca_results.region_pairs, pair_name), 1);
                
                if isempty(pair_result_idx)
                    fprintf('      Warning: Pair not found in CCA results, skipping\n');
                    continue;
                end
                
                pair_result = cca_results.pair_results{pair_result_idx};
                
                % Extract canonical projections
                projections_results = pair_result.projections;
                comp_projections = projections_results.components{component_to_analyze};

                latent1 = reshape(comp_projections.region_i_trials',[],1);  % n_samples × 1
                latent2 = reshape(comp_projections.region_j_trials',[],1);
                
                % Load PSTH data
                psth_file = fullfile(psth_results_dir, sprintf('%s_PSTH_data.mat', session_name));
                psth_data_struct = load(psth_file);
                psth_data = psth_data_struct.psth_data;
                
                % Extract neural data for selected neurons
                region1_trial_data = psth_data.regions.(pair_data.region1).trial_data;
                region2_trial_data = psth_data.regions.(pair_data.region2).trial_data;
                
                selected_neurons1 = session_data.selected_neurons1;
                selected_neurons2 = session_data.selected_neurons2;
                
                region1_trial_data = region1_trial_data(:, selected_neurons1, :);
                region2_trial_data = region2_trial_data(:, selected_neurons2, :);
                
                % Reshape and mean-center
                [n_trials1, n_neurons1, n_timepoints1] = size(region1_trial_data);
                [n_trials2, n_neurons2, n_timepoints2] = size(region2_trial_data);
                
                neural_data1 = reshape(permute(region1_trial_data, [2,3,1]), n_neurons1,[] )';
                neural_data2 = reshape(permute(region2_trial_data, [2,3,1]), n_neurons2,[] )';
                
                % neural_data1 = neural_data1 - mean(neural_data1);
                % neural_data2 = neural_data2 - mean(neural_data2);
                
                
                % Extract GLM coefficients
                beta1 = session_data.region1_coefficients;
                beta2 = session_data.region2_coefficients;
                
                % Perform sensitivity analysis for both regions
                fprintf('      Analyzing Region 1 sensitivity...\n');
                % var_explained1_toprank= ...
                %     analyze_region_sensitivity(neural_data1, latent1, beta1, ...
                %     removal_percentages, n_random_iterations);

                var_explained1_toprank= ...
                    analyze_region_sensitivity(neural_data1, latent1, beta1, ...
                    removal_percentages);
                fprintf('      Analyzing Region 2 sensitivity...\n');
                var_explained2_toprank = ...
                    analyze_region_sensitivity(neural_data2, latent2, beta2, ...
                    removal_percentages);
                
                % Store session results
                session_result = struct();
                session_result.session_name = session_name;
                session_result.region1_toprank = var_explained1_toprank;
                % session_result.region1_random = var_explained1_random;
                session_result.region2_toprank = var_explained2_toprank;
                % session_result.region2_random = var_explained2_random;
                session_result.original_r2_region1 = session_data.region1_stats.r_squared;
                session_result.original_r2_region2 = session_data.region2_stats.r_squared;
                session_result.cca_R2 = session_data.cca_R2;
                
                sensitivity_results.sessions{end+1} = session_result;
                
                fprintf('      Sensitivity analysis completed\n');
                
            catch ME
                fprintf('      Error: %s\n', ME.message);
                continue;
            end
        end
        
        % Save sensitivity results for this pair
        if ~isempty(sensitivity_results.sessions)
            save_file = fullfile(output_dir, sprintf('sensitivity_%s.mat', pair_key));
            save(save_file, 'sensitivity_results', '-v7.3');
            fprintf('  Saved sensitivity results to: %s\n', save_file);
            
            % Create visualization for this pair
            create_sensitivity_visualization(sensitivity_results, output_dir);
        else
            fprintf('  No valid sessions for sensitivity analysis\n');
        end
        
        fprintf('\n');
    end
    
    % Create cross-pair summary
    %create_cross_pair_summary(output_dir);
    
    fprintf('=== Sensitivity Analysis Complete ===\n');
    fprintf('Results saved to: %s\n', output_dir);
end


% function [var_explained_toprank, var_explained_random] = ...
%     analyze_region_sensitivity(neural_data, latent, beta_coeffs, ...
%     removal_percentages, n_random_iterations)
function var_explained_toprank = ...
    analyze_region_sensitivity(neural_data, latent, beta_coeffs, ...
    removal_percentages)
% ANALYZE_REGION_SENSITIVITY
% Systematically evaluates information distribution through neuron removal
%
% COMPUTATIONAL APPROACH:
% For each removal percentage $\rho \in \{0\%, 5\%, \ldots, 90\%\}$:
% 1. Top-ranked strategy: Remove the $\lfloor \rho \cdot N \rfloor$ neurons
%    with largest $|\beta_i|$ values
% 2. Random strategy: Perform Monte Carlo sampling, removing random subsets
% 3. Refit GLM on remaining neurons
% 4. Compute $R^2$ to quantify reconstruction quality
%
% MATHEMATICAL NOTE:
% The GLM refitting ensures that remaining neurons can "compensate" for
% removed neurons through adjusted coefficients. This tests whether
% information truly resides in specific neurons versus being redundantly
% encoded across the population.

    [~, n_neurons] = size(neural_data);
    
    % Rank neurons by absolute coefficient magnitude
    [~, sorted_indices] = sort(abs(beta_coeffs), 'descend');
    
    % Initialize storage
    var_explained_toprank = zeros(size(removal_percentages));
    %var_explained_random = zeros(length(removal_percentages), n_random_iterations);
    
    % Baseline: compute R² with all neurons
    baseline_r2 = compute_glm_r2(neural_data, latent);
    
    for pct_idx = 1:length(removal_percentages)
        pct = removal_percentages(pct_idx);
        n_remove = round(n_neurons * pct / 100);
        n_keep = n_neurons - n_remove;
        
        if n_keep < 2  % Need at least 2 neurons for GLM
            var_explained_toprank(pct_idx) = NaN;
            var_explained_random(pct_idx, :) = NaN;
            continue;
        end
        
        % Top-ranked removal strategy
        % Remove neurons with highest |β| values
        neurons_keep_toprank = sorted_indices((n_remove+1):end);
        neural_data_toprank = neural_data(:, neurons_keep_toprank);
        var_explained_toprank(pct_idx) = compute_glm_r2(neural_data_toprank, latent);
        
        % % Random removal strategy (Monte Carlo approach)
        % for iter = 1:n_random_iterations
        %     % Randomly select neurons to keep
        %     all_neurons = 1:n_neurons;
        %     neurons_keep_random = datasample(all_neurons, n_keep, 'Replace', false);
        %     neural_data_random = neural_data(:, neurons_keep_random);
        %     var_explained_random(pct_idx, iter) = compute_glm_r2(neural_data_random, latent);
        % end
    end
end


function r2 = compute_glm_r2(X, y)
% COMPUTE_GLM_R2 - Calculate R² for GLM prediction
% 
% Efficiently computes goodness-of-fit without storing full statistics

    try
        % Fit GLM
        [beta_with_intercept, ~, ~] = glmfit(X, y, 'normal', ...
            'link', 'identity', 'constant', 'on');
        
        % Predict
        y_pred = glmval(beta_with_intercept, X, 'identity');
        
        % Calculate R²
        ss_total = sum((y - mean(y)).^2);
        ss_residual = sum((y - y_pred).^2);
        r2 = 1 - (ss_residual / ss_total);
        
        % Handle edge cases
        if isnan(r2) || isinf(r2) || r2 < 0
            r2 = 0;
        end
    catch
        r2 = 0;
    end
end


% function create_sensitivity_visualization(sensitivity_results, output_dir)
function create_sensitivity_visualization(sensitivity_results, output_dir)
    % CREATE_TOPRANK_SESSIONS_PLOT Creates visualization showing top-ranked neuron 
    % removal sensitivity for all individual sessions of two regions
    %
    % This function generates a comprehensive plot showing how explained variance
    % changes as top-ranked neurons are systematically removed, with individual
    % lines for each recording session and both brain regions.
    %
    % The visualization reveals:
    % 1. Session-to-session variability in coding strategies
    % 2. Relative importance of top-ranked neurons in each region
    % 3. Identification of outlier sessions with unusual sensitivities
    % 4. Overall robustness of cross-region communication patterns

    % Check if there are sessions to analyze
    n_sessions = length(sensitivity_results.sessions);
    if n_sessions == 0
        warning('No sessions found in sensitivity results');
        return;
    end

    % Get removal percentages
    removal_pct = sensitivity_results.removal_percentages;

    % Create figure
    figure('Position', [100, 100, 800, 900]);
    hold on;

    % Define colors for the two regions (using distinguishable color schemes)

    % Initialize arrays for legend entries
    region1_handles = zeros(n_sessions, 1);
    region2_handles = zeros(n_sessions, 1);
    if contains(sensitivity_results.region1, 'MOs')
        region1_colors = [0.7, 0.7, 1]; % Blue for ALM
        region2_colors = [0.8, 0.5, 0.2]; % Red for other
        color_toprank_region1 = [0, 0, 0.8];
        color_toprank_region2 = [0.8, 0.5, 0.2];
    elseif contains(sensitivity_results.region2, 'MOs')
        % This shouldn't happen if ALM prioritization is working correctly
        region1_colors = [0.8, 0.5, 0.2]; % Red
        region2_colors = [0.7, 0.7, 1]; % Blue ALM
        color_toprank_region1 = [0.8, 0.5, 0.2];
        color_toprank_region2 = [0, 0, 0.8];
    else
        % Neither is ALM
        region1_colors = [1, 0.7, 0.7]; % Red
        region2_colors = [0.7, 0.7, 1]; % Blue ALM
        color_toprank_region1 = [0.8, 0, 0];
        color_toprank_region2 = [0, 0, 0.8];
    end
    % Plot individual sessions for region 1
    for s = 1:n_sessions
        session = sensitivity_results.sessions{s};
        h = plot(removal_pct, session.region1_toprank, '-o', ...
            'Color', region1_colors, 'LineWidth', 1.2, ...
            'MarkerSize', 4, 'MarkerFaceColor', region1_colors);
        region1_handles(s) = h;
    end

    % Plot individual sessions for region 2
    for s = 1:n_sessions
        session = sensitivity_results.sessions{s};
        h = plot(removal_pct, session.region2_toprank, '-s', ...
            'Color', region2_colors, 'LineWidth', 1.2, ...
            'MarkerSize', 4, 'MarkerFaceColor', region2_colors);
        region2_handles(s) = h;
    end

    % Calculate mean performance across sessions for both regions
    region1_means = zeros(length(removal_pct), 1);
    region2_means = zeros(length(removal_pct), 1);
    region1_std = zeros(length(removal_pct), 1);
    region2_std = zeros(length(removal_pct), 1);
    region1_se = zeros(length(removal_pct), 1);
    region2_se = zeros(length(removal_pct), 1);

    for i = 1:length(removal_pct)
        region1_values = zeros(n_sessions, 1);
        region2_values = zeros(n_sessions, 1);

        for s = 1:n_sessions
            session = sensitivity_results.sessions{s};
            region1_values(s) = session.region1_toprank(i);
            region2_values(s) = session.region2_toprank(i);
        end

        region1_means(i) = mean(region1_values);
        region2_means(i) = mean(region2_values);
        region1_std(i) = std(region1_values);
        region2_std(i) = std(region1_values);
        region1_se(i) = std(region1_values)/sqrt(n_sessions);
        region2_se(i) = std(region2_values)/sqrt(n_sessions);
    end

    % Plot mean lines with thicker stroke and distinct colors
    % h1_mean = plot(removal_pct, region1_means, '-o', 'Color', [0, 0, 0.8], ...
    %     'LineWidth', 3.5, 'MarkerSize', 9, 'MarkerFaceColor', [0, 0, 0.8]);
    % 
    % h2_mean = plot(removal_pct, region2_means, '-s', 'Color', [0.8, 0, 0], ...
    %     'LineWidth', 3.5, 'MarkerSize', 9, 'MarkerFaceColor', [0.8, 0, 0]);



    h1_mean = errorbar(removal_pct, region1_means, region1_se, ...
        'Color', color_toprank_region1, 'LineWidth', 3, 'Marker', 'o', ...
        'MarkerFaceColor', color_toprank_region1, 'MarkerSize', 6);

    h2_mean = errorbar(removal_pct, region2_means, region2_se, ...
        'Color', color_toprank_region2, 'LineWidth', 3, 'Marker', 's', ...
        'MarkerFaceColor', color_toprank_region2, 'MarkerSize', 6);

    % Formatting
    xlabel('Percentage of Top-Ranked Neurons Removed (%)', 'FontSize', 30, 'FontWeight', 'normal');
    ylabel('Explained Variance (R²)', 'FontSize', 30, 'FontWeight', 'normal');
    % title(sprintf('Encoding Sensitivity: %s vs %s\n(All %d Sessions)', ...
    %     strrep(sensitivity_results.region1, '_', ' '), ...
    %     strrep(sensitivity_results.region2, '_', ' '), n_sessions), ...
    %     'FontSize', 30,"FontWeight","normal");
    % title(sprintf('(All %d Sessions)', n_sessions), ...
    %     'FontSize', 30,"FontWeight","normal");
    % Legend for mean lines only (to avoid cluttering)
    legend([h1_mean, h2_mean], ...
        {sprintf('%s (Mean)', strrep(sensitivity_results.region1, '_', ' ')), ...
         sprintf('%s (Mean)', strrep(sensitivity_results.region2, '_', ' '))}, ...
        'Location', 'southwest', 'FontSize', 25);

    % Grid and limits
    grid on;
    xlim([-5, 95]);
    ylim([0, 1.05]);
    set(gca, 'FontSize', 28);

    % Add interpretive annotations
    % Calculate session variability at key removal points
    idx_50 = find(removal_pct >= 50, 1, 'first');
    if ~isempty(idx_50)
        % Calculate mean drop at 50% removal
        region1_drop = region1_means(1) - region1_means(idx_50);
        region2_drop = region2_means(1) - region2_means(idx_50);

        % Calculate variability metrics
        region1_var = mean(region1_std);
        region2_var = mean(region2_std);

        % Determine which region has more concentrated coding
        more_concentrated = '';
        if region1_drop > region2_drop * 1.2
            more_concentrated = sprintf('%s has more concentrated coding', ...
                strrep(sensitivity_results.region1, '_', ' '));
        elseif region2_drop > region1_drop * 1.2
            more_concentrated = sprintf('%s has more concentrated coding', ...
                strrep(sensitivity_results.region2, '_', ' '));
        else
            more_concentrated = 'Both regions have similar coding concentration';
        end

        % Add text annotation with analysis
        analysis_text = sprintf(['Session Analysis:\n\n', ...
            '%s:\n  Mean drop at 50%%: %.2f\n  Session variability: %.2f\n\n', ...
            '%s:\n  Mean drop at 50%%: %.2f\n  Session variability: %.2f\n\n', ...
            '%s'], ...
            strrep(sensitivity_results.region1, '_', ' '), region1_drop, region1_var, ...
            strrep(sensitivity_results.region2, '_', ' '), region2_drop, region2_var, ...
            more_concentrated);

        % annotation('textbox', [0.69, 0.67, 0.35, 0.25], 'String', analysis_text, ...
        %     'FontSize', 11, 'BackgroundColor', [1 1 1 1], 'EdgeColor', [1,1,1], ...
        %     'FitBoxToText', 'on');
    end

    % Save figure in multiple formats
    pair_name = sprintf('%s_%s', sensitivity_results.region1, sensitivity_results.region2);
    saveas(gcf, fullfile(output_dir, sprintf('toprank_all_sessions_%s.png', pair_name)));
    %saveas(gcf, fullfile(output_dir, sprintf('toprank_all_sessions_%s.fig', pair_name)));
    %saveas(gcf, fullfile(output_dir, sprintf('toprank_all_sessions_%s.svg', pair_name)));
    close(gcf);

    fprintf('Created visualization of top-ranked neuron sensitivity for all sessions: %s vs %s\n', ...
        strrep(sensitivity_results.region1, '_', ' '), ...
        strrep(sensitivity_results.region2, '_', ' '));
end


function create_cross_pair_summary(output_dir)
% CREATE_CROSS_PAIR_SUMMARY
% Generates comparative analysis across all region pairs
%
% QUANTITATIVE METRICS:
% 1. Concentration Index: $\Delta R^2_{\text{top}}(50\%) - \Delta R^2_{\text{random}}(50\%)$
%    Measures information concentration at 50% removal
% 2. Overall Sensitivity: Total variance loss from 0% to 50% removal
% 3. Distributed vs Sparse Classification based on empirical thresholds

    fprintf('Creating cross-pair summary analysis...\n');
    
    % Load all sensitivity results
    sens_files = dir(fullfile(output_dir, 'sensitivity_*.mat'));
    
    if isempty(sens_files)
        fprintf('No sensitivity results found for summary\n');
        return;
    end
    
    % Initialize storage
    pair_names = {};
    concentration_index_r1 = [];
    concentration_index_r2 = [];
    
    for f = 1:length(sens_files)
        data = load(fullfile(output_dir, sens_files(f).name));
        sens = data.sensitivity_results;
        
        if isempty(sens.sessions)
            continue;
        end
        
        pair_names{end+1} = sprintf('%s-%s', sens.region1, sens.region2);
        
        % Calculate concentration index at 50% removal
        pct_50_idx = find(sens.removal_percentages == 50);
        if ~isempty(pct_50_idx)
            % Aggregate across sessions
            n_sess = length(sens.sessions);
            toprank_r1_50 = zeros(n_sess, 1);
            random_r1_50 = zeros(n_sess, 1);
            toprank_r2_50 = zeros(n_sess, 1);
            random_r2_50 = zeros(n_sess, 1);
            
            for s = 1:n_sess
                toprank_r1_50(s) = sens.sessions{s}.region1_toprank(pct_50_idx);
                random_r1_50(s) = mean(sens.sessions{s}.region1_random(pct_50_idx, :));
                toprank_r2_50(s) = sens.sessions{s}.region2_toprank(pct_50_idx);
                random_r2_50(s) = mean(sens.sessions{s}.region2_random(pct_50_idx, :));
            end
            
            % Concentration index = drop in toprank - drop in random
            % Higher values indicate more concentrated coding
            concentration_index_r1(end+1) = mean(random_r1_50 - toprank_r1_50);
            concentration_index_r2(end+1) = mean(random_r2_50 - toprank_r2_50);
        end
    end
    
    % Create summary bar plot
    figure('Position', [100, 100, 1200, 600]);
    
    x = 1:length(pair_names);
    bar_width = 0.35;
    
    bar(x - bar_width/2, concentration_index_r1, bar_width, 'FaceColor', [0.2, 0.4, 0.8]);
    hold on;
    bar(x + bar_width/2, concentration_index_r2, bar_width, 'FaceColor', [0.8, 0.4, 0.2]);
    
    set(gca, 'XTick', x, 'XTickLabel', pair_names, 'XTickLabelRotation', 45);
    ylabel('Encoding Concentration Index (\Delta R^2)', 'FontSize', 14);
    xlabel('Region Pairs', 'FontSize', 14);
    title('Neural Information Distribution Across Region Pairs', 'FontSize', 16, 'FontWeight', 'bold');
    legend('Region 1', 'Region 2', 'Location', 'best', 'FontSize', 12);
    grid on;
    set(gca, 'FontSize', 11);
    
    % Add interpretation reference lines
    yline(0.1, 'r--', 'Sparse Coding Threshold', 'LineWidth', 1.5, 'FontSize', 10);
    yline(0.05, 'y--', 'Mixed Coding', 'LineWidth', 1.5, 'FontSize', 10);
    
    % Save figure
    saveas(gcf, fullfile(output_dir, 'cross_pair_concentration_summary.png'));
    close(gcf);
    
    fprintf('Cross-pair summary saved\n');
end

%%
oxford_glm_sensitivity_analysis()