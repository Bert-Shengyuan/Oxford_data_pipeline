function oxford_neuron_stability_analysis()
% OXFORD_NEURON_STABILITY_ANALYSIS - Multi-region neuron count stability assessment
%
    clc;clear;
    fprintf('=== Oxford Dataset Neuron Count Stability Analysis ===\n');
    fprintf('Theoretical Framework: Multi-region canonical correlation stability\n');
    fprintf('Methodological Approach: Reference latent comparison with dynamic region pairs\n\n');
    
    %% Configuration Parameters
    % These parameters define both the computational and neurobiological 
    % constraints of our stability analysis
    
    config = struct();
    
    % Data Source Configuration
    config.oxford_base_dir = '/Users/shengyuancai/Downloads/Oxford_dataset/';
    config.results_dir = fullfile(config.oxford_base_dir, 'session_CCA_results');
    config.output_dir = fullfile(config.oxford_base_dir, 'neuron_stability_analysis');
    
    % Analysis Parameters - Theoretical Foundation
    config.reference_neurons = 50;          % Reference population size for "gold standard"
    config.n_reference_iterations = 10;      % Iterations for reference latent averaging
    config.n_test_iterations =  config.n_reference_iterations;           % Iterations per test neuron count
    %config.neuron_counts = [100]; % Test populations
    %config.neuron_counts = [5, 10, 15, 20,25, 30, 35,40,45 50, 60, 70, 80, 90,100,130,140,160,180,190]; % Test populations
    config.neuron_counts = [5, 10, 15, 20,25, 30, 35,40,45]; % Test populations
    % 
    % Statistical Framework
    config.n_cv_folds = 10;                  % Cross-validation folds for CCA
    config.min_region_neurons = config.reference_neurons;         % Minimum neurons required per region
    config.significance_threshold = 0.05;     % Statistical significance for components
    
    % Neurobiological Constraints
    config.time_window_analysis = [-1.5, 3.0]; % Peri-stimulus window for analysis
    config.max_components = 1;                % Focus on primary canonical dimensions
    
    fprintf('Configuration established:\n');
    fprintf('  Reference standard: %d neurons (%d iterations)\n', ...
        config.reference_neurons, config.n_reference_iterations);
    fprintf('  Test populations: %d-%d neurons\n', ...
        min(config.neuron_counts), max(config.neuron_counts));
    fprintf('  Statistical framework: %d-fold cross-validation\n', config.n_cv_folds);
    fprintf('  Neurobiological window: [%.1f, %.1f]s \n', ...
        config.time_window_analysis(1), config.time_window_analysis(2));
    
    % Create output directory structure
    if ~exist(config.output_dir, 'dir')
        mkdir(config.output_dir);
    end
    
    %% Session Discovery and Selection
    % Identify processed sessions with sufficient neural populations for analysis
    
    fprintf('\n=== Phase 1: Session Discovery and Validation ===\n');
    available_sessions = discover_valid_sessions(config);
    
    if isempty(available_sessions)
        error('No sessions found meeting minimum neuron count criteria');
    end
    
    fprintf('Discovered %d sessions meeting analysis criteria\n', length(available_sessions));
    
    % Select representative session for detailed analysis
    % In practice, you would want to run this analysis across multiple sessions
    selected_session_idx = 12;  % Can be modified to analyze different sessions
    selected_session = available_sessions{selected_session_idx};
    
    fprintf('Selected session for stability analysis: %s\n', selected_session.session_name);
    fprintf('  Available region pairs: %d\n', length(selected_session.region_pairs));
    fprintf('  Maximum population size: %d neurons per region\n', selected_session.max_neurons);
    
    %% Region Pair Comprehensive Analysis
    % Analyze ALL region pairs systematically instead of selecting optimal one
    % This provides complete characterization of inter-regional stability patterns
    
    fprintf('\n=== Phase 2: Comprehensive Region Pair Analysis ===\n');
    
    % Load session data once for efficiency
    fprintf('  Loading session data: %s\n', selected_session.session_name);
    session_data = load(selected_session.file_path);
    
    % Create paired_region subdirectory for organizing results
    paired_region_dir = fullfile(config.oxford_base_dir, 'neuron_stability_analysis', 'paired_region');
    if ~exist(paired_region_dir, 'dir')
        mkdir(paired_region_dir);
        fprintf('  Created paired_region directory: %s\n', paired_region_dir);
    end
    
    % Create session-specific subdirectory
    session_output_dir = fullfile(paired_region_dir, selected_session.session_name);
    if ~exist(session_output_dir, 'dir')
        mkdir(session_output_dir);
        fprintf('  Created session directory: %s\n', session_output_dir);
    end
    
    %%
    n_pairs = length(selected_session.region_pairs);
    fprintf('  Analyzing %d region pairs comprehensively...\n', n_pairs);
    
    % Iterate through all region pairs
    for pair_idx = 1:n_pairs
        current_pair = selected_session.region_pairs{pair_idx};
        
        fprintf('\n  --- Processing Pair %d/%d: %s <-> %s ---\n', ...
            pair_idx, n_pairs, current_pair.region1, current_pair.region2);
        fprintf('      Region 1: %d neurons\n', current_pair.n1);
        fprintf('      Region 2: %d neurons\n', current_pair.n2);
        
        % Augment pair structure with necessary fields
        current_pair.n_neurons1 = current_pair.n1;
        current_pair.n_neurons2 = current_pair.n2;
        
        try
            % Calculate baseline correlation for this pair
            fprintf('      Computing baseline correlation...\n');
            current_pair.baseline_r2 = calculate_baseline_correlation(session_data, current_pair, config);
            fprintf('      Baseline : %.3f\n', current_pair.baseline_r2);
            
            %% Phase 3: Reference Latent Establishment for this pair
            fprintf('      Establishing reference latent...\n');
            reference_latent = establish_reference_latent(session_data, current_pair, config);
            
            fprintf('      Reference characteristics:\n');
            fprintf('        Cross-correlation: %.3f\n', reference_latent.cross_correlation);
            fprintf('        Timepoints captured: %d\n', length(reference_latent.time_vector));
            
            %% Phase 4: Multi-Scale Stability Assessment for this pair
            fprintf('      Assessing multi-scale stability...\n');
            stability_results = assess_multi_scale_stability(session_data, current_pair, ...
                                                           reference_latent, config);
            
            %% Phase 5: Results Compilation and Visualization
            fprintf('Compiling results...\n');
            final_results = compile_stability_results(stability_results, reference_latent, ...
                                                    current_pair, selected_session, config);
            
            %%
            % Save results for this specific pair
            pair_name = sprintf('%s_vs_%s', current_pair.region1, current_pair.region2);
            save_file = fullfile(session_output_dir, sprintf('%s_stability_analysis.mat', pair_name));
            save(save_file, 'final_results', 'config');
            fprintf('      Saved to: %s\n', save_file);
            
            %%
            % Generate visualizations for this pair
            fprintf('      Generating visualizations...\n');
            create_stability_visualizations(final_results, config, session_output_dir);
            
            %fprintf('Pair %d/%d complete\n', pair_idx, n_pairs);
            
        catch ME
            fprintf('Error processing pair: %s\n', ME.message);
            fprintf('Continuing to next pair...\n');
            continue;
        end
    end
    
    fprintf('\n=== Phase 2 Complete: All %d region pairs analyzed ===\n', n_pairs);
    fprintf('Results organized in: %s\n', session_output_dir);
    
    fprintf('\n=== Oxford Neuron Stability Analysis Complete ===\n');
    fprintf('Theoretical contribution: Quantified canonical correlation robustness\n');
    fprintf('Methodological advancement: Multi-region stability assessment framework\n');
    fprintf('Practical implications: Minimum sampling requirements for reliable CCA\n');
end

function available_sessions = discover_valid_sessions(config)
% DISCOVER_VALID_SESSIONS - Identify sessions meeting analysis criteria
% This function implements session-level quality control based on neural
% population sizes and data availability

    fprintf('  Scanning CCA results directory for processed sessions...\n');
    
    results_pattern = fullfile(config.results_dir, '*_CCA_results.mat');
    result_files = dir(results_pattern);
    
    available_sessions = {};
    session_count = 0;
    
    for i = 1:length(result_files)
        try
            % Load session CCA results
            file_path = fullfile(result_files(i).folder, result_files(i).name);
            session_data = load(file_path);
            
            % Extract session name from filename
            [~, name, ~] = fileparts(result_files(i).name);
            session_name = strrep(name, '_CCA_results', '');
            
            % Validate session has required fields
            if ~isfield(session_data, 'region_data') || ~isfield(session_data, 'cca_results')
                fprintf('    Session %s: Missing required data fields\n', session_name);
                continue;
            end
            
            % Check neural population criteria
            region_data = session_data.region_data;
            valid_regions = region_data.valid_regions;
            
            if length(valid_regions) < 2
                fprintf('    Session %s: Insufficient regions (%d)\n', session_name, length(valid_regions));
                continue;
            end
            
            % Find maximum neuron count across regions
            max_neurons = 0;
            valid_pairs = {};
            
            for r1_idx = 1:length(valid_regions)
                for r2_idx = (r1_idx+1):length(valid_regions)
                    region1 = valid_regions{r1_idx};
                    region2 = valid_regions{r2_idx};
                    
                    if isfield(region_data.regions, region1) && isfield(region_data.regions, region2)
                        n1 = region_data.regions.(region1).n_neurons;
                        n2 = region_data.regions.(region2).n_neurons;
                        
                        if n1 >= config.min_region_neurons && n2 >= config.min_region_neurons
                            valid_pairs{end+1} = struct('region1', region1, 'region2', region2, ...
                                                       'n1', n1, 'n2', n2);
                            max_neurons = max(max_neurons, min(n1, n2));
                        end
                        max_neurons = max(max_neurons, min(n1, n2));
                    end
                end
            end
            
            if ~isempty(valid_pairs) && max_neurons >= config.reference_neurons
                session_count = session_count + 1;
                available_sessions{session_count} = struct(...
                    'session_name', session_name, ...
                    'file_path', file_path, ...
                    'region_pairs', {valid_pairs}, ...
                    'max_neurons', max_neurons, ...
                    'n_valid_regions', length(valid_regions));
                
                fprintf('    Session %s: VALID (%d regions, %d pairs, max %d neurons)\n', ...
                        session_name, length(valid_regions), length(valid_pairs), max_neurons);
            else
                fprintf('    Session %s: Insufficient neurons (max %d, need %d)\n', ...
                        session_name, max_neurons, config.reference_neurons);
            end
            
        catch ME
            fprintf('    Session %s: Error loading (%s)\n', session_name, ME.message);
        end
    end
    
    fprintf('  Session discovery complete: %d/%d sessions valid\n', ...
            length(available_sessions), length(result_files));
end

function [optimal_pair, session_data] = select_optimal_region_pair(selected_session, config)
% SELECT_OPTIMAL_REGION_PAIR - Choose the best region pair for stability analysis
% Selection criteria: (1) Neural population size, (2) Baseline correlation strength,
% (3) Data quality metrics, (4) Anatomical relevance

    fprintf('  Loading session data: %s\n', selected_session.session_name);
    
    % Load complete session data
    session_data = load(selected_session.file_path);
    
    % Evaluate each potential region pair
    pair_scores = [];
    
    for i = 1:length(selected_session.region_pairs)
        pair = selected_session.region_pairs{i};
        
        % Calculate composite score for pair selection
        % Score components: population size (40%), correlation strength (40%), data quality (20%)
        
        % Component 1: Population size score (normalized to reference)
        min_neurons = min(pair.n1, pair.n2);
        size_score = min(min_neurons / config.reference_neurons, 1.0);
        
        % Component 2: Baseline correlation strength
        correlation_score = calculate_baseline_correlation(session_data, pair, config);
        
        pair_scores(i) = correlation_score;
        
        fprintf('    Pair %s-%s: Corr Score %.3f', ...
                pair.region1, pair.region2, correlation_score);
    end
    
    % Select optimal pair
    [~, best_idx] = max(pair_scores);
    optimal_pair = selected_session.region_pairs{best_idx};
    
    % Add computed metrics to optimal pair structure
    %optimal_pair.quality_score = assess_pair_data_quality(session_data, optimal_pair);
    optimal_pair.baseline_r2 = calculate_baseline_correlation(session_data, optimal_pair, config);
    optimal_pair.n_neurons1 = optimal_pair.n1;
    optimal_pair.n_neurons2 = optimal_pair.n2;
end

function correlation_score = calculate_baseline_correlation(session_data, pair, config)
% CALCULATE_BASELINE_CORRELATION - Estimate baseline CCA performance for pair
% This provides a rapid assessment of canonical correlation strength without
% full iterative analysis

    try
        % Extract regional spike data
        region_data = session_data.region_data;
        
        % Get spike data for both regions
        region1_indices = region_data.regions.(pair.region1).neuron_indices;
        region2_indices = region_data.regions.(pair.region2).neuron_indices;
        
        % Sample subset of neurons for rapid assessment
        n_sample = min(50, min(length(region1_indices), length(region2_indices)));
        rng(666);
        selected_1 = randsample(region1_indices, n_sample);
        selected_2 = randsample(region2_indices, n_sample);
        

        % Extract and reshape spike data
        spike_data = region_data.spike_data;
        spike_data_p = permute(spike_data,[2,3,1]);
        region_1_sampled_p = spike_data_p(selected_1,:,:);
        region_2_sampled_p = spike_data_p(selected_2,:,:);

        % Flatten trial and time dimensions for correlation analysis
        region1_data = reshape(region_1_sampled_p, n_sample,[])';
        region2_data = reshape(region_2_sampled_p, n_sample,[])';
        
        % Mean center data
        region1_data = region1_data - mean(region1_data);
        region2_data = region2_data - mean(region2_data);
        
        % Quick CCA assessment
        [~, ~, r] = canoncorr(region1_data, region2_data);
        
        % Return primary canonical correlation squared
        correlation_score = r(1);
        
    catch
        % Fallback to conservative estimate
        correlation_score = 0.1;
    end
end

% function quality_score = assess_pair_data_quality(session_data, pair)
% % ASSESS_PAIR_DATA_QUALITY - Comprehensive data quality metrics
% % Quality assessment based on: firing rate distributions, temporal stability,
% % trial-to-trial consistency, and signal-to-noise ratios
% 
%     try
%         region_data = session_data.region_data;
% 
%         % Extract spike data for quality assessment
%         region1_indices = region_data.regions.(pair.region1).neuron_indices;
%         region2_indices = region_data.regions.(pair.region2).neuron_indices;
% 
%         spike_data = region_data.spike_data;
% 
%         region1_spikes = spike_data(:, region1_indices, :);
%         region2_spikes = spike_data(:, region2_indices, :);
% 
%         % Quality Metric 1: Firing rate characteristics
%         r1_rates = mean(region1_spikes(:));
%         r2_rates = mean(region2_spikes(:));
%         rate_score = min(1.0, (r1_rates + r2_rates) / 10.0); % Normalized to reasonable firing rates
% 
%         % Quality Metric 2: Signal variability (coefficient of variation)
%         r1_cv = std(region1_spikes(:)) / mean(region1_spikes(:));
%         r2_cv = std(region2_spikes(:)) / mean(region2_spikes(:));
%         variability_score = 1.0 - min(1.0, (r1_cv + r2_cv) / 4.0); % Lower CV is better
% 
%         % Quality Metric 3: Data completeness (non-NaN proportion)
%         r1_complete = 1 - sum(isnan(region1_spikes(:))) / numel(region1_spikes);
%         r2_complete = 1 - sum(isnan(region2_spikes(:))) / numel(region2_spikes);
%         completeness_score = (r1_complete + r2_complete) / 2;
% 
%         % Composite quality score
%         quality_score = (rate_score + variability_score + completeness_score) / 3;
% 
%     catch
%         quality_score = 0.5; % Conservative fallback
%     end
% end

function reference_latent = establish_reference_latent(session_data, optimal_pair, config)
% ESTABLISH_REFERENCE_LATENT - Create gold standard canonical correlation pattern
% This function establishes the reference pattern using maximum available neurons
% and extensive averaging across random sampling iterations

    fprintf('  Establishing reference latent with %d neurons per region...\n', config.reference_neurons);
    
    % Extract regional data
    region_data = session_data.region_data;
    region1_indices = region_data.regions.(optimal_pair.region1).neuron_indices;
    region2_indices = region_data.regions.(optimal_pair.region2).neuron_indices;
    spike_data = region_data.spike_data;
    
    % Initialize storage for reference iterations
    n_timepoints = size(spike_data, 3);
    n_trials = size(spike_data, 1);
    ref_latents_r1 = zeros(config.n_reference_iterations, n_timepoints);
    ref_latents_r2 = zeros(config.n_reference_iterations, n_timepoints);
    ref_correlations = zeros(config.n_reference_iterations, 1);
    
    fprintf('    Running %d reference iterations...\n', config.n_reference_iterations);
    
    for iter = 1:config.n_reference_iterations
        if mod(iter, 10) == 0
            fprintf('      Reference iteration %d/%d\n', iter, config.n_reference_iterations);
        end
        
        % Set reproducible random seed for this iteration
        rng(5000 + iter);
        
        % Randomly sample neurons
        selected_1 = randsample(region1_indices, config.reference_neurons);
        selected_2 = randsample(region2_indices, config.reference_neurons);
        spike_data_p = permute(spike_data,[2,3,1]);
        region_1_sampled_p = spike_data_p(selected_1,:,:);
        region_2_sampled_p = spike_data_p(selected_2,:,:);

        shuffled_trials = randperm(n_trials, n_trials);
        region_2_sampled_p = region_2_sampled_p(:,:,shuffled_trials);
        
        % Flatten trial and time dimensions for correlation analysis
        r1_data = reshape(region_1_sampled_p, config.reference_neurons,[])';
        r2_data = reshape(region_2_sampled_p, config.reference_neurons,[])';
        % Extract and reshape data for CCA

        
        % Mean center data
        r1_data = r1_data - mean(r1_data);
        r2_data = r2_data - mean(r2_data);
        
        % Perform cross-validated CCA
        [A_mean, B_mean, cv_r2] = perform_crossvalidated_cca(r1_data, r2_data, config);
        
        % Project data onto first canonical component
        r1_proj = r1_data * A_mean(:, 1);
        r2_proj = r2_data * B_mean(:, 1);
        
        % Reshape projections back to trial structure
        n_trials = size(spike_data, 1);
        r1_proj_trials = reshape(r1_proj, n_timepoints,n_trials)';
        r2_proj_trials = reshape(r2_proj, n_timepoints,n_trials)';
        
        % Calculate trial-averaged latent trajectories
        ref_latents_r1(iter, :) = mean(r1_proj_trials, 1);
        ref_latents_r2(iter, :) = mean(r2_proj_trials, 1);
        ref_correlations(iter) = cv_r2(1);
    end
    
    % Compute final reference latent as iteration average
    reference_latent = struct();
    reference_latent.region1_latent = mean(abs(ref_latents_r1), 1); % Use absolute value for stability
    reference_latent.region2_latent = mean(abs(ref_latents_r2), 1);
    reference_latent.region1_crossiter_latent = ref_latents_r1;
    reference_latent.region2_crossiter_latent = ref_latents_r2;
    reference_latent.cross_correlation = mean(ref_correlations);
    reference_latent.correlation_std = std(ref_correlations);
    reference_latent.time_vector = linspace(-1.5, 3, n_timepoints); % Placeholder - should be actual time vector
    reference_latent.iteration_correlations = ref_correlations;
    
    fprintf('    Reference latent established successfully\n');
    fprintf('    Mean : %.3f ± %.3f\n', reference_latent.cross_correlation, reference_latent.correlation_std);
end

function stability_results = assess_multi_scale_stability(session_data, optimal_pair, reference_latent, config)
% ASSESS_MULTI_SCALE_STABILITY - Systematic evaluation across neuron count scales
% This is the core stability analysis, evaluating how canonical correlation patterns
% degrade as neural population samples decrease

    fprintf('  Assessing stability across %d neuron count levels...\n', length(config.neuron_counts));
    
    % Extract regional data
    region_data = session_data.region_data;
    region1_indices = region_data.regions.(optimal_pair.region1).neuron_indices;
    region2_indices = region_data.regions.(optimal_pair.region2).neuron_indices;
    spike_data = region_data.spike_data;
    
    % Initialize results structure
    stability_results = struct();
    stability_results.neuron_counts = config.neuron_counts;
    stability_results.correlations_with_reference = cell(length(config.neuron_counts), 1);
    stability_results.mean_correlations = zeros(length(config.neuron_counts), 2);
    stability_results.std_correlations = zeros(length(config.neuron_counts), 2);
    stability_results.cca_performance = zeros(length(config.neuron_counts), config.n_test_iterations);
    
    % Process each neuron count level
    for nc_idx = 1:length(config.neuron_counts)
        n_neurons = config.neuron_counts(nc_idx);
        fprintf('    Testing %d neurons (%d/%d)...\n', n_neurons, nc_idx, length(config.neuron_counts));
        
        % Initialize storage for this neuron count
        correlations_r1 = zeros(config.n_test_iterations, 1);
        correlations_r2 = zeros(config.n_test_iterations, 1);
        cca_r2_values = zeros(config.n_test_iterations, 1);
        
        % Run multiple iterations with different random samples
        for iter = 1:config.n_test_iterations
            if mod(iter, 10) == 0
                fprintf('      Iteration %d/%d\n', iter, config.n_test_iterations);
            end
            
            % Set reproducible random seed
            rng(1000 + nc_idx * 100 + iter);
            
            % Sample neurons
            selected_1 = randsample(region1_indices, n_neurons);
            selected_2 = randsample(region2_indices, n_neurons);
            n_trials = size(spike_data, 1);

            spike_data_p = permute(spike_data,[2,3,1]);
            region_1_sampled_p = spike_data_p(selected_1,:,:);
            region_2_sampled_p = spike_data_p(selected_2,:,:);

            shuffled_trials = randperm(n_trials, n_trials);
            region_2_sampled_p = region_2_sampled_p(:,:,shuffled_trials);

            % Flatten trial and time dimensions for correlation analysis
            r1_data = reshape(region_1_sampled_p, n_neurons,[])';
            r2_data = reshape(region_2_sampled_p, n_neurons,[])';
            
            r1_data = r1_data - mean(r1_data);
            r2_data = r2_data - mean(r2_data);
            
            % Perform CCA
            [A_mean, B_mean, cv_r2] = perform_crossvalidated_cca(r1_data, r2_data, config);
            
            % Project and reshape
            r1_proj = r1_data * A_mean(:, 1);
            r2_proj = r2_data * B_mean(:, 1);
            
            n_trials = size(spike_data, 1);
            n_timepoints = size(spike_data, 3);


            r1_proj_trials = reshape(r1_proj, n_timepoints,n_trials)';
            r2_proj_trials = reshape(r2_proj, n_timepoints,n_trials)';


            % Calculate latent trajectories
            latent_r1 = mean(r1_proj_trials, 1);
            latent_r2 = mean(r2_proj_trials, 1);
            
            % Calculate correlation with reference latent
            correlations_r1(iter) = corr(abs(latent_r1)', reference_latent.region1_latent');
            correlations_r2(iter) = corr(abs(latent_r2)', reference_latent.region2_latent');
            
            % Store CCA performance
            cca_r2_values(iter) = cv_r2(1);
        end
        
        % Store results for this neuron count
        stability_results.correlations_with_reference{nc_idx} = struct(...
            'region1', correlations_r1, 'region2', correlations_r2);
        stability_results.example_latent{nc_idx} = abs(latent_r1);
        stability_results.mean_correlations(nc_idx, :) = [mean(correlations_r1), mean(correlations_r2)];
        stability_results.std_correlations(nc_idx, :) = [std(correlations_r1), std(correlations_r2)];
        stability_results.cca_performance(nc_idx, :) = cca_r2_values;
        
        fprintf('Mean correlation with reference: R1=%.3f±%.3f, R2=%.3f±%.3f\n', ...
                mean(correlations_r1), std(correlations_r1), ...
                mean(correlations_r2), std(correlations_r2));
    end
    
    fprintf('  Multi-scale stability assessment complete\n');
end

function [A_mean, B_mean, cv_r2] = perform_crossvalidated_cca(data1, data2, config)
% PERFORM_CROSSVALIDATED_CCA - Cross-validated canonical correlation analysis
% Implements robust CCA with cross-validation to prevent overfitting

    n_samples = size(data1, 1);
    fold_size = floor(n_samples / config.n_cv_folds);
    
    A_matrices = zeros(size(data1, 2), config.max_components, config.n_cv_folds);
    B_matrices = zeros(size(data2, 2), config.max_components, config.n_cv_folds);
    fold_r2 = zeros(config.n_cv_folds, config.max_components);
    
    for fold = 1:config.n_cv_folds
        % Define train/test splits
        test_start = (fold - 1) * fold_size + 1;
        test_end = min(fold * fold_size, n_samples);
        test_idx = test_start:test_end;
        train_idx = setdiff(1:n_samples, test_idx);
        
        % Split data
        data1_train = data1(train_idx, :);
        data2_train = data2(train_idx, :);
        data1_test = data1(test_idx, :);
        data2_test = data2(test_idx, :);
        
        % Perform CCA with regularization if needed
        try
            [A_fold, B_fold, r_fold] = canoncorr(data1_train, data2_train);
        catch
            % Apply regularization for rank-deficient cases
            lambda = 0.01;
            n_train = size(data1_train, 1);
            p1 = size(data1_train, 2);
            p2 = size(data2_train, 2);
            
            % Add regularization
            data1_reg = [data1_train; sqrt(lambda) * eye(min(n_train, p1))];
            data2_reg = [data2_train; sqrt(lambda) * eye(min(n_train, p2))];
            
            [A_fold, B_fold, r_fold] = canoncorr(data1_reg, data2_reg);
        end
        
        % Ensure we have enough components
        n_comp_available = size(A_fold, 2);
        n_comp_use = min(config.max_components, n_comp_available);
        
        A_matrices(:, 1:n_comp_use, fold) = A_fold(:, 1:n_comp_use);
        B_matrices(:, 1:n_comp_use, fold) = B_fold(:, 1:n_comp_use);
        
        % Calculate test performance
        for comp = 1:n_comp_use
            proj1_test = data1_test * A_fold(:, comp);
            proj2_test = data2_test * B_fold(:, comp);
            fold_r2(fold, comp) = corr(proj1_test, proj2_test);
        end
    end
    
    % Average across folds
    A_mean = mean(A_matrices, 3);
    B_mean = mean(B_matrices, 3);
    cv_r2 = mean(fold_r2, 1);
end

function final_results = compile_stability_results(stability_results, reference_latent, optimal_pair, selected_session, config)
% COMPILE_STABILITY_RESULTS - Comprehensive results compilation with theoretical interpretation
% This function integrates all analysis components into a unified result structure
% suitable for publication and further analysis

    fprintf('  Compiling comprehensive stability analysis results...\n');
    
    final_results = struct();
    
    % Session and Analysis Metadata
    final_results.metadata = struct();
    final_results.metadata.session_name = selected_session.session_name;
    final_results.metadata.analysis_timestamp = datestr(now);
    final_results.metadata.region_pair = optimal_pair;
    final_results.metadata.config = config;
    
    % Reference Latent Characteristics
    final_results.reference = reference_latent;
    
    % Stability Analysis Results
    final_results.stability = stability_results;
    
    % Theoretical Analysis - Stability Metrics
    fprintf('    Computing theoretical stability metrics...\n');
    
    % Metric 1: Stability Threshold Analysis
    % Determine minimum neuron count for reliable correlation (>0.8 with reference)
    threshold_0_8 = find_stability_threshold(stability_results, 0.8);
    threshold_0_9 = find_stability_threshold(stability_results, 0.9);
    threshold_0_95 = find_stability_threshold(stability_results, 0.95);
    
    final_results.thresholds = struct();
    final_results.thresholds.threshold_80 = threshold_0_8;
    final_results.thresholds.threshold_90 = threshold_0_9;
    final_results.thresholds.threshold_95 = threshold_0_95;
    
    % Metric 2: Degradation Profile Analysis
    % Characterize how correlation degrades with decreasing neuron counts
    degradation_profile = analyze_degradation_profile(stability_results);
    final_results.degradation = degradation_profile;
    
    % Metric 3: Regional Specificity Analysis
    % Compare stability between the two brain regions
    % regional_analysis = compare_regional_stability(stability_results, optimal_pair);
    % final_results.regional_comparison = regional_analysis;
    
    % Theoretical Interpretation Summary
    % final_results.interpretation = generate_theoretical_interpretation(final_results);
    
    fprintf('    Results compilation complete\n');
    fprintf('      Stability thresholds: 80%%=%.0f, 90%%=%.0f, 95%%=%.0f neurons\n', ...
            threshold_0_8, threshold_0_9, threshold_0_95);
end

function threshold = find_stability_threshold(stability_results, correlation_level)
% FIND_STABILITY_THRESHOLD - Determine minimum neurons for correlation threshold
    
    mean_corr_both = mean(stability_results.mean_correlations, 2);
    
    above_threshold = mean_corr_both >= correlation_level;
    
    if any(above_threshold)
        threshold_idx = find(above_threshold, 1, 'first');
        threshold = stability_results.neuron_counts(threshold_idx);
    else
        threshold = NaN; % Threshold not reached within tested range
    end
end

function degradation_profile = analyze_degradation_profile(stability_results)
% ANALYZE_DEGRADATION_PROFILE - Characterize correlation degradation pattern
    
    degradation_profile = struct();
    
    mean_corr_both = mean(stability_results.mean_correlations, 2);
    neuron_counts = stability_results.neuron_counts;
    
    % Fit exponential decay model: correlation = a * exp(b * neurons) + c
    try
        ft = fittype('a*exp(b*x) + c', 'independent', 'x', 'dependent', 'y');
        opts = fitoptions('Method', 'NonlinearLeastSquares');
        opts.StartPoint = [0.5, 0.01, 0.3];
        
        [fitted_model, gof] = fit(neuron_counts', mean_corr_both, ft, opts);
        
        degradation_profile.model = fitted_model;
        degradation_profile.goodness_of_fit = gof;
        degradation_profile.decay_constant = fitted_model.b;
        degradation_profile.asymptote = fitted_model.c;
        
    catch
        degradation_profile.model = [];
        degradation_profile.decay_constant = NaN;
        degradation_profile.asymptote = NaN;
    end
    
    % Calculate degradation rate (correlation change per neuron)
    degradation_rates = diff(mean_corr_both) ./ diff(neuron_counts');
    degradation_profile.mean_degradation_rate = mean(degradation_rates);
    degradation_profile.degradation_rates = degradation_rates;
end

function regional_analysis = compare_regional_stability(stability_results, optimal_pair)
% COMPARE_REGIONAL_STABILITY - Analyze differences between brain regions
    
    regional_analysis = struct();
    regional_analysis.region1_name = optimal_pair.region1;
    regional_analysis.region2_name = optimal_pair.region2;
    
    % Compare mean correlations
    r1_correlations = stability_results.mean_correlations(:, 1);
    r2_correlations = stability_results.mean_correlations(:, 2);
    
    regional_analysis.region1_mean = mean(r1_correlations);
    regional_analysis.region2_mean = mean(r2_correlations);
    regional_analysis.difference = regional_analysis.region1_mean - regional_analysis.region2_mean;
    
    % Statistical comparison
    [~, p_value] = ttest2(r1_correlations, r2_correlations);
    regional_analysis.statistical_significance = p_value;
    
    % Stability comparison
    r1_stability = std(r1_correlations);
    r2_stability = std(r2_correlations);
    regional_analysis.region1_stability = r1_stability;
    regional_analysis.region2_stability = r2_stability;
    regional_analysis.more_stable_region = char(optimal_pair.region1 * (r1_stability < r2_stability) + ...
                                               optimal_pair.region2 * (r1_stability >= r2_stability));
end

function interpretation = generate_theoretical_interpretation(final_results)
% GENERATE_THEORETICAL_INTERPRETATION - Synthesize results into theoretical framework
    
    interpretation = struct();
    
    % Population size requirements
    if ~isnan(final_results.thresholds.threshold_90)
        interpretation.minimum_neurons = final_results.thresholds.threshold_90;
        interpretation.reliability_assessment = 'High reliability achieved with practical neuron counts';
    else
        interpretation.minimum_neurons = max(final_results.stability.neuron_counts);
        interpretation.reliability_assessment = 'High reliability requires > 90 neurons per region';
    end
    
    % Degradation characteristics
    if isfield(final_results.degradation, 'decay_constant') && ~isnan(final_results.degradation.decay_constant)
        if final_results.degradation.decay_constant > 0.05
            interpretation.degradation_type = 'Rapid exponential decay';
        elseif final_results.degradation.decay_constant > 0.02
            interpretation.degradation_type = 'Moderate exponential decay';
        else
            interpretation.degradation_type = 'Slow exponential decay';
        end
    else
        interpretation.degradation_type = 'Complex degradation pattern';
    end
    
    % Regional differences
    if abs(final_results.regional_comparison.difference) > 0.1
        interpretation.regional_specificity = 'Significant inter-regional differences in stability';
    else
        interpretation.regional_specificity = 'Similar stability across brain regions';
    end
    
    % Overall assessment
    mean_high_n_corr = mean(final_results.stability.mean_correlations(end-2:end, :), 'all');
    if mean_high_n_corr > 0.9
        interpretation.overall_assessment = 'Excellent canonical correlation stability';
    elseif mean_high_n_corr > 0.8
        interpretation.overall_assessment = 'Good canonical correlation stability';
    elseif mean_high_n_corr > 0.6
        interpretation.overall_assessment = 'Moderate canonical correlation stability';
    else
        interpretation.overall_assessment = 'Poor canonical correlation stability';
    end
end


%% 

function create_stability_visualizations(final_results, config, session_output_dir)
%%
% CREATE_STABILITY_VISUALIZATIONS - Generate publication-quality figures
% Matches the original analyze_single_session_multiple_iterations visualization framework
    
    fprintf('  Generating publication-quality visualizations...\n');
    
    % Use provided session_output_dir if available, otherwise use default config.output_dir
    if nargin < 3
        output_dir = config.output_dir;
    else
        output_dir = session_output_dir;
    end
    
    % Main comparison figure following original framework
    fig1 = figure('Position', [100, 100, 900, 1200]);
    
    % Create main title with Oxford-specific formatting
    session_name = final_results.metadata.session_name;
    region1 = final_results.metadata.region_pair.region1;
    region2 = final_results.metadata.region_pair.region2;
    
    % Main title
    sgtitle(sprintf('%s vs %s (Session %s)', ...
        regexprep(region1, '_', ' '), regexprep(region2, '_', ' '), regexprep(session_name, '_', ' ')), ...
        'FontSize', 16, 'FontWeight', 'bold');
    
    % Plot 1: Reference latent trajectories (matches original plotReferenceLatents)
    subplot(3, 1, 2);
    plotReferenceLatents(final_results);
    
    % Plot 2: Correlation with reference vs neuron count (matches original plotCorrelationWithReference)
    subplot(3, 1, 1);
    plotCorrelationWithReference(final_results);
    
    % Plot 3: Example comparisons at different neuron counts (matches original plotExampleComparisons)
    subplot(3, 1, 3);
    plotExampleComparisons(final_results);
    
    % Save figure using pair-specific naming
    pair_name = sprintf('%s_vs_%s', region1, region2);
    fig_file = fullfile(output_dir, sprintf('%s_reference_comparison_analysis.png', pair_name));
    %saveas(fig1, fig_file);
    saveas(fig1, fig_file);
    close(fig1);  % Close figure to free memory
    
    fprintf('    Visualizations saved to: %s\n', output_dir);
end

function plotReferenceLatents(final_results)
% Plot the reference latent trajectories with standard deviation envelope
% and individual iteration traces
    ref = final_results.reference;
    
    hold on;
    
    % Define colors for the two regions
    color_region1 = [0, 0, 1];      % Blue for region 1
    color_region2 = [1, 0, 0];      % Red for region 2
    
    % ===== Region 1: Plot individual iterations in very light blue =====
    for iter = 1:size(ref.region1_crossiter_latent, 1)
        plot(ref.time_vector, abs(ref.region1_crossiter_latent(iter, :)), ...
            'Color', [color_region1, 0.15], ...  % 15% opacity
            'LineWidth', 0.5, ...
            'HandleVisibility', 'off');  % Don't show in legend
    end
    
    % ===== Region 2: Plot individual iterations in very light red =====
    for iter = 1:size(ref.region2_crossiter_latent, 1)
        plot(ref.time_vector, abs(ref.region2_crossiter_latent(iter, :)), ...
            'Color', [color_region2, 0.15], ...  % 15% opacity
            'LineWidth', 0.5, ...
            'HandleVisibility', 'off');
    end
    
    % ===== Compute mean and standard deviation =====
    % These should match your averaged reference latents
    mean_region1 = ref.region1_latent;
    mean_region2 = ref.region2_latent;
    
    % Standard deviation across iterations
    std_region1 = std(abs(ref.region1_crossiter_latent), 0, 1);  % std along dimension 1
    std_region2 = std(abs(ref.region2_crossiter_latent), 0, 1);
    
    % ===== Region 1: Plot standard deviation shadow =====
    % Create the envelope: mean ± std
    upper_region1 = mean_region1 + std_region1;
    lower_region1 = mean_region1 - std_region1;
    
    % Use fill to create shaded region (requires [x, fliplr(x)] and [upper, fliplr(lower)])
    time_filled = [ref.time_vector, fliplr(ref.time_vector)];
    region1_filled = [upper_region1, fliplr(lower_region1)];
    
    fill(time_filled, region1_filled, color_region1, ...
        'FaceAlpha', 0.2, ...  % 20% transparency
        'EdgeColor', 'none', ...
        'HandleVisibility', 'off');
    
    % ===== Region 2: Plot standard deviation shadow =====
    upper_region2 = mean_region2 + std_region2;
    lower_region2 = mean_region2 - std_region2;
    
    region2_filled = [upper_region2, fliplr(lower_region2)];
    
    fill(time_filled, region2_filled, color_region2, ...
        'FaceAlpha', 0.2, ...
        'EdgeColor', 'none', ...
        'HandleVisibility', 'off');
    
    % ===== Plot mean reference latents (bold lines on top) =====
    plot(ref.time_vector, ref.region1_latent, 'b-', ...
        'LineWidth', 3, ... 
        'DisplayName', sprintf('%s (Reference \\mu)', ...
            final_results.metadata.region_pair.region1));
    
    plot(ref.time_vector, ref.region2_latent, 'r-', ... 
        'LineWidth', 3, ...
        'DisplayName', sprintf('%s (Reference \\mu)', ...
            final_results.metadata.region_pair.region2));  
    
    % ===== Add vertical line at event onset =====
    xline(0, '--k', 'LineWidth', 2, 'Alpha', 0.5, 'DisplayName', 'Bar off');
    
    % ===== Formatting =====
    xlabel('Time (samples)', 'FontSize', 16);
    xlim([-1.5, 3]);
    ylabel('Projection value', 'FontSize', 16);
    title(sprintf('Reference Latents (%d neurons, %d iterations averaged)', ...
        final_results.metadata.config.reference_neurons, ...
        final_results.metadata.config.n_reference_iterations), ...
        'FontWeight', 'normal');
    legend('Location', 'northeast', 'FontSize', 14, 'FontWeight', 'normal');  
    set(gca, 'FontSize', 16);
    grid on;
    
    hold off;
end

function plotCorrelationWithReference(final_results)
% Plot correlation with reference as a function of neuron count - matches original framework
    
    stability = final_results.stability;
    
    hold on;
    
    % Plot with error bars using original styling
    errorbar(stability.neuron_counts, stability.mean_correlations(:,1), ...
        stability.std_correlations(:,1), ...
        'o-', 'LineWidth', 2.5, 'MarkerSize', 8, ...
        'MarkerFaceColor', 'b', 'Color', 'b', ...
        'DisplayName', regexprep(final_results.metadata.region_pair.region1, '_', ' '));
    
    errorbar(stability.neuron_counts, stability.mean_correlations(:,2), ...
        stability.std_correlations(:,2), ...
        's-', 'LineWidth', 2.5, 'MarkerSize', 8, ...
        'MarkerFaceColor', 'r', 'Color', 'r', ...
        'DisplayName', regexprep(final_results.metadata.region_pair.region2, '_', ' '));
    
    legend('Location', 'southeast','FontSize', 14);
    
    % Add reference threshold lines (matching original framework)
    % yline(1, '--k', 'LineWidth', 1, 'Alpha', 0.5);
    % yline(0.95, '--', 'Color', [0, 0.5, 0], 'LineWidth', 1, 'Alpha', 0.5);
    % yline(0.9, '--', 'Color', [0.5, 0.5, 0], 'LineWidth', 1, 'Alpha', 0.5);
    
    xlabel('Number of Neurons', 'FontSize', 16);
    ylabel('R2', 'FontSize', 16);
    title(sprintf('Latent Similarity to Reference (%d neurons)', ...
        final_results.metadata.config.reference_neurons), 'FontSize', 16,'FontWeight','normal');

    grid on;
    xlim([min(stability.neuron_counts)*0.9, max(stability.neuron_counts)]);
    ylim([0, 1.05]);
    
    % Set font size separately using the axes handle
    set(gca, 'FontSize', 16);
end

function plotExampleComparisons(final_results)
% Plot example latent trajectories at different neuron counts - matches original framework
    
    stability = final_results.stability;
    ref = final_results.reference;
    
    hold on;
    
    % Select representative neuron counts (adapted for Oxford data range)
    available_counts = stability.neuron_counts;
    if length(available_counts) >= 4
        example_counts = [available_counts(2), available_counts(round(end/3)), ...
                         available_counts(round(end-5)), available_counts(round(end-4)),available_counts(end)];
    else
        example_counts = available_counts;
    end
    
    % Define transparency levels (from most to least transparent)
    % For n examples, create n transparency levels
    n_examples = length(example_counts);
    % Create yellow hue variations with transparency
    yellow_base = [0, 0.251, 0.451]; % Base yellow color (RGB)
    transparency = 0.3;
    
    % Plot reference in gray (matches original)
    plot(ref.time_vector, ref.region1_latent, '-', ...
         'Color', [0.5 0.5 0.5], 'LineWidth', 4, ...
         'DisplayName', sprintf('Reference (%dn avg)', final_results.metadata.config.reference_neurons));
    
    % Plot examples for first region
    for i = 1:length(example_counts)
        nc = example_counts(i);
        nc_idx = find(stability.neuron_counts == nc);
        if ~isempty(nc_idx)
            intensity_factor = 0.2 + 0.9 * (i / length(example_counts));
            current_color = yellow_base * intensity_factor;
            % Calculate example latent as in original code
            correlation_factor = stability.mean_correlations(nc_idx, 1);
            example_latent = stability.example_latent{nc_idx};
            
            % Plot with transparent yellow - note the RGBA specification
            plot(ref.time_vector, abs(example_latent), '-', ...
                'Color', [current_color, transparency], ... % [R, G, B, Alpha]
                'LineWidth', 2, ...
                'DisplayName', sprintf('%d neurons', nc));
        end
    end

% To enhance the transparency effect for filled areas, you could use:
% fill(x, y, baseYellow, 'FaceAlpha', alphaLevels(i), 'EdgeColor', 'none')
    
    % Add event marker at midpoint
    xline(0, '--k', 'LineWidth', 1.5, 'Alpha', 0.5, 'DisplayName', 'Bar off');

    xlabel('Time (samples)', 'FontSize', 16);
    xlim([-1.5, 3]);
    
    ylabel('Projection value', 'FontSize', 16);
    title(sprintf('%s - Example Comparisons', regexprep(final_results.metadata.region_pair.region1, '_', ' ')),'FontWeight','normal');
    %legend('Location', 'northwest','FontSize', 16);
    grid on;
    set(gca, 'FontSize', 16);
    
end

%% Execute the analysis
oxford_neuron_stability_analysis();