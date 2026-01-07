function region_pca = perform_region_pca(spike_data, config)
% PERFORM_REGION_PCA - Principal component analysis with cross-validation
%
% THEORETICAL FOUNDATION:
% This function implements PCA as a dimensionality reduction technique that
% precedes canonical correlation analysis. The method follows the same rigorous
% statistical framework as CCA, including cross-validation and trial-level
% shuffling to ensure robust component identification.
%
% MATHEMATICAL FRAMEWORK:
% Given neural activity X ∈ ℝ^{trials×neurons×timepoints}, we:
% 1. Reshape to (neurons × trial*timepoints) format
% 2. Apply mean centering: X̂ = X - μ(X)
% 3. Compute covariance: C = (1/N)X̂ᵀX̂
% 4. Extract principal components via eigendecomposition
% 5. Project data onto PC space with cross-validation
%
% CROSS-VALIDATION PROTOCOL:
% Following the CCA methodology in perform_session_cca.m:
% - 10-fold cross-validation for robust component selection
% - Trial-level shuffling to remove temporal dependencies
% - Train/test splits to evaluate generalization performance
%
% INPUTS:
%   spike_data - Neural activity (trials × neurons × timepoints)
%   config     - Configuration structure with PCA parameters
%
% OUTPUT:
%   region_pca - Structure containing:
%     .coefficients       - PCA coefficient matrix (neurons × components)
%     .explained_variance - Variance explained by each component
%     .explained_variance_ratio - Proportion of variance explained
%     .n_components       - Number of retained components
%     .cv_results         - Cross-validation performance metrics
%     .projections        - Trial-level projections in PC space

    try
        n_trials = size(spike_data, 1);
        n_neurons = size(spike_data, 2);
        n_timepoints = size(spike_data, 3);
        
        fprintf('      Input dimensions: %d trials × %d neurons × %d timepoints\n', ...
               n_trials, n_neurons, n_timepoints);
        
        %% Step 1: Data Reshaping and Organization
        % CRITICAL: Following the exact data organization used in analyze_region_pair_cca
        % from perform_session_cca.m to maintain methodological consistency.
        %
        % Original CCA approach (lines 122-136 of perform_session_cca.m):
        %   1. Permute to (neurons × timepoints × trials)
        %   2. Apply trial-level shuffling
        %   3. Reshape to (neurons × trial*timepoints)
        %   4. Transpose to (trial*timepoints × neurons)
        
        % Trial-level shuffling (consistent with CCA line 127)
        rng(12345, 'twister');
        shuffled_trials = randperm(n_trials, n_trials);
        
        % Permute data to (neurons × timepoints × trials) - CCA line 129
        spike_data_permuted = permute(spike_data, [2, 3, 1]);
        
        % Apply trial shuffling - CCA line 131
        spike_data_shuffled = spike_data_permuted(:, :, shuffled_trials);
        
        % Reshape to (neurons × trial*timepoints) - CCA line 133
        X = reshape(spike_data_shuffled, n_neurons, n_trials * n_timepoints);
        
        % Transpose to (trial*timepoints × neurons) - CCA line 135
        X = X';
        
        % Mean centering (critical for PCA) - CCA line 139
        X_mean = mean(X);
        X_centered = X - X_mean;
        
        fprintf('      Data reshaped: %d observations × %d features\n', ...
               size(X_centered, 1), size(X_centered, 2));
        
        %% Step 2: Determine Number of Components
        % Select components based on variance threshold or maximum count
        max_components = min(size(X_centered, 1), size(X_centered, 2));
        %max_components = min(size(X_centered, 1), 50);

        % Use configuration parameter if available, otherwise use CCA default
        if isfield(config, 'pca_n_components')
            n_components = min(config.pca_n_components, max_components);
        else
            % Default to same component count as CCA for consistency
            n_components = min(config.target_neurons, max_components);
        end
        
        fprintf('      Extracting %d principal components\n', n_components);
        
        %% Step 3: Cross-Validated PCA Analysis
        % METHODOLOGICAL CONSISTENCY:
        % Following the 10-fold cross-validation protocol from perform_cv_cca
        % in perform_session_cca.m (lines 188-251) to ensure statistical rigor.
        
        n_folds = config.cv_folds; % Typically 10, matching CCA
        fold_size = floor(size(X_centered, 1) / n_folds);
        
        % Initialize cross-validation arrays
        cv_reconstruction_error = zeros(n_folds, n_components);
        pca_coefficients_cv = zeros(n_neurons, n_components, n_folds);
        explained_variance_cv = zeros(n_folds, max_components);
        
        fprintf('      Cross-validation: %d folds\n', n_folds);
        
        for fold = 1:n_folds
            % Define train/test split (CCA lines 207-213)
            test_idx = (fold-1)*fold_size + 1 : fold*fold_size;
            train_idx = setdiff(1:size(X_centered, 1), test_idx);
            
            X_train = X_centered(train_idx, :);
            X_test = X_centered(test_idx, :);
            
            % Perform PCA on training data
            [coeff_fold, ~, ~,~,latent_fold] = pca(X_train, 'NumComponents', n_components);
            
            % Store coefficients and explained variance
            pca_coefficients_cv(:, :, fold) = coeff_fold;
            explained_variance_cv(fold, :) = latent_fold';
            
            % Evaluate on test set: compute reconstruction error for each component
            for comp = 1:n_components
                % Project test data onto principal components
                X_test_proj = X_test * coeff_fold(:, 1:comp);
                
                % Reconstruct from projection
                X_test_recon = X_test_proj * coeff_fold(:, 1:comp)';
                
                % Compute reconstruction error (mean squared error)
                cv_reconstruction_error(fold, comp) = mean(sum((X_test - X_test_recon).^2, 2));
            end
        end
        % coeff_final = mean(pca_coefficients_cv, 3);


        %% Step 4: Compute Final PCA on Full Dataset
        % After cross-validation, compute final PCA on complete dataset
        % This provides the definitive transformation matrix
        
        % 
        % [coeff_final, score_final, latent_final, ~, explained_final] = pca(X_centered, ...
        %      'NumComponents', n_components);
        
        coeff_final = mean(pca_coefficients_cv, 3);

        min_global = min(abs(coeff_final(:)));  % Global minimum value
        max_global = max(abs(coeff_final(:)));  % Global maximum value
        
        % Calculate the range (denominator in our normalization formula)
        data_range = max_global - min_global;
        
        % Apply the min-max normalization transformation
        % Each element is shifted by the minimum and scaled by the range
        coeff_final = (coeff_final - min_global) / data_range;

        %% Step 5: Calculate Trial-Level Projections
        % CRITICAL: Following the projection methodology from calculate_canonical_projections
        % in perform_session_cca.m (lines 320-377) to enable direct comparison with CCA results.
        %
        % Original data organization for trial-level analysis:
        %   1. Start with original spike_data (trials × neurons × timepoints)
        %   2. Permute to (neurons × timepoints × trials)
        %   3. Reshape to (neurons × trial*timepoints)
        %   4. Transpose and project
        %   5. Reshape back to trial-level format
        
        % Permute original data (no shuffling for projections) - CCA line 330
        spike_data_p = permute(spike_data, [2, 3, 1]);
        
        % Reshape to (neurons × trial*timepoints) - CCA line 334
        X_proj = reshape(spike_data_p, n_neurons, n_trials * n_timepoints)';
        [X_proj,mean_X,std_X] = zscore(X_proj,0,1);
        % X_proj2 = zscore(X_proj,0,2);
        % Mean center consistently - CCA line 339
        % X_proj = X_proj - mean(X_proj);
        
        % Project onto principal components
        projections_all = X_proj * coeff_final;
        
        % Reshape back to trial format (timepoints × trials) then transpose
        % Following CCA lines 356-357
        projections_trial = cell(n_components, 1);
        for comp = 1:n_components
            proj_comp = projections_all(:, comp);
            proj_trial = reshape(proj_comp, n_timepoints, n_trials)';
            projections_trial{comp} = proj_trial;
        end
        
        %% Step 6: Construct Comprehensive Results Structure
        region_pca = struct();
        
        % PCA transformation matrices
        % region_pca.coefficients = coeff_final;           % Principal component loadings
        % region_pca.scores = score_final;                 % Projected data in PC space
        % region_pca.latent = latent_final;               % Eigenvalues (variance)
        % region_pca.explained = explained_final;          % Percentage of variance explained
        
        % Variance metrics
        region_pca.explained_variance = mean(explained_variance_cv,1);

        region_pca.explained_variance_ratio = region_pca.explained_variance' ;
        region_pca.cumulative_variance = cumsum(region_pca.explained_variance');

        region_pca.coefficients = coeff_final;

        % Component information
        region_pca.n_components = n_components;
        region_pca.n_neurons = n_neurons;
        region_pca.n_trials = n_trials;
        region_pca.n_timepoints = n_timepoints;
        
        % Cross-validation results (analogous to CCA cv_results)
        region_pca.cv_results = struct();
        region_pca.cv_results.reconstruction_error = cv_reconstruction_error;
        region_pca.cv_results.mean_reconstruction_error = mean(cv_reconstruction_error, 1);
        region_pca.cv_results.std_reconstruction_error = std(cv_reconstruction_error, 0, 1);
        region_pca.cv_results.pca_coefficients_cv = pca_coefficients_cv;
        region_pca.cv_results.explained_variance_cv = explained_variance_cv;
        region_pca.cv_results.mean_coefficients = mean(pca_coefficients_cv, 3);
        
        % Trial-level projections for visualization (analogous to CCA projections)
        region_pca.projections = struct();
        region_pca.projections.components = projections_trial;
        region_pca.projections.time_axis = linspace(-1.5, 3.0, n_timepoints); % Standard timing
        region_pca.projections.n_components = n_components;
        
        % Calculate summary statistics for each component
        for comp = 1:n_components
            proj_data = projections_trial{comp};
            region_pca.projections.mean{comp} = mean(proj_data, 1);
            region_pca.projections.std{comp} = std(proj_data, 0, 1);
        end
        
        % Metadata
        region_pca.data_mean = X_mean;
        region_pca.analysis_timestamp = datestr(now);
        
        fprintf('      PCA completed: %d components, %.2f%% top5 variance explained\n', ...
               n_components, sum(region_pca.explained_variance_ratio(1:5)));
        
    catch ME
        fprintf('      Error in PCA analysis: %s\n', ME.message);
        fprintf('      Stack trace: %s (Line %d)\n', ME.stack(1).name, ME.stack(1).line);
        region_pca = [];
    end
end