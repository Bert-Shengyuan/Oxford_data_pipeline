function cca_results = perform_session_pcca(region_data, session_name, config)
% PERFORM_SESSION_PCCA  Partial canonical correlation analysis across all region pairs.
%
% Implements pCCA following Gonzalez et al. (2026, Nature):
%
%   "pCCA is a statistical technique for quantifying the linear relationship
%    between two multivariate sets after removing the influence of one or more
%    covariate sets."
%
% For each target pair (X_i, X_j), the jointly recorded activity of all
% other valid regions is assembled into a nuisance matrix Z.  The linear
% projection of Z onto X_i and X_j is removed before CCA is applied to
% the residuals:
%
%   X_res = X - Z (Z'Z + lambda*I)^{-1} Z' X
%   Y_res = Y - Z (Z'Z + lambda*I)^{-1} Z' Y
%   [W_x, W_y] = CCA( X_res, Y_res )
%
% This removes apparent pairwise correlations that are driven by a third
% recorded area (see Supplementary Fig. 3 of Gonzalez et al. 2026).
%
% IMPORTANT IMPLEMENTATION DETAIL — WITHIN-FOLD NUISANCE REGRESSION:
%   The Z-regression coefficients (beta_X, beta_Y) are estimated on the
%   training fold only; the same coefficients are then applied to the test
%   fold.  This prevents leakage from future samples into the residuals used
%   for CCA evaluation — a subtlety not visible in the paper but necessary
%   for an unbiased CV estimate.
%
% FALLBACK:  When fewer than three valid regions exist, or if no other region
%   survives the preprocessing step, the function falls back to standard CCA
%   and sets pair_result.is_partial = false.
%
% FUNCTION SIGNATURE (identical to perform_session_cca.m and
%   perform_session_tkcca.m):
%
%   cca_results = perform_session_pcca(region_data, session_name, config)
%
% REQUIRED config FIELDS (inherit from existing pipeline):
%   .cv_folds               number of CV folds (e.g., 5)
%   .significance_threshold percentile threshold for significance (e.g., 95)
%   .min_neurons_per_region minimum neuron count
%
% OPTIONAL config FIELD (new, specific to pCCA):
%   .pcca_ridge_lambda      ridge penalty for nuisance regression (default 1e-4)
%   .pcca_n_components_cap  hard cap on CCA components extracted (default: no cap)
%
% RETURNED STRUCTURE — cca_results:
%   .session_name
%   .analysis_timestamp
%   .config
%   .region_pairs           cell array of 'RegionA_RegionB' labels
%   .pair_results           cell array of pair_result structs (see below)
%
% FIELDS IN EACH pair_result:
%
%   --- Legacy interface (mirrors perform_session_cca.m) ---
%   .region_i               name of source region
%   .region_j               name of target region
%   .target_neurons         number of sampled neurons per region
%   .selected_neurons_i     neuron indices sampled from region_i
%   .selected_neurons_j     neuron indices sampled from region_j
%   .original_neuron_counts [n_orig_i, n_orig_j]
%   .cv_results             struct with subfields:
%       .cv_R2              (n_folds, n_components)   per-fold held-out corr
%       .mean_cv_R2         (1, n_components)         mean across folds
%       .std_cv_R2          (1, n_components)         std  across folds
%       .mean_A_matrix      (n_i, n_components)       fold-averaged W_x
%       .mean_B_matrix      (n_j, n_components)       fold-averaged W_y
%       .A_matrices         (n_i, n_components, n_folds)
%       .B_matrices         (n_j, n_components, n_folds)
%   .significant_components indices of significant pCCA dimensions
%   .max_R2                 maximum canonical correlation across components
%   .mean_R2                mean canonical correlation across components
%   .mean_A_matrix          (n_i, n_components)  duplicate of cv_results field
%   .mean_B_matrix          (n_j, n_components)
%   .projections            struct with trial-by-time projections onto sig dims
%
%   --- pCCA-specific fields (new relative to perform_session_cca.m) ---
%   .nuisance_regions       cell array of region names composing Z
%   .nuisance_n_neurons     total number of neurons in Z
%   .variance_X_retained    fraction of X variance remaining after Z removal
%   .variance_Y_retained    fraction of Y variance remaining after Z removal
%   .is_partial             logical; false if Z was empty (fallback to CCA)
%   .dominant_rho           canonical correlation of the first (dominant) dim
%   .subspace_dim           number of statistically significant pCCA dimensions
%   .gini_weights_i         Gini coefficient of |W_x| on the dominant dim
%   .gini_weights_j         Gini coefficient of |W_y| on the dominant dim
%   .mutual_information     MI = -sum_k log(1 - rho_k^2) over all components
%                           (equation from Gonzalez et al. 2026 Methods)

    fprintf('  Executing partial canonical correlation analysis (pCCA)...\n');

    cca_results = struct();
    cca_results.session_name       = session_name;
    cca_results.analysis_timestamp = datestr(now);
    cca_results.config             = config;
    cca_results.region_pairs       = {};
    cca_results.pair_results       = {};

    n_pairs = size(region_data.region_pairs, 1);
    if n_pairs == 0
        fprintf('  No valid region pairs — pCCA skipped.\n');
        return;
    end

    % ======================================================================
    %  STEP 1:  Preprocess ALL valid regions once, outside the pair loop.
    %
    %  Rationale: (a) efficiency — each region is reshaped/z-scored only
    %  once; (b) consistency — all regions entering Z are preprocessed
    %  identically to X and Y, satisfying the Gaussian assumption stated in
    %  the paper ("All smoothed spike trains were z-scored by subtracting
    %  the mean firing rate and dividing by the standard deviation").
    %
    %  Each region becomes an (N_samples x n_neurons) matrix where
    %  N_samples = n_trials * n_timepoints and every column has zero mean and
    %  unit variance across all samples.
    % ======================================================================
    fprintf('  Preprocessing all valid regions (reshape + z-score)...\n');

    region_matrices = struct();   % z-scored (N_samples x n_neurons) per region
    region_neuron_idx = struct(); % sampled neuron indices per region

    for reg_idx = 1:length(region_data.valid_regions)
        rname = region_data.valid_regions{reg_idx};

        if ~isfield(region_data.regions, rname)
            continue;
        end
        sel = region_data.regions.(rname).selected_neurons;
        if isempty(sel)
            continue;
        end

        raw = region_data.regions.(rname).spike_data(:, sel, :);  % (n_trials, n_sel, T)
        if isempty(raw)
            continue;
        end

        [~, n_neu, ~] = size(raw);
        if n_neu < config.min_neurons_per_region
            fprintf('    %s: only %d neurons — skipped.\n', rname, n_neu);
            continue;
        end

        % Reshape: (n_neurons, T, n_trials) -> (N_samples, n_neurons)
        raw_p = permute(raw, [2, 3, 1]);
        mat   = reshape(raw_p, n_neu, numel(raw) / n_neu)';

        % Z-score each neuron across all samples (paper convention)
        mat = zscore(mat, 0, 1);

        region_matrices.(rname)    = mat;
        region_neuron_idx.(rname)  = sel;
    end

    all_valid = fieldnames(region_matrices);
    n_valid   = length(all_valid);

    fprintf('  %d/%d regions have valid preprocessed data: %s\n', ...
            n_valid, length(region_data.valid_regions), strjoin(all_valid, ', '));

    if n_valid < 2
        fprintf('  Fewer than 2 regions have valid data — pCCA aborted.\n');
        return;
    end

    % ======================================================================
    %  STEP 2:  Pair loop.
    % ======================================================================
    for pair_idx = 1:n_pairs
        ri_idx = region_data.region_pairs(pair_idx, 1);
        rj_idx = region_data.region_pairs(pair_idx, 2);

        ri_name = region_data.valid_regions{ri_idx};
        rj_name = region_data.valid_regions{rj_idx};

        fprintf('    Pair %d/%d: %s — %s\n', pair_idx, n_pairs, ri_name, rj_name);

        if ~isfield(region_matrices, ri_name) || ~isfield(region_matrices, rj_name)
            fprintf('      Skipping: one or both regions absent from preprocessed set.\n');
            continue;
        end

        pair_result = run_pcca_pair(region_matrices, region_neuron_idx, ...
                                    ri_name, rj_name, all_valid, ...
                                    region_data, config);

        if ~isempty(pair_result)
            cca_results.region_pairs{end+1} = sprintf('%s_%s', ri_name, rj_name);
            cca_results.pair_results{end+1} = pair_result;

            fprintf('      Done — dominant rho=%.3f, %d sig. dims, is_partial=%d\n', ...
                    pair_result.dominant_rho, pair_result.subspace_dim, ...
                    pair_result.is_partial);
        else
            fprintf('      pCCA failed for this pair.\n');
        end
    end

    fprintf('  pCCA completed for %d/%d pairs.\n', ...
            length(cca_results.pair_results), n_pairs);
end


% ==========================================================================
%  Core pair-level pCCA
% ==========================================================================
function pair_result = run_pcca_pair(region_matrices, region_neuron_idx, ...
                                      ri_name, rj_name, all_valid, ...
                                      region_data, config)
% RUN_PCCA_PAIR  Partial CCA between ri_name and rj_name, conditioning on
% the concatenated activity of all other simultaneously recorded regions.
%
% The nuisance regression is performed WITHIN each CV fold: beta coefficients
% are fit on training-fold data and then applied to test-fold data, preventing
% any information leakage across the CV split.

    try
        X_full = region_matrices.(ri_name);   % (N_samples, n_i)
        Y_full = region_matrices.(rj_name);   % (N_samples, n_j)

        [N_samples, n_i] = size(X_full);
        n_j              = size(Y_full, 2);

        % --- Ridge penalty for nuisance regression ---
        if isfield(config, 'pcca_ridge_lambda')
            lambda_reg = config.pcca_ridge_lambda;
        else
            lambda_reg = 1e-4;
        end

        % ----------------------------------------------------------------
        %  Build nuisance matrix Z from all OTHER regions.
        %
        %  setdiff over cell arrays preserves the alphabetical ordering of
        %  the remaining regions, which makes nuisance_regions deterministic
        %  across sessions and therefore interpretable in cross-session
        %  comparisons.
        % ----------------------------------------------------------------
        nuisance_names = setdiff(all_valid, {ri_name, rj_name});
        is_partial     = ~isempty(nuisance_names);

        if is_partial
            Z_full = [];
            for k = 1:length(nuisance_names)
                Z_full = [Z_full, region_matrices.(nuisance_names{k})]; %#ok<AGROW>
            end
            nuisance_n_neurons = size(Z_full, 2);
            fprintf('      Nuisance Z: %s (%d neurons)\n', ...
                    strjoin(nuisance_names, ' + '), nuisance_n_neurons);
        else
            Z_full             = [];
            nuisance_n_neurons = 0;
            fprintf('      Only 2 valid regions — using standard CCA (is_partial=false).\n');
        end

        % ----------------------------------------------------------------
        %  Number of CCA components to extract.
        %  Upper bound: min(n_i, n_j).  Optional hard cap from config.
        % ----------------------------------------------------------------
        n_components = min(n_i, n_j);
        if isfield(config, 'pcca_n_components_cap') && ...
                ~isempty(config.pcca_n_components_cap)
            n_components = min(n_components, config.pcca_n_components_cap);
        end

        n_folds   = config.cv_folds;
        fold_size = floor(N_samples / n_folds);

        if fold_size < n_components
            % Degenerate case: too few samples per fold for reliable CCA
            fprintf('      Warning: fold_size=%d < n_components=%d; capping.\n', ...
                    fold_size, n_components);
            n_components = max(1, fold_size - 1);
        end

        if n_folds < 2
            fprintf('      Cannot perform CV with fewer than 2 folds.\n');
            pair_result = [];
            return;
        end

        rng(12345, 'twister');

        cv_rho    = nan(n_folds, n_components);
        Wx_cv     = zeros(n_i, n_components, n_folds);
        Wy_cv     = zeros(n_j, n_components, n_folds);
        var_X_ret = nan(n_folds, 1);
        var_Y_ret = nan(n_folds, 1);

        fprintf('      CV: %d folds, %d components, N_samples=%d\n', ...
                n_folds, n_components, N_samples);

        for fold = 1:n_folds
            test_idx  = (fold-1)*fold_size + 1 : fold*fold_size;
            train_idx = setdiff(1:N_samples, test_idx);

            X_tr = X_full(train_idx, :);
            Y_tr = Y_full(train_idx, :);
            X_te = X_full(test_idx,  :);
            Y_te = Y_full(test_idx,  :);

            if is_partial
                Z_tr = Z_full(train_idx, :);
                Z_te = Z_full(test_idx,  :);

                % --- Nuisance regression (train fold only) ---
                % Solve: beta = (Z'Z + lambda*I)^{-1} Z' X  via normal equations.
                % The (n_z x n_z) system is cheap when n_z << N_samples.
                n_z     = size(Z_tr, 2);
                ZtZ_reg = Z_tr' * Z_tr + lambda_reg * eye(n_z);

                beta_X = ZtZ_reg \ (Z_tr' * X_tr);   % (n_z, n_i)
                beta_Y = ZtZ_reg \ (Z_tr' * Y_tr);   % (n_z, n_j)

                X_res_tr = X_tr - Z_tr * beta_X;     % residuals, train
                Y_res_tr = Y_tr - Z_tr * beta_Y;

                % Apply SAME beta to test fold (no leakage)
                X_res_te = X_te - Z_te * beta_X;
                Y_res_te = Y_te - Z_te * beta_Y;

                % Diagnostic: fraction of variance retained after Z removal
                % (train fold; averaged over folds for the final report)
                var_X_ret(fold) = trace(X_res_tr' * X_res_tr) / ...
                                  (trace(X_tr' * X_tr) + eps);
                var_Y_ret(fold) = trace(Y_res_tr' * Y_res_tr) / ...
                                  (trace(Y_tr' * Y_tr) + eps);
            else
                % Standard CCA: residuals equal raw inputs
                X_res_tr = X_tr;  Y_res_tr = Y_tr;
                X_res_te = X_te;  Y_res_te = Y_te;
                var_X_ret(fold) = 1;
                var_Y_ret(fold) = 1;
            end

            % --- CCA on residuals of training fold ---
            [Wx_f, Wy_f] = local_robust_cca(X_res_tr, Y_res_tr, n_components);
            if isempty(Wx_f)
                fprintf('        Fold %d: CCA on residuals failed.\n', fold);
                continue;
            end

            Wx_cv(:, :, fold) = Wx_f;
            Wy_cv(:, :, fold) = Wy_f;

            % --- Held-out correlation on residuals of test fold ---
            for c = 1:n_components
                u = X_res_te * Wx_f(:, c);
                v = Y_res_te * Wy_f(:, c);
                if std(u) > 0 && std(v) > 0
                    cv_rho(fold, c) = corr(u, v);
                end
            end
        end

        mean_cv_rho = mean(cv_rho, 1, 'omitnan');
        std_cv_rho  = std(cv_rho,  0, 1, 'omitnan');
        Wx_mean     = mean(Wx_cv, 3);
        Wy_mean     = mean(Wy_cv, 3);

        % --- Significance: same percentile rule as the legacy pipeline ----
        sig = find(mean_cv_rho >= prctile(mean_cv_rho, config.significance_threshold));

        % ----------------------------------------------------------------
        %  Assemble cv_results substruct — mirrors the legacy interface so
        %  all downstream plotting / aggregation code that reads .cv_results
        %  continues to work without modification.
        % ----------------------------------------------------------------
        cv_results = struct( ...
            'cv_R2',         cv_rho, ...
            'mean_cv_R2',    mean_cv_rho, ...
            'std_cv_R2',     std_cv_rho, ...
            'mean_A_matrix', Wx_mean, ...
            'mean_B_matrix', Wy_mean, ...
            'A_matrices',    Wx_cv, ...
            'B_matrices',    Wy_cv);

        % --- Legacy scalar metadata ---
        sel_i  = region_neuron_idx.(ri_name);
        sel_j  = region_neuron_idx.(rj_name);
        orig_i = region_data.regions.(ri_name).original_neurons;
        orig_j = region_data.regions.(rj_name).original_neurons;

        % ----------------------------------------------------------------
        %  Construct pair_result: legacy fields first, pCCA-specific after.
        % ----------------------------------------------------------------
        pair_result = struct();

        % --- Legacy fields ---
        pair_result.region_i               = ri_name;
        pair_result.region_j               = rj_name;
        pair_result.target_neurons         = region_data.regions.(ri_name).target_neurons;
        pair_result.selected_neurons_i     = sel_i;
        pair_result.selected_neurons_j     = sel_j;
        pair_result.original_neuron_counts = [orig_i, orig_j];
        pair_result.cv_results             = cv_results;
        pair_result.significant_components = sig;
        pair_result.max_R2                 = max(mean_cv_rho);
        pair_result.mean_R2                = mean(mean_cv_rho, 'omitnan');
        pair_result.mean_A_matrix          = Wx_mean;
        pair_result.mean_B_matrix          = Wy_mean;

        % --- pCCA-specific fields ---

        % Which regions composed Z, and how many neurons total
        pair_result.nuisance_regions    = nuisance_names;
        pair_result.nuisance_n_neurons  = nuisance_n_neurons;

        % Variance retained in X and Y after nuisance removal.
        % Values close to 1.0 mean Z explains little shared variance and the
        % pCCA result will be close to plain CCA.  Low values indicate strong
        % confounding influence from Z, making partialling essential.
        pair_result.variance_X_retained = mean(var_X_ret, 'omitnan');
        pair_result.variance_Y_retained = mean(var_Y_ret, 'omitnan');

        % Flag so downstream code knows whether partial conditioning was applied
        pair_result.is_partial          = is_partial;

        % Dominant-dimension metrics
        if ~isempty(sig)
            pair_result.dominant_rho  = mean_cv_rho(sig(1));
            pair_result.subspace_dim  = numel(sig);

            % Gini coefficient of absolute subspace weight magnitudes.
            % Follows Gonzalez et al. (2026) Methods: monotonically decreasing
            % Gini along CA3->CA1->RSC captures progressive subspace expansion.
            pair_result.gini_weights_i = gini_coeff(abs(Wx_mean(:, sig(1))), n_i);
            pair_result.gini_weights_j = gini_coeff(abs(Wy_mean(:, sig(1))), n_j);
        else
            pair_result.dominant_rho   = max(mean_cv_rho);
            pair_result.subspace_dim   = 0;
            pair_result.gini_weights_i = NaN;
            pair_result.gini_weights_j = NaN;
        end

        % Mutual information estimate.
        % From Gonzalez et al. (2026) Methods:
        %   MI = -sum_i log(1 - rho_i^2)
        % where rho_i is the i-th canonical correlation over all (not just
        % significant) components.  Assumes jointly Gaussian populations.
        rho2 = min(mean_cv_rho .^ 2, 1 - eps);   % guard numerical rho > 1
        pair_result.mutual_information = -sum(log(1 - rho2));

        % --- Projections (visualization) ---
        if ~isempty(sig)
            pair_result.projections = calculate_pcca_projections( ...
                region_data, ri_name, rj_name, is_partial, ...
                Z_full, Wx_mean, Wy_mean, sig, mean_cv_rho, lambda_reg);
        end

        fprintf('      max rho=%.3f, mean rho=%.3f, MI=%.3f, %d sig. dims\n', ...
                pair_result.max_R2, pair_result.mean_R2, ...
                pair_result.mutual_information, pair_result.subspace_dim);
        fprintf('      var. retained: X=%.2f  Y=%.2f\n', ...
                pair_result.variance_X_retained, pair_result.variance_Y_retained);

    catch ME
        fprintf('      Error in pCCA pair: %s\n', ME.message);
        pair_result = [];
    end
end


% ==========================================================================
%  Helpers
% ==========================================================================

function [A, B] = local_robust_cca(X, Y, n_components)
% LOCAL_ROBUST_CCA  Ridge-augmented canoncorr, identical in logic to the
% existing robust_cca routines in perform_session_cca.m and the local copy
% in perform_session_tkcca.m.  Duplicated here so this file is self-contained.

    try
        rX = rank(X);
        rY = rank(Y);
        n_comp_safe = min([n_components, rX, rY, size(X,2), size(Y,2)]);

        if n_comp_safe < 1
            A = []; B = []; return;
        end

        if rX >= n_components && rY >= n_components
            [A, B, ~] = canoncorr(X, Y);
        else
            fprintf('          Regularising CCA (rank X=%d, Y=%d)\n', rX, rY);
            lambda = 0.01;
            A = []; B = [];
            for attempt = 1:5
                X_reg = [X; sqrt(lambda) * eye(n_components, size(X,2))];
                Y_reg = [Y; sqrt(lambda) * eye(n_components, size(Y,2))];
                try
                    [A, B, ~] = canoncorr(X_reg, Y_reg);
                    if size(A,2) >= n_components, break; end
                catch
                    lambda = lambda * 10;
                end
            end
        end

        if isempty(A) || isempty(B)
            return;
        end

        if size(A,2) < n_components || size(B,2) < n_components
            kept = min(size(A,2), size(B,2));
            if kept > 0
                A = A(:,1:kept); B = B(:,1:kept);
                if kept < n_components
                    A = [A, zeros(size(A,1), n_components-kept)];
                    B = [B, zeros(size(B,1), n_components-kept)];
                end
            else
                A = []; B = [];
            end
        else
            A = A(:,1:n_components);
            B = B(:,1:n_components);
        end

    catch ME
        fprintf('          local_robust_cca error: %s\n', ME.message);
        A = []; B = [];
    end
end


function G = gini_coeff(w_abs, N)
% GINI_COEFF  Gini coefficient of absolute subspace weight magnitudes.
%
% Matches the formulation in Gonzalez et al. (2026) Methods:
%
%   G = (N / (N-1)) * [ sum_i (2i - N - 1) * x_i ] / (N^2 * mean(x))
%
% where x_i are the weight magnitudes sorted in ascending order.
% The factor N/(N-1) is Bessel's correction, making G an unbiased estimator.
% G = 0 means uniform weights (all neurons equally involved);
% G = 1 means a single neuron carries all the weight (maximally sparse).

    if N < 2 || numel(w_abs) < 2
        G = NaN; return;
    end

    w = sort(w_abs(:));       % ascending
    m = mean(w);
    if m == 0
        G = 0; return;
    end

    ii = (1:N)';
    G  = (N / (N-1)) * sum((2*ii - N - 1) .* w) / (N^2 * m);
    G  = max(0, min(1, G));   % clamp to [0, 1] against numerical noise
end


function projections = calculate_pcca_projections(region_data, ri_name, rj_name, ...
                                                    is_partial, Z_full, ...
                                                    Wx, Wy, sig, mean_cv_rho, ...
                                                    lambda_reg)
% CALCULATE_PCCA_PROJECTIONS  Single-trial projections onto pCCA dimensions.
%
% Applies the full-data nuisance regression (using all N_samples to estimate
% beta_X, beta_Y) and then projects the residuals onto the mean canonical
% weight vectors Wx(:,c) and Wy(:,c) for each significant dimension c.
%
% The output structure mirrors calculate_canonical_projections in
% perform_session_cca.m, so all existing plotting code that iterates over
% .projections.components{} is immediately compatible.

    try
        sel_i  = region_data.regions.(ri_name).selected_neurons;
        sel_j  = region_data.regions.(rj_name).selected_neurons;

        Xi = region_data.regions.(ri_name).spike_data(:, sel_i, :);
        Xj = region_data.regions.(rj_name).spike_data(:, sel_j, :);

        [n_trials, n_i, T] = size(Xi);
        n_j = size(Xj, 2);

        % Reshape to (N_samples x n_neurons) and z-score, matching Step 1
        Xi_flat = reshape(permute(Xi, [2,3,1]), n_i, n_trials*T)';
        Xj_flat = reshape(permute(Xj, [2,3,1]), n_j, n_trials*T)';

        Xi_z = zscore(Xi_flat, 0, 1);
        Xj_z = zscore(Xj_flat, 0, 1);

        % Remove nuisance on full data for visualization
        if is_partial && ~isempty(Z_full)
            n_z     = size(Z_full, 2);
            ZtZ_reg = Z_full' * Z_full + lambda_reg * eye(n_z);

            beta_X = ZtZ_reg \ (Z_full' * Xi_z);
            beta_Y = ZtZ_reg \ (Z_full' * Xj_z);

            Xi_res = Xi_z - Z_full * beta_X;
            Xj_res = Xj_z - Z_full * beta_Y;
        else
            Xi_res = Xi_z;
            Xj_res = Xj_z;
        end

        projections = struct();
        projections.n_components = numel(sig);
        projections.time_axis    = linspace(-1.5, 3.0, T);
        projections.components   = cell(numel(sig), 1);

        for ic = 1:numel(sig)
            c = sig(ic);

            x_proj = Xi_res * Wx(:, c);    % (N_samples, 1)
            y_proj = Xj_res * Wy(:, c);

            % Reshape to (n_trials x T) — consistent with legacy pipeline
            x_trials = reshape(x_proj, T, n_trials)';
            y_trials = reshape(y_proj, T, n_trials)';

            projections.components{ic} = struct( ...
                'component_number', c, ...
                'R2',               mean_cv_rho(c), ...
                'region_i_mean',    mean(x_trials, 1), ...
                'region_j_mean',    mean(y_trials, 1), ...
                'region_i_std',     std(x_trials, 0, 1), ...
                'region_j_std',     std(y_trials, 0, 1), ...
                'region_i_trials',  x_trials, ...
                'region_j_trials',  y_trials);
        end

    catch ME
        fprintf('        Error in pCCA projections: %s\n', ME.message);
        projections = [];
    end
end