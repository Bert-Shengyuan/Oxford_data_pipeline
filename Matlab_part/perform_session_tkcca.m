function cca_results= perform_session_tkcca(region_data, session_name, config)
% ANALYZE_REGION_PAIR_TKCCA  Temporal kernel CCA between two brain regions.
%
% Implements the time-delay embedding form of tkCCA (Biessmann et al. 2010)
% adapted to trial-structured Neuropixels data. Standard CCA is generalised by
% allowing one source to enter through a spatio-temporal filter w_x(\tau):
%
%   max_{w_x(\tau), w_y}  Corr( \sum_\tau X(t-\tau) w_x(\tau),  Y(t) w_y ).
%
% Via the embedding  \tilde X(t) = [ X(t-\tau_1) | ... | X(t-\tau_K) ]
% this reduces to ordinary CCA between \tilde X and Y in the embedded space.
% The per-lag filter w_x(\tau_k) is then read off as the k-th column-block of
% the embedded variate \tilde w_x.
%
% Sign / lag convention:
%   tau > 0  =>  X(t-tau) leads Y(t)  =>  region_i leads region_j by tau bins.
%   tau < 0  =>  X(t-tau) = X(t+|tau|) lags Y(t)  =>  region_j leads region_i.
%
% Adaptations beyond the paper:
%   * Embedding is performed WITHIN each trial. Edge timepoints where any
%     required lag t-\tau leaves [1, T] are dropped.
%   * Cross-validation is split at the TRIAL level. Sample-level k-fold would
%     leak lagged neurons between train and test the moment \tilde X is used.
%
% Required config fields beyond the legacy CCA call:
%   config.tkcca_lags             integer vector of lags in bins, e.g. -5:5.
%                                 Default: -5:5.
%   config.tkcca_max_components   cap on number of canonical components
%                                 retained (the embedded dim n_i*K can be
%                                 large). Default: 20.
%
% New fields appended to pair_results (in addition to the legacy interface):
%   .wx_temporal              (n_i, K, n_sig)  -- w_x(\tau_k)
%   .wy_stationary            (n_j, n_sig)     -- w_y
%   .canonical_correlogram    (K,   n_sig)     -- \rho(\tau_k)
%   .tkcca_lags_bins          (1, K)
%   .tkcca_lags_seconds       (1, K)           -- bin_width = 4.5 / T
    fprintf('  Executing canonical correlation analysis...\n');
    
    cca_results = struct();
    cca_results.session_name = session_name;
    cca_results.analysis_timestamp = datestr(now);
    cca_results.config = config;
    cca_results.region_pairs = {};
    cca_results.pair_results = {};
    
    n_pairs = size(region_data.region_pairs, 1);
    
    if n_pairs == 0
        fprintf('  No valid region pairs for CCA analysis\n');
        return;
    end
    
    % Resample all regions once before pairwise analysis
    fprintf('  Resampling neurons across all regions...\n');
    sampled_region_data = region_data;
    sampled_region_data.sampled_neurons = struct();


    for pair_idx = 1:n_pairs
        region_i_idx = region_data.region_pairs(pair_idx, 1);
        region_j_idx = region_data.region_pairs(pair_idx, 2);
        
        region_i_name = region_data.valid_regions{region_i_idx};
        region_j_name = region_data.valid_regions{region_j_idx};
        try
            % ------ configuration ------------------------------------------------
            if ~isfield(config, 'tkcca_lags') || isempty(config.tkcca_lags)
                lag_bins = -5:5;
            else
                lag_bins = config.tkcca_lags(:)';
            end
            if ~isfield(config, 'tkcca_max_components') || isempty(config.tkcca_max_components)
                max_components = 20;
            else
                max_components = config.tkcca_max_components;
            end
            K = numel(lag_bins);
            % ------ extract pre-sampled spike data ------------------------------
            selected_neurons_i = region_data.regions.(region_i_name).selected_neurons;
            selected_neurons_j = region_data.regions.(region_j_name).selected_neurons;
            Xi = region_data.regions.(region_i_name).spike_data(:, selected_neurons_i, :);
            Xj = region_data.regions.(region_j_name).spike_data(:, selected_neurons_j, :);
    
            if isempty(Xi) || isempty(Xj)
                fprintf('      tkCCA: insufficient neurons in %s or %s\n', region_i_name, region_j_name);
                pair_results = []; return;
            end
    
            target_neurons = region_data.regions.(region_i_name).target_neurons;
            orig_i = region_data.regions.(region_i_name).original_neurons;
            orig_j = region_data.regions.(region_j_name).original_neurons;
    
            fprintf('      tkCCA pre-sampled: %s=%d, %s=%d (from %d, %d); K=%d lags\n', ...
                    region_i_name, target_neurons, region_j_name, target_neurons, ...
                    orig_i, orig_j, K);
    
            [n_trials, n_i, T] = size(Xi);
            n_j = size(Xj, 2);
    
            % ------ run tkCCA core ----------------------------------------------
            core_out = tkcca_core(Xi, Xj, lag_bins, max_components, config);
            if isempty(core_out)
                pair_results = []; return;
            end
    
            % ------ assemble pair_results (legacy + tkCCA fields) ----------------
            bin_width_s = 4.5 / T;
    
            % cv_results substruct mirrors legacy interface so downstream
            % plotting / aggregation code that reads .cv_results still works.
            cv_results = struct( ...
                'mean_cv_R2',   core_out.mean_cv_rho, ...
                'std_cv_R2',    core_out.std_cv_rho, ...
                'cv_R2',        core_out.cv_rho, ...
                'mean_A_matrix',core_out.W_tilde_x, ...   % (n_i*K, n_components)
                'mean_B_matrix',core_out.W_y, ...         % (n_j,   n_components)
                'A_matrices',   core_out.W_tilde_x_cv, ...
                'B_matrices',   core_out.W_y_cv);
    
            sig = core_out.significant_components;
    
            pair_results = struct();
            pair_results.region_i               = region_i_name;
            pair_results.region_j               = region_j_name;
            pair_results.target_neurons         = target_neurons;
            pair_results.selected_neurons_i     = selected_neurons_i;
            pair_results.selected_neurons_j     = selected_neurons_j;
            pair_results.original_neuron_counts = [orig_i, orig_j];
            pair_results.cv_results             = cv_results;
            pair_results.significant_components = sig;
            pair_results.max_R2                 = max(core_out.mean_cv_rho);
            pair_results.mean_R2                = mean(core_out.mean_cv_rho, 'omitnan');
            pair_results.mean_A_matrix          = core_out.W_tilde_x;
            pair_results.mean_B_matrix          = core_out.W_y;
    
            % --- tkCCA-specific outputs ---
            % wx_temporal has shape (n_i, K, n_sig): the spatio-temporal filter
            % for each significant component.
            wx_full = reshape(core_out.W_tilde_x, n_i, K, []);
            pair_results.wx_temporal           = wx_full(:, :, sig);
            pair_results.wy_stationary         = core_out.W_y(:, sig);
            pair_results.canonical_correlogram = core_out.correlogram(:, 1:numel(sig));
            pair_results.tkcca_lags_bins       = lag_bins;
            pair_results.tkcca_lags_seconds    = lag_bins * bin_width_s;
    
            % Projections for visualisation (analogue of the legacy projections):
            if ~isempty(sig)
                pair_results.projections = calculate_tkcca_projections( ...
                    Xi, Xj, core_out.W_tilde_x, core_out.W_y, ...
                    lag_bins, sig, core_out.mean_cv_rho);
            end
    
            fprintf('      tkCCA done: max rho = %.3f, mean rho = %.3f, %d sig components\n', ...
                    pair_results.max_R2, pair_results.mean_R2, numel(sig));

        catch ME
            fprintf('      Error in tkCCA analysis (%s - %s): %s\n', ...
                    region_i_name, region_j_name, ME.message);
            pair_results = [];
        end
        if ~isempty(pair_results)
            cca_results.region_pairs{end+1} = sprintf('%s_%s', region_i_name, region_j_name);
            cca_results.pair_results{end+1} = pair_results;

        else
            fprintf('   CCA failed for this pair\n');
        end
    end
end

% =========================================================================
%  CORE: embedding + CV CCA + correlogram
% =========================================================================
function out = tkcca_core(Xi, Xj, lag_bins, max_components, config)
% TKCCA_CORE  Run tkCCA on a (sub-)block of trial data.
%
% Inputs:
%   Xi             (n_trials, n_i, T)
%   Xj             (n_trials, n_j, T)
%   lag_bins       integer vector of lags in bins
%   max_components cap on # canonical components
%   config         must contain .cv_folds and .significance_threshold
%
% Output struct fields:
%   .W_tilde_x       (n_i*K, n_comp)  embedded variate, averaged over folds
%   .W_y             (n_j,   n_comp)  stationary variate, averaged over folds
%   .W_tilde_x_cv    per-fold tensor
%   .W_y_cv          per-fold tensor
%   .cv_rho          (n_folds, n_comp) per-fold held-out canonical corrs
%   .mean_cv_rho     (1, n_comp)
%   .std_cv_rho      (1, n_comp)
%   .correlogram     (K, n_sig) rho(tau_k) for each significant component
%   .significant_components

    [n_trials, n_i, T] = size(Xi);
    n_j = size(Xj, 2);
    K   = numel(lag_bins);

    % --- valid centre timepoints inside each trial ------------------------
    t_lo = 1 + max(0,  max(lag_bins));
    t_hi = T + min(0,  min(lag_bins));
    if t_hi < t_lo
        fprintf('        tkcca_core: lag span (%d) > trial length (%d)\n', ...
                max(lag_bins) - min(lag_bins), T);
        out = []; return;
    end
    valid_t = t_lo:t_hi;
    T_valid = numel(valid_t);

    % --- build embedded design matrix \tilde X and target Y ---------------
    % \tilde X has shape (n_trials * T_valid,  n_i * K)
    % Y_target has shape (n_trials * T_valid,  n_j)
    X_tilde  = zeros(n_trials * T_valid, n_i * K);
    Y_target = zeros(n_trials * T_valid, n_j);
    for r = 1:n_trials
        Xi_r = squeeze(Xi(r, :, :));   % (n_i, T)
        Xj_r = squeeze(Xj(r, :, :));   % (n_j, T)
        row0 = (r-1) * T_valid;
        for tt = 1:T_valid
            t = valid_t(tt);
            row = row0 + tt;
            for k = 1:K
                c_lo = (k-1) * n_i + 1;
                c_hi =  k    * n_i;
                X_tilde(row, c_lo:c_hi) = Xi_r(:, t - lag_bins(k))';
            end
            Y_target(row, :) = Xj_r(:, t)';
        end
    end

    % Mean-centre (CCA requires zero-mean inputs).
    X_tilde  = X_tilde  - mean(X_tilde,  1);
    Y_target = Y_target - mean(Y_target, 1);

    % --- trial-level k-fold CCA ------------------------------------------
    n_components = min([n_i * K, n_j, max_components]);
    n_folds = min(config.cv_folds, n_trials);
    if n_folds < 2
        fprintf('tkcca_core: n_trials=%d < 2, CV impossible\n', n_trials);
        out = []; return;
    end

    % Deterministic but trial-permuted fold assignment.
    rng(12345, 'twister');
    trial_perm      = randperm(n_trials);           % e.g. [3,4,2,5,6,1]
 % [1,2,3,4,5,6]
    fold_of_trial   = mod(0:n_trials-1, n_folds) + 1;

    cv_rho        = nan(n_folds, n_components);
    W_tilde_x_cv  = zeros(n_i * K, n_components, n_folds);
    W_y_cv        = zeros(n_j,     n_components, n_folds);

    fprintf('        tkcca CV: %d folds, %d components (embedded dim = %d)\n', ...
            n_folds, n_components, n_i * K);
    try
        for f = 1:n_folds
            %test_trials     = trial_perm(randperm(round(n_trials*0.25)));
            test_trials     = trial_perm((fold_of_trial == f));
            train_trials    = trial_perm(~(fold_of_trial == f));
    
            train_rows = trial_rows(train_trials, T_valid);
            test_rows  = trial_rows(test_trials,  T_valid);
    
            Xtr = X_tilde(train_rows,  :);
            Ytr = Y_target(train_rows, :);
            Xte = X_tilde(test_rows,   :);
            Yte = Y_target(test_rows,  :);
    
            [Wx_f, Wy_f] = local_robust_cca(Xtr, Ytr, n_components);
            if isempty(Wx_f)
                fprintf('        fold %d: CCA failed\n', f); continue;
            end
    
            W_tilde_x_cv(:, :, f) = Wx_f;
            W_y_cv(:, :, f)       = Wy_f;
    
            for c = 1:n_components
                cv_rho(f, c) = corr(Xte * Wx_f(:, c), Yte * Wy_f(:, c));
            end
        end
    catch ME
        fprintf(ME.message);
    end

    mean_cv_rho = mean(cv_rho, 1, 'omitnan');
    std_cv_rho  = std(cv_rho,  0, 1, 'omitnan');
    W_tilde_x   = mean(W_tilde_x_cv, 3);
    W_y         = mean(W_y_cv,       3);

    % --- significance: same percentile rule as the legacy pipeline -------
    sig_components = find(mean_cv_rho >= ...
                          prctile(mean_cv_rho, config.significance_threshold));

    % --- canonical correlogram \rho(\tau_k) per significant component ----
    % By Eq. 14:  rho(tau_k) = Corr( X(tau_k) w_x(tau_k),  Y w_y ).
    % X(tau_k) is exactly the k-th column-block of X_tilde on the valid
    % centre timepoints, so this is one dot product per (k, c).
    correlogram = nan(K, numel(sig_components));
    wx_full = reshape(W_tilde_x, n_i, K, n_components);
    for ic = 1:numel(sig_components)
        c = sig_components(ic);
        proj_y = Y_target * W_y(:, c);
        for k = 1:K
            c_lo = (k-1) * n_i + 1;
            c_hi =  k    * n_i;
            proj_xk = X_tilde(:, c_lo:c_hi) * wx_full(:, k, c);
            correlogram(k, ic) = corr(abs(proj_xk), abs(proj_y));
        end
    end

    out = struct( ...
        'W_tilde_x',              W_tilde_x, ...
        'W_y',                    W_y, ...
        'W_tilde_x_cv',           W_tilde_x_cv, ...
        'W_y_cv',                 W_y_cv, ...
        'cv_rho',                 cv_rho, ...
        'mean_cv_rho',            mean_cv_rho, ...
        'std_cv_rho',             std_cv_rho, ...
        'correlogram',            correlogram, ...
        'significant_components', sig_components);
end


% =========================================================================
%  Helpers
% =========================================================================
function rows = trial_rows(trial_subset, T_valid)
% Convert trial indices to row indices in the (n_trials*T_valid)-row matrix.
    rows = zeros(numel(trial_subset) * T_valid, 1);
    for ii = 1:numel(trial_subset)
        r = trial_subset(ii);
        rows((ii-1)*T_valid + 1 : ii*T_valid) = (r-1)*T_valid + 1 : r*T_valid;
    end
end


function [A, B] = local_robust_cca(X, Y, n_components)
% Ridge-augmented canoncorr identical in logic to the existing robust_cca
% in perform_session_cca.m, duplicated locally so this file is self-contained.
    try
        rX = rank(X); rY = rank(Y);
        if rX >= n_components && rY >= n_components
            [A, B, ~] = canoncorr(X, Y);
        else
            lambda = 0.01;
            for attempt = 1:5
                X_reg = [X; sqrt(lambda) * eye(n_components, size(X, 2))];
                Y_reg = [Y; sqrt(lambda) * eye(n_components, size(Y, 2))];
                try
                    [A, B, ~] = canoncorr(X_reg, Y_reg);
                    if size(A, 2) >= n_components, break; end
                catch
                    lambda = lambda * 10;
                end
            end
        end
        if size(A, 2) < n_components || size(B, 2) < n_components
            kept = min(size(A, 2), size(B, 2));
            if kept > 0
                A = A(:, 1:kept); B = B(:, 1:kept);
                if kept < n_components
                    A = [A, zeros(size(A, 1), n_components - kept)];
                    B = [B, zeros(size(B, 1), n_components - kept)];
                end
            else
                A = []; B = [];
            end
        else
            A = A(:, 1:n_components);
            B = B(:, 1:n_components);
        end
    catch ME
        fprintf('          local_robust_cca error: %s\n', ME.message);
        A = []; B = [];
    end
end


function projections = calculate_tkcca_projections(Xi, Xj, W_tilde_x, W_y, ...
                                                    lag_bins, sig_components, mean_cv_rho)
% Single-trial projections analogous to calculate_canonical_projections.
% For region_i this is the temporal convolution
%   z_i(r, t) = \sum_\tau  w_x(\tau)^T  X_i(r, :, t-\tau)
% For region_j this is the stationary projection X_j(:, :, t) * w_y.
% Both regions are z-scored across all (trial * time) samples first, matching
% the existing pipeline convention.

    [n_trials, n_i, T] = size(Xi);
    n_j = size(Xj, 2);
    K   = numel(lag_bins);
    n_sig = numel(sig_components);

    Xi_flat = reshape(permute(Xi, [2 3 1]), n_i, n_trials * T)';
    Xj_flat = reshape(permute(Xj, [2 3 1]), n_j, n_trials * T)';
    Xi_flat = zscore(Xi_flat, 0, 1);
    Xj_flat = zscore(Xj_flat, 0, 1);


    Xi_z = permute(reshape(Xi_flat', n_i, T, n_trials), [3 1 2]);  % (n_trials, n_i, T)
    Xj_z = permute(reshape(Xj_flat', n_j, T, n_trials), [3 1 2]);  % (n_trials, n_j, T)

    wx_full = reshape(W_tilde_x, n_i, K, []);

    projections = struct();
    projections.n_components = n_sig;
    projections.time_axis    = linspace(-1.5, 3.0, T);
    projections.lag_bins     = lag_bins;
    projections.components   = cell(n_sig, 1);

    for ic = 1:n_sig
        c = sig_components(ic);

        % Region i: temporal convolution across all lags.
        z_i = zeros(n_trials, T);
        for k = 1:K
            tau = lag_bins(k);
            wk_all  = squeeze(wx_full(:, k, :));   
            wk = wk_all (:,c);% (n_i, 1)
            Xi_shift = zeros(n_trials, n_i, T);
            src = (1:T) - tau;
            v   = find(src >= 1 & src <= T);
            Xi_shift(:, :, v) = Xi_z(:, :, src(v));
            % Project: contribution_k(r, t) = Xi_shift(r, :, t) * wk
            contribution_k = squeeze(sum( ...
                bsxfun(@times, Xi_shift, reshape(wk, 1, n_i, 1)), 2));
            z_i = z_i + contribution_k;
        end

        % Region j: stationary projection.
        z_j = squeeze(sum(bsxfun(@times, Xj_z, reshape(W_y(:, c), 1, n_j, 1)), 2));

        projections.components{ic} = struct( ...
            'component_number', c, ...
            'R2',               mean_cv_rho(c), ...
            'region_i_mean',    mean(z_i, 1), ...
            'region_j_mean',    mean(z_j, 1), ...
            'region_i_std',     std(z_i, 0, 1), ...
            'region_j_std',     std(z_j, 0, 1), ...
            'region_i_trials',  z_i, ...
            'region_j_trials',  z_j);
    end
end


