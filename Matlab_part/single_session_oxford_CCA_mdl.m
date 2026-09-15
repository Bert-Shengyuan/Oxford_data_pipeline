function single_session_oxford_CCA_mdl(session_list, server_config, analysis_config, t_approach, session_logger)
% SINGLE_SESSION_OXFORD_CCA_MDL - Enhanced CCA pipeline for MDL-format data
%
% This function implements the complete single-session CCA analysis pipeline for
% the updated Oxford dataset format using MDL (Model Data Layer) files instead
% of pre-segmented trial files.
%
% KEY CHANGES FROM ORIGINAL PIPELINE:
% 1. Data Acquisition: Downloads {session}.mdl.mat instead of {session}.cue_dlc_bar_off.trial.mat
% 2. Trial Segmentation: Uses t_approach timestamps to extract trial epochs from continuous data
% 3. Trial Selection: Filters for label == 'cued hit long' trials only
% 4. Cleanup: Raw MDL and cell_metrics files under proc/ are kept by default;
%    set analysis_config.delete_raw_data = true to delete them after a
%    session finishes (success or failure) instead.
%
% MATHEMATICAL FRAMEWORK:
% The continuous firing rate data F_mdl ∈ ℝ^{N_neurons × T} is segmented into
% K trials using t_approach alignment times τ_k:
%
%   F_trial(k) = F_mdl[:, τ_k - pre_bins(mode) : τ_k + post_bins(mode)]  for k = 1, ..., K
%
% Each trial window spans 3.0s total, split around the alignment event
% per analysis_config.alignment_mode (see segment_mdl_to_trials.m).
%
% INPUTS:
%   session_list    - Cell array of {animal_id, date_string} pairs
%   server_config   - Server connection parameters struct:
%                     .host, .username, .base_dir
%   analysis_config - CCA analysis parameters struct:
%                     .local_base_dir, .min_neurons_per_region, .target_neurons,
%                     .time_window, .n_components, .cv_folds, .significance_threshold
%   t_approach      - Table from get_tapproach.m containing behavioral timestamps
%                     Required columns: {animal_id, session_date, session_name, start_time, label}
%   session_logger  - (Optional) Enhanced logging system for comprehensive tracking
%
% OUTPUTS:
%   Results are saved to disk:
%   - CCA results: {local_base_dir}/session_CCA_results/{session}_CCA_results.mat
%   - PSTH data:   {local_base_dir}/session_PSTH_results/{session}_PSTH_data.mat
%
% EXAMPLE USAGE:
%   session_list = {{'yp020', '220401'}, {'yp021', '220402'}};
%   server_config.host = 'hpc-login-1.cubi.bihealth.org';
%   server_config.username = 'shca10_c';
%   server_config.base_dir = '/data/cephfs-2/unmirrored/groups/peng/YP_Oxford/';
%   analysis_config.local_base_dir = '/Users/user/Oxford_dataset/';
%   t_approach = load('t_approach.mat').t_approach;
%   single_session_oxford_CCA_mdl(session_list, server_config, analysis_config, t_approach);

    % Handle backward compatibility for existing code
    if nargin < 5
        session_logger = [];
        fprintf('Note: Running without enhanced logging system.\n');
    end
    
    % Validate t_approach input
    if ~istable(t_approach)
        error('t_approach must be a MATLAB table. Load using: t_approach = load(''t_approach.mat'').t_approach');
    end
    
    required_cols = {'start_time', 'label'};
    missing_cols = setdiff(required_cols, t_approach.Properties.VariableNames);
    if ~isempty(missing_cols)
        error('t_approach table missing required columns: %s', strjoin(missing_cols, ', '));
    end

    % By default, the raw MDL and cell-metrics files downloaded into
    % {local_base_dir}/proc/{session_id}/{session_name} are kept after
    % processing. Set analysis_config.delete_raw_data = true to opt into
    % deleting them (e.g. to reclaim disk space during large batch runs).
    delete_raw_data = isfield(analysis_config, 'delete_raw_data') && analysis_config.delete_raw_data;

    fprintf('=== Oxford Single-Session CCA Pipeline (MDL Format) ===\n');
    fprintf('t_approach table loaded: %d rows, %d columns\n', height(t_approach), width(t_approach));
    
    %% Initialize Analysis Infrastructure
    
    % Create output directories for organized result storage
    % cca_results_dir = fullfile(analysis_config.local_base_dir, 'session_CCA_results');
    % psth_results_dir = fullfile(analysis_config.local_base_dir, 'session_PSTH_results');
    % 
    % if ~exist(cca_results_dir, 'dir'), mkdir(cca_results_dir); end
    % if ~exist(psth_results_dir, 'dir'), mkdir(psth_results_dir); end
    % Create new analysis results directory (Task C requirement)
    
    analysis_results_dir = fullfile(analysis_config.local_base_dir, analysis_config.data_folder);
    if ~exist(analysis_results_dir, 'dir')
        mkdir(analysis_results_dir);
        fprintf('  Created new analysis results directory: %s\n', analysis_results_dir);
    end
    % Initialize comprehensive session statistics tracking
    session_stats = struct();
    session_stats.total_sessions = length(session_list);
    session_stats.successful_downloads = 0;
    session_stats.successful_analyses = 0;
    session_stats.failed_sessions = {};
    session_stats.processing_times = zeros(length(session_list), 1);
    
    fprintf('Processing %d sessions...\n\n', session_stats.total_sessions);
    
    %% Main Processing Loop
    
    for session_idx = 1:length(session_list)
        session_info = session_list{session_idx};
        session_id = session_info{1};
        date_str = session_info{2};
        session_name = sprintf('%s_%s', session_id, date_str);
        
        fprintf('\n╔══════════════════════════════════════════════════════════════╗\n');
        fprintf('║  Session %d/%d: %s\n', session_idx, length(session_list), session_name);
        fprintf('╚══════════════════════════════════════════════════════════════╝\n');
        
        session_start_time = tic;
        
        try
            %% Phase 1: Check for Existing Results or Download Data
            fprintf('\n[Phase 1] Checking for existing results...\n');

            cca_results_file = fullfile(analysis_results_dir, sprintf('%s_analysis_results.mat', session_name));
            region_data = [];
            data_source = '';
            existing_analysis_results = struct();

            % Check if results already exist
            if exist(cca_results_file, 'file')
                fprintf('  Found existing analysis results file\n');

                try
                    saved_results = load(cca_results_file);

                    if isfield(saved_results, 'region_data')
                        region_data = saved_results.region_data;
                        existing_analysis_results = saved_results;
                        data_source = 'cached_cca_results';
                        fprintf('  Successfully loaded region data from cache\n');
                        fprintf('  Valid regions: %s\n', strjoin(region_data.valid_regions, ', '));
                    else
                        fprintf('  Warning: Cached file missing region_data field\n');
                    end
                catch load_error
                    fprintf('  Error loading cached results: %s\n', load_error.message);
                end
            end
            
            % If no cached data, proceed with download
            if isempty(region_data)
                fprintf('  No valid cache found. Initiating MDL data download...\n');
                
                try
                    % Use the same verification approach as download_single_session
                    download_success = download_single_session_mdl(session_id, date_str, ...
                                                              analysis_config.local_base_dir, ...
                                                              server_config);
                    
                    if ~download_success
                        % Document download failure with specific diagnostic information
                        error_message = sprintf('Server connection failed for session %s_%s', session_id, date_str);
                        fprintf('Download failed for session %s. Skipping...\n', session_name);
                        
                        % Enhanced logging of download failures
                        if ~isempty(session_logger)
                            log_session_failure(session_logger, session_name, 'DOWNLOAD_FAILED', ...
                                'DATA_ACQUISITION', error_message);
                        end
                        
                        % Legacy tracking for backward compatibility
                        session_stats.failed_sessions{end+1} = {session_name, 'Download failed'};
                        continue;
                    end
                    
                    data_source = 'mdl_download';
                    session_stats.successful_downloads = session_stats.successful_downloads + 1;
                    
                    
                catch download_error
                    error_msg = sprintf('Download failed: %s', download_error.message);
                    fprintf('  %s\n', error_msg);
                    session_stats.failed_sessions{end+1} = {session_name, error_msg};
                    continue;
                end
            end
            
            %% Phase 2: Data Extraction and Trial Segmentation
            session_data = [];
            
            if strcmp(data_source, 'mdl_download')
                fprintf('\n[Phase 2] Extracting and segmenting MDL data...\n');
                
                try
                    % Extract session data using the new MDL extraction function
                    session_data = extract_session_data_mdl(session_id, date_str, ...
                                                           analysis_config, t_approach);
                    
                    if isempty(session_data)
                        error('Data extraction returned empty result');
                    end
                    
                    fprintf('  Extraction successful: %d trials, %d neurons\n', ...
                            session_data.n_trials, session_data.n_neurons);
                    
                catch extraction_error
                    error_msg = sprintf('Extraction failed: %s', extraction_error.message);
                    fprintf('  %s\n', error_msg);
                    session_stats.failed_sessions{end+1} = {session_name, error_msg};
                    
                    % Cleanup even on failure (only if opted in)
                    if delete_raw_data
                        cleanup_session_mdl_files(session_id, date_str, analysis_config.local_base_dir, false);
                    end
                    continue;
                end
            end
            
            %% Phase 3: Regional Analysis
            if isempty(region_data) && ~isempty(session_data)
                fprintf('\n[Phase 3] Organizing data by brain region...\n');
                
                try
                    region_data = perform_region_analysis(session_data, analysis_config);
                    
                    if isempty(region_data.valid_regions)
                        error('No regions meet minimum neuron threshold (%d)', ...
                              analysis_config.min_neurons_per_region);
                    end
                    
                    fprintf('  Valid regions: %d (%s)\n', length(region_data.valid_regions), ...
                            strjoin(region_data.valid_regions, ', '));
                    
                catch region_error
                    error_msg = sprintf('Regional analysis failed: %s', region_error.message);
                    fprintf('  %s\n', error_msg);
                    session_stats.failed_sessions{end+1} = {session_name, error_msg};

                    if delete_raw_data
                        cleanup_session_mdl_files(session_id, date_str, analysis_config.local_base_dir, false);
                    end
                    continue;
                end
            end
            
            %% Phase 4: PCA Analysis (if applicable)
            % When region_data was reused from cache, PCA results are also
            % reused rather than recomputed, so an existing cached file only
            % gains the newly requested kernel's result field.
            if strcmp(data_source, 'cached_cca_results') && isfield(existing_analysis_results, 'pca_results')
                fprintf('\n[Phase 4] Reusing cached PCA results...\n');
                pca_results = existing_analysis_results.pca_results;
            else
                fprintf('\n[Phase 4] Performing PCA analysis...\n');

                pca_results = struct();
                pca_results.session_name = session_name;
                pca_results.analysis_timestamp = datestr(now);
                pca_results.config = analysis_config;

                try
                    % Perform PCA for each valid region
                    for region_idx = 1:length(region_data.valid_regions)
                        region_name = region_data.valid_regions{region_idx};
                        selected_neurons = region_data.regions.(region_name).selected_neurons;

                        spike_data = region_data.regions.(region_name).spike_data(:, selected_neurons, :);


                        fprintf('  Performing PCA for region: %s\n', region_name);
                        fprintf('    Original dimensions: %d neurons × %d trials × %d timepoints\n', ...
                               size(spike_data, 2), size(spike_data, 1), size(spike_data, 3));

                        % Perform PCA with cross-validation following CCA methodology
                        region_pca = perform_region_pca(spike_data, analysis_config);

                        if ~isempty(region_pca)
                            pca_results.(region_name) = region_pca;
                        else
                            fprintf('    Warning: PCA failed for region %s\n', region_name);
                        end
                    end

                    fprintf('  PCA analysis completed for %d regions\n', ...
                           length(fieldnames(pca_results)) - 3); % Subtract metadata fields

                catch pca_error
                    fprintf('  Warning: PCA analysis encountered error: %s\n', pca_error.message);
                    fprintf('  Continuing with kernel analysis...\n');
                end
            end

            %% Phase 5: Cross-Regional Kernel Analysis (CCA / pCCA / tkCCA / none)
            kernel_type = 'cca';
            if isfield(analysis_config, 'kernel_type') && ~isempty(analysis_config.kernel_type)
                kernel_type = lower(analysis_config.kernel_type);
            end

            switch kernel_type
                case 'cca'
                    kernel_field = 'cca_result';
                    kernel_fn = @perform_session_cca;
                case 'pcca'
                    kernel_field = 'pcca_result';
                    kernel_fn = @perform_session_pcca;
                case 'tkcca'
                    kernel_field = 'tkcca_result';
                    kernel_fn = @perform_session_tkcca;
                case 'none'
                    kernel_field = '';
                    kernel_fn = [];
                otherwise
                    error('Unknown kernel_type "%s" (expected cca, pcca, tkcca, or none)', kernel_type);
            end

            kernel_result = [];

            if isempty(kernel_fn)
                fprintf('\n[Phase 5] Kernel step skipped (kernel_type = ''none'') — returning aligned spike data only.\n');
            else
                fprintf('\n[Phase 5] Performing cross-regional %s analysis...\n', upper(kernel_type));

                try
                    kernel_result = kernel_fn(region_data, session_name, analysis_config);

                    if isempty(kernel_result.pair_results)
                        fprintf('  Warning: No valid region pairs for %s\n', upper(kernel_type));
                    else
                        fprintf('  %s completed: %d region pairs analyzed\n', ...
                                upper(kernel_type), length(kernel_result.pair_results));

                        % Extract summary statistics
                        max_R2_values = cellfun(@(x) x.max_R2, kernel_result.pair_results);
                        fprintf('  Maximum R²: %.3f\n', max(max_R2_values));
                    end

                catch kernel_error
                    error_msg = sprintf('%s analysis failed: %s', upper(kernel_type), kernel_error.message);
                    fprintf('  %s\n', error_msg);
                    session_stats.failed_sessions{end+1} = {session_name, error_msg};

                    if delete_raw_data
                        cleanup_session_mdl_files(session_id, date_str, analysis_config.local_base_dir, false);
                    end
                    continue;
                end
            end

            %% Phase 6: Save Results
            fprintf('\n[Phase 6] Saving analysis results...\n');

            % Start from any previously cached results so that result fields
            % from other kernels (e.g. a prior pcca_result) are preserved,
            % and only the newly computed kernel field is added/overwritten.
            analysis_results = existing_analysis_results;
            analysis_results.session_name = session_name;
            analysis_results.analysis_timestamp = datestr(now);
            analysis_results.pipeline_version = '4.0_configurable_kernel_alignment_subregion';

            % Include PCA results from Phase 4
            analysis_results.pca_results = pca_results;

            % Include region data (now with subregion labels) for downstream analyses
            analysis_results.region_data = region_data;

            % Append the newly computed kernel result, if any
            if ~isempty(kernel_field)
                analysis_results.(kernel_field) = kernel_result;
            end

            % Save to new comprehensive analysis results file (Task C)
            analysis_results_file = fullfile(analysis_results_dir, ...
                sprintf('%s_analysis_results.mat', session_name));
            
            try
                save(analysis_results_file, '-struct', 'analysis_results', '-v7.3');
                fprintf('  Comprehensive analysis results saved: %s\n', analysis_results_file);
                
                % Verify file integrity
                file_info = dir(analysis_results_file);
                fprintf('  File size: %.2f MB\n', file_info.bytes / 1024 / 1024);
            catch save_error
                fprintf('  Warning: Error saving analysis results: %s\n', save_error.message);
            end
            
            %% Phase 7: Cleanup Raw Data
            % fprintf('\n[Phase 7] Cleaning up raw data files...\n');
            % 
            % if strcmp(data_source, 'mdl_download')
            %     cleanup_session_mdl_files(session_id, date_str, analysis_config.local_base_dir, true);
            % else
            %     fprintf('  Skipping cleanup (data loaded from cache)\n');
            % end
            
            %% Session Complete
            session_time = toc(session_start_time);
            session_stats.processing_times(session_idx) = session_time;
            
            fprintf('\n✓ Session %s completed in %.1f seconds\n', session_name, session_time);
            
        catch session_error
            % Catch-all for unexpected errors
            error_msg = sprintf('Unexpected error: %s', session_error.message);
            fprintf('\n✗ Session %s failed: %s\n', session_name, error_msg);
            session_stats.failed_sessions{end+1} = {session_name, error_msg};
            
            % Attempt cleanup (only if opted in)
            if delete_raw_data
                try
                    cleanup_session_mdl_files(session_id, date_str, analysis_config.local_base_dir, false);
                catch
                    % Ignore cleanup errors
                end
            end
        end
    end
    
    %% Pipeline Summary
    fprintf('\n');
    fprintf('╔══════════════════════════════════════════════════════════════╗\n');
    fprintf('║                    PIPELINE SUMMARY                          ║\n');
    fprintf('╚══════════════════════════════════════════════════════════════╝\n');
    fprintf('Total sessions: %d\n', session_stats.total_sessions);
    fprintf('Successful downloads: %d\n', session_stats.successful_downloads);
    fprintf('Successful analyses: %d\n', session_stats.successful_analyses);
    fprintf('Failed sessions: %d\n', length(session_stats.failed_sessions));
    fprintf('Success rate: %.1f%%\n', 100 * session_stats.successful_analyses / session_stats.total_sessions);
    
    if ~isempty(session_stats.failed_sessions)
        fprintf('\nFailed sessions:\n');
        for i = 1:length(session_stats.failed_sessions)
            failed = session_stats.failed_sessions{i};
            fprintf('  - %s: %s\n', failed{1}, failed{2});
        end
    end
    
    valid_times = session_stats.processing_times(session_stats.processing_times > 0);
    if ~isempty(valid_times)
        fprintf('\nProcessing time statistics:\n');
        fprintf('  Mean: %.1f seconds\n', mean(valid_times));
        fprintf('  Total: %.1f minutes\n', sum(valid_times) / 60);
    end
    
    fprintf('\nResults location:\n');
    fprintf('  Analysis results: %s\n', analysis_results_dir);
    % fprintf('  PSTH data: %s\n', psth_results_dir);
end
