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
% 4. Cleanup: Removes both MDL and cell_metrics files after processing
%
% MATHEMATICAL FRAMEWORK:
% The continuous firing rate data F_mdl ∈ ℝ^{N_neurons × T} is segmented into
% K trials using t_approach alignment times τ_k:
%
%   F_trial(k) = F_mdl[:, τ_k - 75 : τ_k + 150]  for k = 1, ..., K
%
% Each trial window spans [-1.5s, 3.0s] relative to the behavioral event.
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
            
            % Check if results already exist
            if exist(cca_results_file, 'file')
                fprintf('  Found existing CCA results file\n');
                
                try
                    saved_results = load(cca_results_file);
                    
                    if isfield(saved_results, 'region_data')
                        region_data = saved_results.region_data;
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
                    
                    % Cleanup even on failure
                    cleanup_session_mdl_files(session_id, date_str, analysis_config.local_base_dir, false);
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
                    
                    cleanup_session_mdl_files(session_id, date_str, analysis_config.local_base_dir, false);
                    continue;
                end
            end
            
            %% Phase 4: PCA Analysis (if applicable)
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
                fprintf('  Continuing with CCA analysis...\n');
            end
            
            %% Phase 5: CCA Analysis
            fprintf('\n[Phase 5] Performing cross-regional CCA analysis...\n');
            
            try
                cca_results = perform_session_cca(region_data, session_name, analysis_config);
                
                if isempty(cca_results.pair_results)
                    fprintf('  Warning: No valid region pairs for CCA\n');
                else
                    fprintf('  CCA completed: %d region pairs analyzed\n', length(cca_results.pair_results));
                    
                    % Extract summary statistics
                    max_R2_values = cellfun(@(x) x.max_R2, cca_results.pair_results);
                    fprintf('  Maximum R²: %.3f\n', max(max_R2_values));
                end
                
            catch cca_error
                error_msg = sprintf('CCA analysis failed: %s', cca_error.message);
                fprintf('  %s\n', error_msg);
                session_stats.failed_sessions{end+1} = {session_name, error_msg};
                
                cleanup_session_mdl_files(session_id, date_str, analysis_config.local_base_dir, false);
                continue;
            end
            
            %% Phase 6: Save Results
            fprintf('\n[Phase 6] Saving analysis results...\n');
            
            % Construct comprehensive results structure
            analysis_results = struct();
            analysis_results.session_name = session_name;
            analysis_results.analysis_timestamp = datestr(now);
            analysis_results.pipeline_version = '3.0_with_PCA_using_MDL_and_tapproach';
            
            % Include PCA results from Phase 3.5
            analysis_results.pca_results = pca_results;
            
            % Include CCA results from Phase 4
            analysis_results.cca_results = cca_results;
            
            % Include region data for downstream analyses
            analysis_results.region_data = region_data;
            
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
            fprintf('\n[Phase 7] Cleaning up raw data files...\n');
            
            if strcmp(data_source, 'mdl_download')
                cleanup_session_mdl_files(session_id, date_str, analysis_config.local_base_dir, true);
            else
                fprintf('  Skipping cleanup (data loaded from cache)\n');
            end
            
            %% Session Complete
            session_time = toc(session_start_time);
            session_stats.processing_times(session_idx) = session_time;
            
            fprintf('\n✓ Session %s completed in %.1f seconds\n', session_name, session_time);
            
        catch session_error
            % Catch-all for unexpected errors
            error_msg = sprintf('Unexpected error: %s', session_error.message);
            fprintf('\n✗ Session %s failed: %s\n', session_name, error_msg);
            session_stats.failed_sessions{end+1} = {session_name, error_msg};
            
            % Attempt cleanup
            try
                cleanup_session_mdl_files(session_id, date_str, analysis_config.local_base_dir, false);
            catch
                % Ignore cleanup errors
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
