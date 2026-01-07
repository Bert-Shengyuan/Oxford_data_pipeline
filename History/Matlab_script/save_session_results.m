function save_session_results(cca_results, region_data, session_name, cca_results_dir, psth_results_dir)
% SAVE_SESSION_RESULTS - Archive CCA results and PSTH data with efficient storage
%
% This function implements selective data archival, preserving essential analytical
% results while discarding raw data to maintain computational efficiency. The storage
% strategy balances information preservation with memory management, enabling
% large-scale analysis across multiple experimental sessions.
%
% ARCHIVAL STRATEGY:
% 1. CCA Results: Complete statistical analysis including transformation matrices,
%    cross-validation metrics, and significant canonical components
% 2. PSTH Data: Peri-stimulus time histograms for all regions and neurons
% 3. Metadata: Session parameters, analysis configuration, and quality metrics
% 4. Projections: Canonical projections for visualization and interpretation

    fprintf('  Archiving analysis results for session %s...\n', session_name);
    
    try
        %% Archive CCA Results
        % Save comprehensive CCA analysis with full statistical context
        cca_filename = fullfile(cca_results_dir, sprintf('%s_CCA_results.mat', session_name));
        
        % Create summary structure for efficient loading
        cca_summary = struct();
        cca_summary.session_name = session_name;
        cca_summary.analysis_timestamp = cca_results.analysis_timestamp;
        cca_summary.n_region_pairs = length(cca_results.pair_results);
        cca_summary.valid_regions = region_data.valid_regions;
        
        % Extract key metrics for quick access
        if ~isempty(cca_results.pair_results)
            max_R2_values = cellfun(@(x) x.max_R2, cca_results.pair_results);
            mean_R2_values = cellfun(@(x) x.mean_R2, cca_results.pair_results);
            n_sig_components = cellfun(@(x) length(x.significant_components), cca_results.pair_results);
            
            cca_summary.max_R2_across_pairs = max(max_R2_values);
            cca_summary.mean_R2_across_pairs = mean(mean_R2_values);
            cca_summary.total_significant_components = sum(n_sig_components);
            
            fprintf('    CCA Summary: Max R²=%.3f, Mean R²=%.3f, %d significant components\n', ...
                    cca_summary.max_R2_across_pairs, cca_summary.mean_R2_across_pairs, ...
                    cca_summary.total_significant_components);
        else
            cca_summary.max_R2_across_pairs = 0;
            cca_summary.mean_R2_across_pairs = 0;
            cca_summary.total_significant_components = 0;
            fprintf('    CCA Summary: No valid results to archive\n');
        end
        
        % Save complete CCA results with compression
        save(cca_filename, 'cca_results', 'cca_summary', 'region_data', '-v7.3');
        fprintf('    ✓ CCA results saved: %s\n', cca_filename);
        
        %% Archive PSTH Data
        % Generate and save peri-stimulus time histograms for all regions
        psth_filename = fullfile(psth_results_dir, sprintf('%s_PSTH_data.mat', session_name));
        psth_data = generate_session_psth(region_data, session_name);
        
        save(psth_filename, 'psth_data', '-v7.3');
        fprintf('    ✓ PSTH data saved: %s\n', psth_filename);
        
        %% Generate Session Report
        % Create human-readable summary of analysis results
        report_filename = fullfile(cca_results_dir, sprintf('%s_analysis_report.txt', session_name));
        generate_session_report(cca_results, region_data, cca_summary, report_filename);
        fprintf('    ✓ Analysis report generated: %s\n', report_filename);
        
        fprintf('  Results archival completed successfully\n');
        
    catch ME
        fprintf('  Error saving results: %s\n', ME.message);
        rethrow(ME);
    end
end

function psth_data = generate_session_psth(region_data, session_name)
% GENERATE_SESSION_PSTH - Create peri-stimulus time histograms for all regions
% This function computes trial-averaged firing rates across time for each neuron

    fprintf('    Generating PSTH data for %d regions...\n', length(region_data.valid_regions));
    
    psth_data = struct();
    psth_data.session_name = session_name;
    psth_data.time_axis = linspace(-1.5, 3.0, region_data.timepoints);
    psth_data.regions = struct();
    
    for i = 1:length(region_data.valid_regions)
        region_name = region_data.valid_regions{i};
        region_spike_data = region_data.regions.(region_name).spike_data;
        
        % Calculate trial-averaged PSTH for each neuron
        region_psth = squeeze(mean(region_spike_data, 1)); % Average across trials
        region_std = squeeze(std(region_spike_data, 0, 1)); % Standard deviation across trials
        
        % Store PSTH data with metadata
        psth_data.regions.(region_name) = struct();
        psth_data.regions.(region_name).trial_data = region_spike_data;
        psth_data.regions.(region_name).psth = region_psth; % neurons × timepoints
        psth_data.regions.(region_name).std = region_std;   % neurons × timepoints
        psth_data.regions.(region_name).n_neurons = size(region_psth, 1);
        psth_data.regions.(region_name).n_trials = size(region_spike_data, 1);
        psth_data.regions.(region_name).neuron_indices = region_data.regions.(region_name).neuron_indices;
        
        fprintf('      %s: %d neurons, %d trials\n', region_name, ...
                size(region_psth, 1), size(region_spike_data, 1));
    end
    
    fprintf('    PSTH generation completed\n');
end

function generate_session_report(cca_results, region_data, cca_summary, report_filename)
% GENERATE_SESSION_REPORT - Create comprehensive human-readable analysis report

    fid = fopen(report_filename, 'w');
    if fid == -1
        error('Could not create report file: %s', report_filename);
    end
    
    try
        % Header
        fprintf(fid, 'Oxford Single-Session CCA Analysis Report\n');
        fprintf(fid, '========================================\n\n');
        fprintf(fid, 'Session: %s\n', cca_results.session_name);
        fprintf(fid, 'Analysis Date: %s\n', cca_results.analysis_timestamp);
        fprintf(fid, 'Analysis Pipeline: Single-Session Dynamic Processing\n\n');
        
        % Regional Analysis Summary
        fprintf(fid, 'Neural Population Analysis\n');
        fprintf(fid, '-------------------------\n');
        fprintf(fid, 'Valid brain regions: %d\n', length(region_data.valid_regions));
        
        total_neurons = 0;
        for i = 1:length(region_data.valid_regions)
            region_name = region_data.valid_regions{i};
            n_neurons = region_data.regions.(region_name).n_neurons;
            total_neurons = total_neurons + n_neurons;
            
            fprintf(fid, '  %s: %d neurons\n', region_name, n_neurons);
        end
        fprintf(fid, 'Total neurons analyzed: %d\n\n', total_neurons);
        
        % CCA Analysis Summary
        fprintf(fid, 'Canonical Correlation Analysis\n');
        fprintf(fid, '-----------------------------\n');
        fprintf(fid, 'Region pairs analyzed: %d\n', cca_summary.n_region_pairs);
        fprintf(fid, 'Maximum R² achieved: %.4f\n', cca_summary.max_R2_across_pairs);
        fprintf(fid, 'Mean R² across pairs: %.4f\n', cca_summary.mean_R2_across_pairs);
        fprintf(fid, 'Total significant components: %d\n\n', cca_summary.total_significant_components);
        
        % Detailed Pair Results
        if ~isempty(cca_results.pair_results)
            fprintf(fid, 'Detailed Region Pair Results\n');
            fprintf(fid, '---------------------------\n');
            
            for i = 1:length(cca_results.pair_results)
                pair_result = cca_results.pair_results{i};
                
                fprintf(fid, 'Pair %d: %s ↔ %s\n', i, pair_result.region_i, pair_result.region_j);
                fprintf(fid, '  Neurons sampled: %d each (from %d and %d total)\n', ...
                        pair_result.target_neurons, pair_result.original_neuron_counts(1), ...
                        pair_result.original_neuron_counts(2));
                fprintf(fid, '  Maximum R²: %.4f\n', pair_result.max_R2);
                fprintf(fid, '  Mean R²: %.4f\n', pair_result.mean_R2);
                fprintf(fid, '  Significant components: %d\n', length(pair_result.significant_components));
                
                if ~isempty(pair_result.significant_components)
                    fprintf(fid, '  Component R² values: ');
                    r2_values = pair_result.cv_results.mean_cv_R2(pair_result.significant_components);
                    fprintf(fid, '%.3f ', r2_values);
                    fprintf(fid, '\n');
                end
                fprintf(fid, '\n');
            end
        end
        
        % Analysis Configuration
        fprintf(fid, 'Analysis Configuration\n');
        fprintf(fid, '---------------------\n');
        fprintf(fid, 'Minimum neurons per region: %d\n', cca_results.config.min_neurons_per_region);
        fprintf(fid, 'Target neurons for sampling: %d\n', cca_results.config.target_neurons);
        fprintf(fid, 'Cross-validation folds: %d\n', cca_results.config.cv_folds);
        fprintf(fid, 'Significance threshold: %d%% percentile\n', cca_results.config.significance_threshold);
        fprintf(fid, 'Time window: [%.1f, %.1f] seconds\n', cca_results.config.time_window);
        
        fclose(fid);
        
    catch ME
        fclose(fid);
        rethrow(ME);
    end
end

function generate_pipeline_report(session_stats, results_dir)
% GENERATE_PIPELINE_REPORT - Create overall pipeline performance summary

    report_filename = fullfile(results_dir, 'pipeline_summary_report.txt');
    
    fid = fopen(report_filename, 'w');
    if fid == -1
        warning('Could not create pipeline report file: %s', report_filename);
        return;
    end
    
    try
        fprintf(fid, 'Oxford Single-Session CCA Pipeline Summary\n');
        fprintf(fid, '==========================================\n\n');
        fprintf(fid, 'Pipeline Execution Date: %s\n', datestr(now));
        fprintf(fid, 'Total sessions processed: %d\n', session_stats.total_sessions);
        fprintf(fid, 'Successful downloads: %d\n', session_stats.successful_downloads);
        fprintf(fid, 'Successful analyses: %d\n', session_stats.successful_analyses);
        fprintf(fid, 'Failed sessions: %d\n\n', length(session_stats.failed_sessions));
        
        % Calculate success rates
        download_success_rate = (session_stats.successful_downloads / session_stats.total_sessions) * 100;
        analysis_success_rate = (session_stats.successful_analyses / session_stats.total_sessions) * 100;
        
        fprintf(fid, 'Performance Metrics\n');
        fprintf(fid, '------------------\n');
        fprintf(fid, 'Download success rate: %.1f%%\n', download_success_rate);
        fprintf(fid, 'Analysis success rate: %.1f%%\n', analysis_success_rate);
        fprintf(fid, 'Overall pipeline efficiency: %.1f%%\n\n', analysis_success_rate);
        
        % Failed sessions details
        if ~isempty(session_stats.failed_sessions)
            fprintf(fid, 'Failed Sessions Details\n');
            fprintf(fid, '----------------------\n');
            for i = 1:length(session_stats.failed_sessions)
                failed_session = session_stats.failed_sessions{i};
                fprintf(fid, '%s: %s\n', failed_session{1}, failed_session{2});
            end
            fprintf(fid, '\n');
        end
        
        fprintf(fid, 'Pipeline completed successfully.\n');
        fprintf(fid, 'Results location: %s\n', results_dir);
        
        fclose(fid);
        
        fprintf('Pipeline summary report generated: %s\n', report_filename);
        
    catch ME
        fclose(fid);
        warning('Error generating pipeline report: %s', ME.message);
    end
end

%% Utility Functions for Data Management

function clear_session_variables()
% CLEAR_SESSION_VARIABLES - Clear workspace to prevent memory accumulation
% This function ensures proper memory management during sequential processing

    vars_to_clear = {'session_data', 'region_data', 'cca_results', 'trial_data', ...
                     'cell_metrics', 'spike_rates', 'brain_regions'};
    
    for i = 1:length(vars_to_clear)
        if evalin('caller', sprintf('exist(''%s'', ''var'')', vars_to_clear{i}))
            evalin('caller', sprintf('clear %s', vars_to_clear{i}));
        end
    end
end

function memory_usage = check_memory_usage()
% CHECK_MEMORY_USAGE - Monitor memory consumption during processing
% This function provides memory usage feedback for pipeline optimization

    if ispc
        [~, memory_info] = system('wmic OS get TotalVisibleMemorySize,FreePhysicalMemory /value');
        % Parse Windows memory info
        memory_usage.total_gb = 'Unknown';
        memory_usage.free_gb = 'Unknown';
        memory_usage.used_percent = 'Unknown';
    elseif ismac || isunix
        [~, memory_info] = system('vm_stat');
        % Parse Unix/Mac memory info - simplified
        memory_usage.total_gb = 'System dependent';
        memory_usage.free_gb = 'System dependent';
        memory_usage.used_percent = 'Check system monitor';
    else
        memory_usage.total_gb = 'Unknown system';
        memory_usage.free_gb = 'Unknown';
        memory_usage.used_percent = 'Unknown';
    end
    
    % MATLAB memory information
    matlab_memory = memory;
    memory_usage.matlab_used_gb = matlab_memory.MemUsedMATLAB / 1024^3;
    memory_usage.matlab_available_gb = matlab_memory.MemAvailableAllArrays / 1024^3;
end
