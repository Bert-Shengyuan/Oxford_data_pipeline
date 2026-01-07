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