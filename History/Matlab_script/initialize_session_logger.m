function session_logger = initialize_session_logger(logging_config, analysis_config)
% INITIALIZE_SESSION_LOGGER - Initialize comprehensive session tracking system
%
% THEORETICAL FOUNDATION:
% This logging system implements a systematic approach to documenting experimental
% exclusions in neurophysiology analysis pipelines. The system addresses the critical
% need for transparent reporting of session failures, which is essential for:
% 1. Scientific reproducibility and peer review
% 2. Quality control and protocol optimization  
% 3. Statistical power calculations for future experiments
% 4. Identification of systematic technical issues
%
% MATHEMATICAL FRAMEWORK:
% The logging system tracks the mapping function ℱ: 𝒮 → ℰ ∪ ℛ where:
% 𝒮 = set of all experimental sessions
% ℰ = enumerated failure modes 
% ℛ = successful analysis results
%
% This ensures complete observability of the experimental selection process.
%
% INPUTS:
%   logging_config - Configuration structure for logging parameters
%   analysis_config - Configuration structure for analysis parameters
%
% OUTPUT:
%   session_logger - Initialized logger structure with all tracking capabilities
%
% SCIENTIFIC RATIONALE:
% Comprehensive failure documentation prevents publication bias and allows for
% proper assessment of experimental success rates, which is crucial for 
% methodology validation and resource allocation in neuroscience research.

    % Initialize the logger structure with comprehensive tracking capabilities
    session_logger = struct();
    
    % Essential file system components
    session_logger.log_directory = logging_config.log_directory;
    session_logger.log_file_path = fullfile(logging_config.log_directory, logging_config.log_filename);
    session_logger.log_level = logging_config.log_level;
    
    % Session tracking arrays - organized by failure type for systematic analysis
    session_logger.sessions = struct();
    session_logger.sessions.total_queued = [];           % All sessions in processing queue
    session_logger.sessions.successful = [];            % Successfully completed analyses
    session_logger.sessions.failed_download = [];       % Network/server access failures
    session_logger.sessions.failed_extraction = [];     % Data format/corruption issues
    session_logger.sessions.failed_neural_criteria = []; % Insufficient neural populations
    session_logger.sessions.failed_region_validation = []; % Brain region mapping issues
    session_logger.sessions.failed_cca_computation = []; % Mathematical/numerical failures
    session_logger.sessions.failed_critical_error = []; % Unexpected system failures
    
    % Detailed failure documentation - provides specific diagnostic information
    session_logger.failure_details = struct();
    session_logger.failure_details.download_errors = containers.Map();      % Network error messages
    session_logger.failure_details.extraction_errors = containers.Map();    % File format issues
    session_logger.failure_details.neural_statistics = containers.Map();    % Neuron count data
    session_logger.failure_details.region_problems = containers.Map();      % Brain region issues
    session_logger.failure_details.cca_diagnostics = containers.Map();      % Mathematical failures
    session_logger.failure_details.system_errors = containers.Map();        % Critical system failures
    
    % Performance and resource tracking - enables computational optimization
    session_logger.performance = struct();
    session_logger.performance.session_start_times = containers.Map();
    session_logger.performance.session_durations = containers.Map();
    session_logger.performance.memory_usage = containers.Map();
    session_logger.performance.processing_stages = containers.Map();
    
    % Metadata for scientific documentation
    session_logger.metadata = struct();
    session_logger.metadata.analysis_start_time = datetime('now');
    session_logger.metadata.matlab_version = version();
    session_logger.metadata.analysis_config = analysis_config;
    session_logger.metadata.logging_config = logging_config;
    session_logger.metadata.total_sessions_planned = 0;  % Will be set during pipeline start
    
    % Statistical tracking for real-time assessment
    session_logger.statistics = struct();
    session_logger.statistics.success_rate = 0;
    session_logger.statistics.most_common_failure = '';
    session_logger.statistics.sessions_processed = 0;
    
    % Create and initialize the log file with header information
    create_log_file_header(session_logger);
    
    % Log the initialization event
    log_entry(session_logger, 'SYSTEM', 'Logger initialized', sprintf(...
        'Comprehensive session logging system activated with %s verbosity level', ...
        logging_config.log_level));
    
    fprintf('  ✓ Session logger initialized with comprehensive tracking\n');
    fprintf('    - Failure categorization: 6 distinct failure modes\n');
    fprintf('    - Performance monitoring: Enabled\n');
    fprintf('    - Statistical analysis: Real-time computation\n');
end

function create_log_file_header(session_logger)
% CREATE_LOG_FILE_HEADER - Initialize log file with comprehensive metadata
% This function creates a standardized header that documents the experimental
% context and computational environment for reproducibility

    fid = fopen(session_logger.log_file_path, 'w');
    if fid == -1
        error('Failed to create log file: %s', session_logger.log_file_path);
    end
    
    % Write standardized scientific header
    fprintf(fid, '================================================================================\n');
    fprintf(fid, 'OXFORD NEUROPHYSIOLOGY CCA ANALYSIS - SESSION PROCESSING LOG\n');
    fprintf(fid, '================================================================================\n');
    fprintf(fid, 'Analysis Start Time: %s\n', datestr(session_logger.metadata.analysis_start_time));
    fprintf(fid, 'MATLAB Version: %s\n', session_logger.metadata.matlab_version);
    fprintf(fid, 'Log Level: %s\n', session_logger.log_level);
    fprintf(fid, 'Log Directory: %s\n', session_logger.log_directory);
    fprintf(fid, '\n');
    
    % Document analysis configuration for reproducibility
    fprintf(fid, 'ANALYSIS CONFIGURATION:\n');
    fprintf(fid, '- Minimum neurons per region: %d\n', session_logger.metadata.analysis_config.min_neurons_per_region);
    fprintf(fid, '- Target neuron count: %d\n', session_logger.metadata.analysis_config.target_neurons);
    fprintf(fid, '- Time window: [%.1f, %.1f] seconds\n', session_logger.metadata.analysis_config.time_window);
    fprintf(fid, '- CCA components: %d\n', session_logger.metadata.analysis_config.n_components);
    fprintf(fid, '- Cross-validation folds: %d\n', session_logger.metadata.analysis_config.cv_folds);
    fprintf(fid, '- Significance threshold: %d percentile\n', session_logger.metadata.analysis_config.significance_threshold);
    fprintf(fid, '\n');
    
    % Create column headers for systematic data entry
    fprintf(fid, 'SESSION PROCESSING LOG:\n');
    fprintf(fid, '--------------------------------------------------------------------------------\n');
    fprintf(fid, 'Timestamp | Session_ID | Status | Stage | Details\n');
    fprintf(fid, '--------------------------------------------------------------------------------\n');
    
    fclose(fid);
end

function log_entry(session_logger, session_id, status, stage, details)
% LOG_ENTRY - Write a standardized entry to the session log
% 
% This function implements a consistent logging format that enables both
% human readability and programmatic analysis of session processing outcomes
%
% INPUTS:
%   session_logger - Logger structure
%   session_id - Unique session identifier
%   status - Processing status (SUCCESS, FAILED, WARNING, INFO)
%   stage - Processing stage where event occurred
%   details - Detailed description of the event

    % Only log if logging is enabled and file exists
    if isempty(session_logger) || ~exist(session_logger.log_file_path, 'file')
        return;
    end
    
    % Open log file in append mode
    fid = fopen(session_logger.log_file_path, 'a');
    if fid == -1
        warning('Failed to write to log file: %s', session_logger.log_file_path);
        return;
    end
    
    % Create standardized timestamp
    timestamp = datestr(datetime('now'), 'yyyy-mm-dd HH:MM:SS');
    
    % Write formatted log entry
    fprintf(fid, '%s | %s | %s | %s | %s\n', ...
        timestamp, session_id, status, stage, details);
    
    fclose(fid);
    
    % Also display to console for real-time monitoring
    if strcmp(session_logger.log_level, 'verbose')
        fprintf('  [%s] %s: %s - %s\n', timestamp, session_id, status, details);
    end
end

function log_pipeline_start(session_logger, session_list)
% LOG_PIPELINE_START - Document the initiation of pipeline processing
% This creates a comprehensive record of the experimental scope and parameters

    session_logger.metadata.total_sessions_planned = length(session_list);
    
    % Log the pipeline initiation
    log_entry(session_logger, 'PIPELINE', 'INFO', 'INITIALIZATION', ...
        sprintf('Pipeline started with %d sessions queued for processing', length(session_list)));
    
    % Document each session in the processing queue
    for i = 1:length(session_list)
        session_info = session_list{i};
        session_name = sprintf('%s_%s', session_info{1}, session_info{2});
        session_logger.sessions.total_queued{end+1} = session_name;
        
        log_entry(session_logger, session_name, 'INFO', 'QUEUED', ...
            sprintf('Session %d/%d added to processing queue', i, length(session_list)));
    end
end

function log_session_start(session_logger, session_name)
% LOG_SESSION_START - Record the beginning of individual session processing
% This enables performance tracking and timeout detection

    if isempty(session_logger)
        return;
    end
    
    % Record start time for performance analysis
    session_logger.performance.session_start_times(session_name) = datetime('now');
    
    log_entry(session_logger, session_name, 'INFO', 'START', ...
        'Session processing initiated');
end

function log_session_success(session_logger, session_name, processing_time, cca_summary)
% LOG_SESSION_SUCCESS - Document successful session completion
% This records both performance metrics and scientific outcomes

    if isempty(session_logger)
        return;
    end
    
    % Add to successful sessions list
    session_logger.sessions.successful{end+1} = session_name;
    
    % Record performance metrics
    session_logger.performance.session_durations(session_name) = processing_time;
    
    % Update real-time statistics
    session_logger.statistics.sessions_processed = session_logger.statistics.sessions_processed + 1;
    session_logger.statistics.success_rate = length(session_logger.sessions.successful) / ...
        session_logger.statistics.sessions_processed * 100;
    
    % Log detailed success information
    success_details = sprintf(...
        'Processing time: %.1f min, Regions: %d, Max R²: %.3f, Sig. components: %d', ...
        processing_time, length(cca_summary.valid_regions), ...
        cca_summary.max_R2_across_pairs, cca_summary.total_significant_components);
    
    log_entry(session_logger, session_name, 'SUCCESS', 'COMPLETED', success_details);
end

function log_session_failure(session_logger, session_name, failure_type, stage, error_details)
% LOG_SESSION_FAILURE - Document session processing failures with detailed diagnostics
%
% This function implements the core failure documentation system that enables
% systematic analysis of experimental exclusions. Each failure type is categorized
% and stored with sufficient detail for diagnostic analysis.
%
% FAILURE CATEGORIZATION SYSTEM:
% 1. DOWNLOAD_FAILED - Network/server connectivity issues
% 2. EXTRACTION_FAILED - Data format/file corruption issues  
% 3. NEURAL_CRITERIA_FAILED - Insufficient neural populations
% 4. REGION_VALIDATION_FAILED - Brain region mapping problems
% 5. CCA_COMPUTATION_FAILED - Mathematical/numerical issues
% 6. CRITICAL_ERROR - Unexpected system failures
%
% This taxonomic approach enables systematic identification of bottlenecks
% and technical issues in the experimental pipeline.

    if isempty(session_logger)
        return;
    end
    
    % Categorize and store the failure based on type
    switch upper(failure_type)
        case 'DOWNLOAD_FAILED'
            session_logger.sessions.failed_download{end+1} = session_name;
            session_logger.failure_details.download_errors(session_name) = error_details;
            
        case 'EXTRACTION_FAILED'
            session_logger.sessions.failed_extraction{end+1} = session_name;
            session_logger.failure_details.extraction_errors(session_name) = error_details;
            
        case 'NEURAL_CRITERIA_FAILED'
            session_logger.sessions.failed_neural_criteria{end+1} = session_name;
            session_logger.failure_details.neural_statistics(session_name) = error_details;
            
        case 'REGION_VALIDATION_FAILED'
            session_logger.sessions.failed_region_validation{end+1} = session_name;
            session_logger.failure_details.region_problems(session_name) = error_details;
            
        case 'CCA_COMPUTATION_FAILED'
            session_logger.sessions.failed_cca_computation{end+1} = session_name;
            session_logger.failure_details.cca_diagnostics(session_name) = error_details;
            
        case 'CRITICAL_ERROR'
            session_logger.sessions.failed_critical_error{end+1} = session_name;
            session_logger.failure_details.system_errors(session_name) = error_details;
            
        otherwise
            warning('Unknown failure type: %s', failure_type);
            session_logger.sessions.failed_critical_error{end+1} = session_name;
            session_logger.failure_details.system_errors(session_name) = ...
                sprintf('Unknown failure type: %s - %s', failure_type, error_details);
    end
    
    % Update processing statistics
    session_logger.statistics.sessions_processed = session_logger.statistics.sessions_processed + 1;
    session_logger.statistics.success_rate = length(session_logger.sessions.successful) / ...
        session_logger.statistics.sessions_processed * 100;
    
    % Determine most common failure type for real-time feedback
    update_failure_statistics(session_logger);
    
    % Log the failure with appropriate verbosity
    log_entry(session_logger, session_name, 'FAILED', stage, ...
        sprintf('%s: %s', failure_type, error_details));
end

function update_failure_statistics(session_logger)
% UPDATE_FAILURE_STATISTICS - Compute real-time failure mode statistics
% This enables identification of the most common failure patterns for
% immediate feedback and protocol optimization

    failure_counts = struct();
    failure_counts.download = length(session_logger.sessions.failed_download);
    failure_counts.extraction = length(session_logger.sessions.failed_extraction);
    failure_counts.neural_criteria = length(session_logger.sessions.failed_neural_criteria);
    failure_counts.region_validation = length(session_logger.sessions.failed_region_validation);
    failure_counts.cca_computation = length(session_logger.sessions.failed_cca_computation);
    failure_counts.critical_error = length(session_logger.sessions.failed_critical_error);
    
    % Find the most common failure mode
    [max_count, failure_names] = max(struct2array(failure_counts));
    failure_types = fieldnames(failure_counts);
    
    if max_count > 0
        session_logger.statistics.most_common_failure = failure_types{failure_names};
    else
        session_logger.statistics.most_common_failure = 'none';
    end
end

function log_critical_pipeline_failure(session_logger, matlab_error)
% LOG_CRITICAL_PIPELINE_FAILURE - Document catastrophic pipeline failures
% This captures system-level failures that prevent pipeline completion

    if isempty(session_logger)
        return;
    end
    
    % Extract detailed error information
    error_details = sprintf('MATLAB Error: %s | Stack: %s', ...
        matlab_error.message, matlab_error.stack(1).name);
    
    log_entry(session_logger, 'PIPELINE', 'CRITICAL_FAILURE', 'SYSTEM', error_details);
end

function generate_exclusion_analysis_report(session_logger)
% GENERATE_EXCLUSION_ANALYSIS_REPORT - Create comprehensive failure analysis
%
% This function generates a detailed statistical analysis of session exclusions
% that meets scientific publication standards. The report includes:
% 1. Quantitative failure rate analysis
% 2. Categorical breakdown of exclusion reasons  
% 3. Performance metrics and computational efficiency
% 4. Recommendations for protocol optimization
%
% SCIENTIFIC IMPORTANCE:
% This systematic documentation of experimental exclusions is essential for:
% - Transparent reporting in peer review
% - Meta-analyses and systematic reviews
% - Protocol validation and improvement
% - Resource allocation and planning

    if isempty(session_logger)
        return;
    end
    
    % Generate timestamp for report
    report_timestamp = datestr(datetime('now'), 'yyyymmdd_HHMMSS');
    report_filename = sprintf('exclusion_analysis_report_%s.txt', report_timestamp);
    report_path = fullfile(session_logger.log_directory, report_filename);
    
    % Create comprehensive exclusion analysis report
    fid = fopen(report_path, 'w');
    if fid == -1
        warning('Failed to create exclusion analysis report');
        return;
    end
    
    % Report header with metadata
    fprintf(fid, '================================================================================\n');
    fprintf(fid, 'OXFORD CCA PIPELINE - SESSION EXCLUSION ANALYSIS REPORT\n');
    fprintf(fid, '================================================================================\n');
    fprintf(fid, 'Report Generated: %s\n', datestr(datetime('now')));
    fprintf(fid, 'Analysis Period: %s to %s\n', ...
        datestr(session_logger.metadata.analysis_start_time), datestr(datetime('now')));
    fprintf(fid, 'Total Sessions Planned: %d\n', session_logger.metadata.total_sessions_planned);
    fprintf(fid, 'Sessions Processed: %d\n', session_logger.statistics.sessions_processed);
    fprintf(fid, '\n');
    
    % Overall success statistics
    total_successful = length(session_logger.sessions.successful);
    total_failed = session_logger.statistics.sessions_processed - total_successful;
    
    fprintf(fid, 'OVERALL ANALYSIS STATISTICS:\n');
    fprintf(fid, '- Successful analyses: %d (%.1f%%)\n', total_successful, session_logger.statistics.success_rate);
    fprintf(fid, '- Failed analyses: %d (%.1f%%)\n', total_failed, 100 - session_logger.statistics.success_rate);
    fprintf(fid, '- Most common failure: %s\n', session_logger.statistics.most_common_failure);
    fprintf(fid, '\n');
    
    % Detailed failure categorization
    fprintf(fid, 'DETAILED FAILURE CATEGORIZATION:\n');
    fprintf(fid, '1. Download Failures: %d sessions\n', length(session_logger.sessions.failed_download));
    fprintf(fid, '2. Data Extraction Failures: %d sessions\n', length(session_logger.sessions.failed_extraction));
    fprintf(fid, '3. Neural Criteria Failures: %d sessions\n', length(session_logger.sessions.failed_neural_criteria));
    fprintf(fid, '4. Region Validation Failures: %d sessions\n', length(session_logger.sessions.failed_region_validation));
    fprintf(fid, '5. CCA Computation Failures: %d sessions\n', length(session_logger.sessions.failed_cca_computation));
    fprintf(fid, '6. Critical System Errors: %d sessions\n', length(session_logger.sessions.failed_critical_error));
    fprintf(fid, '\n');
    
    % List all excluded sessions with specific reasons
    fprintf(fid, 'DETAILED EXCLUSION LIST:\n');
    fprintf(fid, '================================================================================\n');
    
    % Helper function to write exclusion details
    write_exclusion_category(fid, 'DOWNLOAD FAILURES', session_logger.sessions.failed_download, ...
        session_logger.failure_details.download_errors);
    write_exclusion_category(fid, 'DATA EXTRACTION FAILURES', session_logger.sessions.failed_extraction, ...
        session_logger.failure_details.extraction_errors);
    write_exclusion_category(fid, 'NEURAL CRITERIA FAILURES', session_logger.sessions.failed_neural_criteria, ...
        session_logger.failure_details.neural_statistics);
    write_exclusion_category(fid, 'REGION VALIDATION FAILURES', session_logger.sessions.failed_region_validation, ...
        session_logger.failure_details.region_problems);
    write_exclusion_category(fid, 'CCA COMPUTATION FAILURES', session_logger.sessions.failed_cca_computation, ...
        session_logger.failure_details.cca_diagnostics);
    write_exclusion_category(fid, 'CRITICAL SYSTEM ERRORS', session_logger.sessions.failed_critical_error, ...
        session_logger.failure_details.system_errors);
    
    % Performance analysis
    if session_logger.metadata.logging_config.track_performance_metrics
        fprintf(fid, '\nPERFORMAN ANALYSIS:\n');
        fprintf(fid, '================================================================================\n');
        
        % Calculate performance statistics
        if ~isempty(session_logger.performance.session_durations.keys)
            durations = cell2mat(session_logger.performance.session_durations.values);
            fprintf(fid, 'Processing Time Statistics (successful sessions only):\n');
            fprintf(fid, '- Mean processing time: %.1f minutes\n', mean(durations));
            fprintf(fid, '- Median processing time: %.1f minutes\n', median(durations));
            fprintf(fid, '- Min/Max processing time: %.1f / %.1f minutes\n', min(durations), max(durations));
            fprintf(fid, '- Standard deviation: %.1f minutes\n', std(durations));
        end
    end
    
    % Scientific recommendations based on failure patterns
    fprintf(fid, '\n');
    fprintf(fid, 'RECOMMENDATIONS FOR PROTOCOL OPTIMIZATION:\n');
    fprintf(fid, '================================================================================\n');
    generate_optimization_recommendations(fid, session_logger);
    
    fclose(fid);
    
    % Log the report generation
    log_entry(session_logger, 'SYSTEM', 'INFO', 'REPORT', ...
        sprintf('Exclusion analysis report generated: %s', report_filename));
    
    fprintf('✓ Comprehensive exclusion analysis report generated: %s\n', report_path);
end

function write_exclusion_category(fid, category_name, session_list, error_details)
% WRITE_EXCLUSION_CATEGORY - Helper function to document specific failure categories
% This provides detailed information for each excluded session

    if isempty(session_list)
        return;
    end
    
    fprintf(fid, '\n%s (%d sessions):\n', category_name, length(session_list));
    fprintf(fid, '----------------------------------------\n');
    
    for i = 1:length(session_list)
        session_name = session_list{i};
        if error_details.isKey(session_name)
            details = error_details(session_name);
            fprintf(fid, '- %s: %s\n', session_name, details);
        else
            fprintf(fid, '- %s: [Details not available]\n', session_name);
        end
    end
end

function generate_optimization_recommendations(fid, session_logger)
% GENERATE_OPTIMIZATION_RECOMMENDATIONS - Provide evidence-based recommendations
% This analysis helps improve experimental protocols based on observed failure patterns

    total_failures = session_logger.statistics.sessions_processed - length(session_logger.sessions.successful);
    
    if total_failures == 0
        fprintf(fid, '✓ No failures detected - pipeline is operating optimally\n');
        return;
    end
    
    % Analyze failure patterns and provide specific recommendations
    failure_counts = struct();
    failure_counts.download = length(session_logger.sessions.failed_download);
    failure_counts.extraction = length(session_logger.sessions.failed_extraction);
    failure_counts.neural_criteria = length(session_logger.sessions.failed_neural_criteria);
    failure_counts.region_validation = length(session_logger.sessions.failed_region_validation);
    failure_counts.cca_computation = length(session_logger.sessions.failed_cca_computation);
    failure_counts.critical_error = length(session_logger.sessions.failed_critical_error);
    
    % Generate targeted recommendations based on dominant failure modes
    if failure_counts.download > total_failures * 0.3
        fprintf(fid, '⚠ High download failure rate (%.1f%%) detected:\n', ...
            failure_counts.download / total_failures * 100);
        fprintf(fid, '  - Consider implementing retry mechanisms with exponential backoff\n');
        fprintf(fid, '  - Verify network connectivity and server accessibility\n');
        fprintf(fid, '  - Consider batch downloading during off-peak hours\n\n');
    end
    
    if failure_counts.neural_criteria > total_failures * 0.3
        fprintf(fid, '⚠ High neural criteria failure rate (%.1f%%) detected:\n', ...
            failure_counts.neural_criteria / total_failures * 100);
        fprintf(fid, '  - Consider reducing minimum neuron threshold (%d neurons currently)\n', ...
            session_logger.metadata.analysis_config.min_neurons_per_region);
        fprintf(fid, '  - Implement adaptive thresholds based on brain region\n');
        fprintf(fid, '  - Review spike sorting quality metrics\n\n');
    end
    
    if failure_counts.extraction > total_failures * 0.2
        fprintf(fid, '⚠ Data extraction failures detected (%.1f%%):\n', ...
            failure_counts.extraction / total_failures * 100);
        fprintf(fid, '  - Implement data integrity checks during download\n');
        fprintf(fid, '  - Add support for alternative data formats\n');
        fprintf(fid, '  - Consider implementing data recovery mechanisms\n\n');
    end
    
    % Overall efficiency assessment
    success_rate = session_logger.statistics.success_rate;
    if success_rate >= 80
        fprintf(fid, '✓ Pipeline efficiency is within acceptable range (%.1f%% success rate)\n', success_rate);
    elseif success_rate >= 60
        fprintf(fid, '⚠ Pipeline efficiency could be improved (%.1f%% success rate)\n', success_rate);
        fprintf(fid, '  - Focus on addressing the most common failure mode: %s\n', ...
            session_logger.statistics.most_common_failure);
    else
        fprintf(fid, '❌ Pipeline efficiency requires immediate attention (%.1f%% success rate)\n', success_rate);
        fprintf(fid, '  - Systematic review of all failure modes recommended\n');
        fprintf(fid, '  - Consider pilot studies with modified parameters\n');
    end
end
