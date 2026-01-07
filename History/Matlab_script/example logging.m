%% PRACTICAL EXAMPLES: Enhanced Logging in Action
% 
% This file demonstrates how the enhanced logging system captures and documents
% different types of session failures. Each example illustrates the transformation
% from basic error reporting to comprehensive scientific documentation.
%
% PEDAGOGICAL PURPOSE:
% These examples illustrate the critical difference between:
% 1. Basic error handling: "Session failed"
% 2. Scientific documentation: "Session failed due to insufficient VISp neurons 
%    (n=23 < 50 required), while MOs had adequate population (n=67)"
%
% This level of detail enables systematic protocol optimization and transparent
% reporting of experimental exclusions in peer-reviewed publications.

%% Example 1: Neural Criteria Failure - Detailed Population Analysis
% 
% SCENARIO: Session yp014_220210 has some brain regions with insufficient neurons
% 
% BASIC LOGGING (Original system):
% "Session yp014_220210 failed: Insufficient neurons"
%
% ENHANCED LOGGING (Our system):
function demonstrate_neural_criteria_logging()
    fprintf('=== Example 1: Neural Criteria Failure Documentation ===\n');
    
    % Simulate session failure due to insufficient neural populations
    session_name = 'yp014_220210';
    
    % Detailed neuronal statistics that our system would capture
    neural_stats = struct();
    neural_stats.total_neurons = 342;
    neural_stats.regions_found = {'VISp', 'MOs', 'SC', 'CP'};
    neural_stats.neuron_counts = [23, 67, 45, 89];  % Per region
    neural_stats.threshold = 50;
    neural_stats.regions_above_threshold = 2;  % MOs and CP only
    
    % Our enhanced system logs this comprehensive information:
    detailed_failure_message = sprintf(['Neural criteria failure: %d total neurons across %d regions. ' ...
        'Regions above threshold (%d neurons): %d/4. ' ...
        'Specific counts: VISp=%d, MOs=%d, SC=%d, CP=%d. ' ...
        'Analysis requires minimum 2 region pairs with ≥%d neurons each.'], ...
        neural_stats.total_neurons, length(neural_stats.regions_found), ...
        neural_stats.threshold, neural_stats.regions_above_threshold, ...
        neural_stats.neuron_counts(1), neural_stats.neuron_counts(2), ...
        neural_stats.neuron_counts(3), neural_stats.neuron_counts(4), ...
        neural_stats.threshold);
    
    fprintf('Enhanced failure documentation:\n');
    fprintf('"%s"\n\n', detailed_failure_message);
    
    % SCIENTIFIC BENEFIT: This detailed information enables researchers to:
    % 1. Adjust neuron count thresholds based on empirical data
    % 2. Identify problematic brain regions requiring protocol modification
    % 3. Report precise exclusion criteria in methods sections
    % 4. Calculate statistical power for future experiments
    
    fprintf('Scientific applications of this documentation:\n');
    fprintf('- Protocol optimization: Consider reducing threshold from %d to %d neurons\n', ...
        neural_stats.threshold, min(neural_stats.neuron_counts(neural_stats.neuron_counts > 0)));
    fprintf('- Region-specific analysis: VISp and SC show consistently low yields\n');
    fprintf('- Power analysis: %.1f%% of regions meet current criteria\n', ...
        neural_stats.regions_above_threshold / length(neural_stats.regions_found) * 100);
end

%% Example 2: CCA Computation Failure - Mathematical Diagnostics
%
% SCENARIO: CCA computation fails due to numerical instability
%
function demonstrate_cca_failure_logging()
    fprintf('=== Example 2: CCA Computation Failure Documentation ===\n');
    
    session_name = 'yp021_220403';
    
    % Mathematical diagnostics that our system captures
    cca_diagnostics = struct();
    cca_diagnostics.region_pair = {'VISp', 'MOs'};
    cca_diagnostics.matrix_dimensions = [250, 52; 250, 67];  % [observations, neurons]
    cca_diagnostics.condition_numbers = [1.2e8, 3.4e9];     % Covariance matrix condition
    cca_diagnostics.rank_deficiency = [0, 2];               % Rank deficiencies detected
    cca_diagnostics.numerical_tolerance = 1e-12;
    
    % Enhanced diagnostic message
    diagnostic_message = sprintf(['CCA computation failure in %s↔%s: ' ...
        'Matrix conditioning issues detected. ' ...
        'Covariance condition numbers: X=%.1e, Y=%.1e (threshold=1e6). ' ...
        'Rank deficiencies: X=%d, Y=%d neurons. ' ...
        'Numerical instability due to near-singular covariance matrices. ' ...
        'Recommendation: Increase regularization or reduce neuron count.'], ...
        cca_diagnostics.region_pair{1}, cca_diagnostics.region_pair{2}, ...
        cca_diagnostics.condition_numbers(1), cca_diagnostics.condition_numbers(2), ...
        cca_diagnostics.rank_deficiency(1), cca_diagnostics.rank_deficiency(2));
    
    fprintf('Enhanced CCA failure documentation:\n');
    fprintf('"%s"\n\n', diagnostic_message);
    
    % SCIENTIFIC BENEFIT: Mathematical diagnostics enable:
    % 1. Identification of numerical stability issues
    % 2. Optimization of regularization parameters
    % 3. Assessment of data quality and preprocessing needs
    % 4. Development of robust CCA implementations
    
    fprintf('Mathematical insights from this documentation:\n');
    fprintf('- Matrix conditioning: Region Y requires regularization (cond=%.1e)\n', ...
        cca_diagnostics.condition_numbers(2));
    fprintf('- Rank analysis: %d neurons showing linear dependency in region Y\n', ...
        cca_diagnostics.rank_deficiency(2));
    fprintf('- Algorithmic recommendation: Implement ridge regularization with λ=%.1e\n', ...
        1/cca_diagnostics.condition_numbers(2));
end

%% Example 3: Download Failure - Network Diagnostics
%
% SCENARIO: Session download fails due to server connectivity issues
%
function demonstrate_download_failure_logging()
    fprintf('=== Example 3: Download Failure Documentation ===\n');
    
    session_name = 'yp022_220405';
    
    % Network diagnostics that our system captures
    network_diagnostics = struct();
    network_diagnostics.server = 'hpc-login-1.cubi.bihealth.org';
    network_diagnostics.connection_attempts = 3;
    network_diagnostics.timeout_duration = 300;  % seconds
    network_diagnostics.data_size_expected = 2.3;  % GB
    network_diagnostics.partial_download = 0.8;   % GB received
    network_diagnostics.error_code = 'SSH_DISCONNECT_PROTOCOL_ERROR';
    network_diagnostics.retry_timestamp = datetime('2024-03-15 14:23:17');
    
    % Enhanced network failure documentation
    network_message = sprintf(['Download failure for session %s: ' ...
        'Server: %s, Error: %s. ' ...
        'Transfer interrupted after %.1f/%.1f GB (%.1f%% complete). ' ...
        'Connection attempts: %d, Timeout: %d seconds. ' ...
        'Last retry: %s. ' ...
        'Recommendation: Implement exponential backoff retry mechanism.'], ...
        session_name, network_diagnostics.server, network_diagnostics.error_code, ...
        network_diagnostics.partial_download, network_diagnostics.data_size_expected, ...
        network_diagnostics.partial_download/network_diagnostics.data_size_expected*100, ...
        network_diagnostics.connection_attempts, network_diagnostics.timeout_duration, ...
        datestr(network_diagnostics.retry_timestamp));
    
    fprintf('Enhanced download failure documentation:\n');
    fprintf('"%s"\n\n', network_message);
    
    % SCIENTIFIC BENEFIT: Network diagnostics enable:
    % 1. Infrastructure optimization and capacity planning
    % 2. Identification of systematic connectivity issues
    % 3. Development of robust data transfer protocols
    % 4. Resource allocation for high-performance computing
    
    fprintf('Infrastructure insights from this documentation:\n');
    fprintf('- Transfer efficiency: %.1f%% completion suggests network instability\n', ...
        network_diagnostics.partial_download/network_diagnostics.data_size_expected*100);
    fprintf('- Retry strategy: %d attempts indicate need for exponential backoff\n', ...
        network_diagnostics.connection_attempts);
    fprintf('- Capacity planning: %.1f GB sessions require %d+ minute transfer windows\n', ...
        network_diagnostics.data_size_expected, network_diagnostics.timeout_duration/60);
end

%% Example 4: Comprehensive Session Report Generation
%
% This demonstrates how our system generates publication-ready exclusion reports
%
function demonstrate_exclusion_report_generation()
    fprintf('=== Example 4: Publication-Ready Exclusion Analysis ===\n');
    
    % Simulated analysis outcomes from a typical experiment
    experiment_summary = struct();
    experiment_summary.total_sessions = 40;
    experiment_summary.successful = 28;
    experiment_summary.failed_download = 3;
    experiment_summary.failed_neural_criteria = 6;
    experiment_summary.failed_cca_computation = 2;
    experiment_summary.failed_other = 1;
    
    % Generate publication-quality exclusion summary
    fprintf('Publication-Ready Methods Section Text:\n');
    fprintf('----------------------------------------\n');
    
    success_rate = experiment_summary.successful / experiment_summary.total_sessions * 100;
    
    methods_text = sprintf(['Of %d total recording sessions, %d (%.1f%%) met all inclusion criteria ' ...
        'for canonical correlation analysis. Exclusions were systematically documented: ' ...
        '%d sessions (%.1f%%) failed due to insufficient neural populations (<%d neurons per region), ' ...
        '%d sessions (%.1f%%) experienced server connectivity issues during data acquisition, ' ...
        '%d sessions (%.1f%%) encountered numerical instability during CCA computation ' ...
        '(matrix condition number >1×10⁶), and %d session (%.1f%%) failed due to data corruption. ' ...
        'All exclusions were determined prior to statistical analysis to prevent selection bias.'], ...
        experiment_summary.total_sessions, experiment_summary.successful, success_rate, ...
        experiment_summary.failed_neural_criteria, ...
        experiment_summary.failed_neural_criteria/experiment_summary.total_sessions*100, 50, ...
        experiment_summary.failed_download, ...
        experiment_summary.failed_download/experiment_summary.total_sessions*100, ...
        experiment_summary.failed_cca_computation, ...
        experiment_summary.failed_cca_computation/experiment_summary.total_sessions*100, ...
        experiment_summary.failed_other, ...
        experiment_summary.failed_other/experiment_summary.total_sessions*100);
    
    fprintf('"%s"\n\n', methods_text);
    
    % SCIENTIFIC BENEFIT: This level of documentation provides:
    % 1. Complete transparency for peer review
    % 2. Precise exclusion criteria for replication studies
    % 3. Statistical power calculations for meta-analyses
    % 4. Evidence-based protocol optimization recommendations
    
    fprintf('Benefits for scientific publication:\n');
    fprintf('- Peer review transparency: All exclusions documented with specific criteria\n');
    fprintf('- Replication support: Precise protocols enable independent validation\n');
    fprintf('- Meta-analysis readiness: Standardized reporting format\n');
    fprintf('- Protocol optimization: Evidence-based recommendations for improvement\n');
end

%% Main demonstration function
function run_enhanced_logging_examples()
    fprintf('ENHANCED OXFORD CCA LOGGING SYSTEM - PRACTICAL EXAMPLES\n');
    fprintf('=======================================================\n\n');
    
    fprintf('This demonstration illustrates the transformation from basic error handling\n');
    fprintf('to comprehensive scientific documentation of experimental exclusions.\n\n');
    
    demonstrate_neural_criteria_logging();
    fprintf('\n');
    
    demonstrate_cca_failure_logging();
    fprintf('\n');
    
    demonstrate_download_failure_logging();
    fprintf('\n');
    
    demonstrate_exclusion_report_generation();
    
    fprintf('\n=======================================================\n');
    fprintf('SUMMARY: Enhanced logging transforms experimental failures from\n');
    fprintf('uninformative error messages into actionable scientific insights\n');
    fprintf('that enable protocol optimization and transparent reporting.\n');
end

% Execute the demonstration
if ~exist('suppress_demo', 'var')
    run_enhanced_logging_examples();
end