function oxford_glm_summary_analysis_STR()
% OXFORD_GLM_SUMMARY_ANALYSIS_STR
% STR-focused aggregation and visualization of GLM sensitivity results
%
% THEORETICAL FRAMEWORK:
% This function synthesizes sensitivity analysis results to quantify
% encoding concentration across STR-related neural pathways. We employ
% the concentration metric defined as:
%
%   $C = R^2(\mathcal{N}_0) - R^2(\mathcal{N}_{0.5})$
%
% where $\mathcal{N}_\rho$ denotes the neuron set remaining after removing
% proportion $\rho \in [0, 1]$ of top-ranked neurons (ranked by absolute
% GLM coefficient magnitude $|\beta_i|$).
%
% INTERPRETATION:
% - Lower $R^2(\mathcal{N}_{0.5})$ indicates steeper $R^2$ decay, hence
%   higher encoding concentration (sparse representation)
% - Higher $R^2(\mathcal{N}_{0.5})$ suggests distributed encoding across
%   the neural population
%
% ARCHITECTURAL NOTE:
% This script operates downstream of the two-stage sensitivity pipeline:
%   1. oxford_GLM_CCA_coefficients_extract.m → GLM coefficient estimation
%   2. oxford_GLM_sensitivity.m → sensitivity_*.mat file generation
%   3. This script → STR-specific aggregation and visualization
%
% DATA STRUCTURE (from sensitivity_*.mat files):
%   sensitivity_results.region1              - First region identifier
%   sensitivity_results.region2              - Second region identifier
%   sensitivity_results.removal_percentages  - [1 × n_steps] removal schedule
%   sensitivity_results.sessions{s}.region1_toprank  - [1 × n_steps] R² values
%   sensitivity_results.sessions{s}.region2_toprank  - [1 × n_steps] R² values
%
% OUTPUTS:
%   - STR regional drop heatmap (asymmetric matrix visualization)
%   - STR concentration bar plot (pathway comparison)
%   - STR pathway boxplot with paired statistical tests
%
% USAGE:
%   oxford_glm_summary_analysis_STR()

    clc; close all;
    fprintf('=== Oxford Dataset: STR-Focused GLM Summary Analysis ===\n\n');
    
    %% =====================================================================
    %  CONFIGURATION SECTION
    %  =====================================================================
    
    % Base directory for Oxford dataset
    base_results_dir = '/Users/shengyuancai/Downloads/Oxford_dataset/';
    
    % Data source: Choose between task conditions
    % Options: 'sessions_cued_hit_long_results' or 'sessions_spont_short_results'
    data_source = 'sessions_cued_hit_long_result';
    
    % Full path to session results
    session_results_dir = fullfile(base_results_dir, data_source);
    
    % Analysis parameters
    component_to_analyze = 1;
    
    % GLM results directory (from coefficient extraction script)
    glm_results_dir = fullfile(session_results_dir, ...
        sprintf('GLM_Analysis_Component_%d', component_to_analyze));
    
    % Sensitivity analysis output directory
    output_dir = fullfile(glm_results_dir, 'sensitivity_analysis');
    
    %% =====================================================================
    %  ANATOMICAL ORDERING DEFINITION
    %  =====================================================================
    
    ANATOMICAL_ORDER = {
        'mPFC', 'ORB', 'MOp', 'MOs', 'OLF', ...  % Cortical regions
        'STR', 'STRv', 'HIPP', ...               % Striatal & limbic
        'MD', 'LP', 'VALVM', 'VPMPO', 'ILM', ... % Thalamic nuclei
        'HY', ...                                 % Hypothalamic
        'fiber', ...                              % Fiber tracts
        'other'                                   % Catch-all category
    };
    
    %% =====================================================================
    %  DIRECTORY VERIFICATION
    %  =====================================================================
    
    fprintf('Configuration:\n');
    fprintf('  Data source: %s\n', data_source);
    fprintf('  GLM results directory: %s\n', glm_results_dir);
    fprintf('  Component analyzed: %d\n', component_to_analyze);
    fprintf('  Output directory: %s\n\n', output_dir);
    
    % Verify output directory exists
    if ~exist(output_dir, 'dir')
        error(['Sensitivity analysis directory not found at:\n  %s\n' ...
               'Please run oxford_GLM_sensitivity.m first.'], output_dir);
    end
    
    %% =====================================================================
    %  EXECUTE STR-FOCUSED ANALYSES
    %  =====================================================================
    
    fprintf('\n========================================\n');
    fprintf('Beginning STR-Focused Sensitivity Analysis\n');
    fprintf('========================================\n\n');
    
    % Execute all three analysis functions
    % All functions load from sensitivity_*.mat files
    create_STR_regional_drop_heatmap(output_dir, ANATOMICAL_ORDER);
    create_STR_concentration_barplot(output_dir, ANATOMICAL_ORDER);
    create_STR_pathway_boxplot(output_dir, ANATOMICAL_ORDER);
    
    fprintf('\n========================================\n');
    fprintf('STR analysis complete!\n');
    fprintf('All results saved to: %s\n', output_dir);
    fprintf('========================================\n');
end


%% =========================================================================
%  FUNCTION 1: STR REGIONAL DROP HEATMAP
%  =========================================================================

function create_STR_regional_drop_heatmap(output_dir, anatomical_order)
% CREATE_STR_REGIONAL_DROP_HEATMAP Creates asymmetric heatmap from sensitivity files
%
% MATHEMATICAL FORMULATION:
% The matrix element $M_{ij}$ represents the mean residual $R^2$ of region $i$
% at 50% neuron removal when paired with region $j$:
%
%   $M_{ij} = \mathbb{E}_{\text{sessions}}\left[ R^2_i(\mathcal{N}_{0.5}) \mid \text{paired with } j \right]$
%
% Since neural encoding asymmetry is expected, we have $M_{ij} \neq M_{ji}$
% in general. Lower values indicate higher encoding concentration.
%
% INPUTS:
%   output_dir       - Directory containing sensitivity_*.mat files
%   anatomical_order - Cell array defining canonical region ordering

    fprintf('Creating STR-focused regional drop heatmap from sensitivity files...\n');
    
    % Define STR target regions with consistent naming
    STR_target_regions = {'mPFC', 'ORB', 'MOp', 'MOs', 'LP', 'MD'};
    
    % Include STR itself for within-region analysis
    all_regions = ['STR', STR_target_regions];
    n_regions = length(all_regions);
    
    % Initialize concentration matrix with NaN for missing data
    % This maintains distinction between "zero concentration" and "no data"
    concentration_matrix = nan(n_regions, n_regions);
    session_count_matrix = zeros(n_regions, n_regions);
    
    % Load sensitivity analysis results from files
    sensitivity_files = dir(fullfile(output_dir, 'sensitivity_*.mat'));
    fprintf('  Found %d sensitivity analysis files\n', length(sensitivity_files));
    
    if isempty(sensitivity_files)
        error('No sensitivity files found. Please run oxford_GLM_sensitivity.m first.');
    end
    
    % Process each sensitivity result file
    for f = 1:length(sensitivity_files)
        file_path = fullfile(output_dir, sensitivity_files(f).name);
        data = load(file_path);
        
        % Extract sensitivity_results structure
        if ~isfield(data, 'sensitivity_results')
            fprintf('  Warning: File %s missing sensitivity_results field, skipping\n', ...
                sensitivity_files(f).name);
            continue;
        end
        
        sensitivity_results = data.sensitivity_results;
        
        % Apply minimum session threshold for statistical robustness
        if ~isfield(sensitivity_results, 'sessions') || ...
           length(sensitivity_results.sessions) < 2
            fprintf('  Skipping pair with insufficient sessions (<2)\n');
            continue;
        end
        
        % Extract region identifiers
        if ~isfield(sensitivity_results, 'region1') || ...
           ~isfield(sensitivity_results, 'region2')
            fprintf('  Warning: Missing region identifiers in %s\n', ...
                sensitivity_files(f).name);
            continue;
        end
        
        region1 = sensitivity_results.region1;
        region2 = sensitivity_results.region2;
        
        % Check if this pairing involves STR and one of our target regions
        [is_STR_pair, idx1, idx2] = check_STR_pairing(region1, region2, all_regions);
        
        if ~is_STR_pair
            continue;
        end
        
        % Find 50% removal index
        if ~isfield(sensitivity_results, 'removal_percentages')
            fprintf('  Warning: Missing removal_percentages in %s\n', ...
                sensitivity_files(f).name);
            continue;
        end
        
        removal_pct = sensitivity_results.removal_percentages;
        idx_50 = find(removal_pct >= 50, 1, 'first');
        
        if isempty(idx_50)
            fprintf('  Warning: No 50%% removal point found for %s vs %s\n', ...
                region1, region2);
            continue;
        end
        
        % Extract session-level concentration measurements
        n_sessions = length(sensitivity_results.sessions);
        region1_concentrations = zeros(n_sessions, 1);
        region2_concentrations = zeros(n_sessions, 1);
        valid_sessions = 0;
        
        for s = 1:n_sessions
            session = sensitivity_results.sessions{s};
            
            if ~isfield(session, 'region1_toprank') || ...
               ~isfield(session, 'region2_toprank')
                fprintf('    Warning: Session %d missing toprank fields\n', s);
                continue;
            end
            
            valid_sessions = valid_sessions + 1;
            region1_concentrations(valid_sessions) = session.region1_toprank(idx_50);
            region2_concentrations(valid_sessions) = session.region2_toprank(idx_50);
        end
        
        % Trim to valid sessions
        region1_concentrations = region1_concentrations(1:valid_sessions);
        region2_concentrations = region2_concentrations(1:valid_sessions);
        
        if valid_sessions < 2
            continue;
        end
        
        % Calculate mean encoding concentration across sessions
        region1_mean = mean(region1_concentrations);
        region2_mean = mean(region2_concentrations);
        
        % Store in asymmetric matrix
        concentration_matrix(idx1, idx2) = region1_mean;
        concentration_matrix(idx2, idx1) = region2_mean;
        session_count_matrix(idx1, idx2) = valid_sessions;
        session_count_matrix(idx2, idx1) = valid_sessions;
        
        fprintf('  Processed %s vs %s: R² = %.3f, %.3f (n=%d sessions)\n', ...
            region1, region2, region1_mean, region2_mean, valid_sessions);
    end
    
    % Create publication-quality heatmap visualization
    figure('Position', [100, 100, 900, 800]);
    h = imagesc(concentration_matrix);
    
    % Configure NaN transparency for missing data visualization
    set(h, 'AlphaData', ~isnan(concentration_matrix));
    
    % Apply perceptually uniform colormap
    % Blue (high R² = low concentration) → Red (low R² = high concentration)
    n_colors = 256;
    custom_colormap = [linspace(0.2, 1, n_colors)', ...
                       linspace(0.4, 0.2, n_colors)', ...
                       linspace(0.8, 0.2, n_colors)'];
    colormap(custom_colormap);
    
    % Set dynamic color axis based on actual data range
    valid_data = concentration_matrix(~isnan(concentration_matrix));
    if ~isempty(valid_data)
        clim([min(valid_data), max(valid_data)]);
    end
    
    % Configure colorbar
    c = colorbar;
    c.Label.String = 'Mean R^2 at 50% Neuron Removal';
    c.Label.FontSize = 16;
    c.FontSize = 14;
    c.Label.Rotation = 90;
    
    % Configure axis appearance
    ax = gca;
    ax.XAxisLocation = 'bottom';
    set(ax, 'XTick', 1:n_regions, 'XTickLabel', all_regions, ...
            'YTick', 1:n_regions, 'YTickLabel', all_regions);
    set(ax, 'FontSize', 14);
    xtickangle(45);
    
    xlabel('Partner Region', 'FontSize', 16, 'FontWeight', 'bold');
    ylabel('Source Region', 'FontSize', 16, 'FontWeight', 'bold');
    title('STR Pathway Encoding Concentration (Asymmetric Matrix)', ...
        'FontSize', 18, 'FontWeight', 'bold');
    
    % Add value annotations
    for i = 1:n_regions
        for j = 1:n_regions
            if ~isnan(concentration_matrix(i, j))
                text(j, i, sprintf('%.2f', concentration_matrix(i, j)), ...
                    'HorizontalAlignment', 'center', ...
                    'VerticalAlignment', 'middle', ...
                    'FontSize', 11, 'FontWeight', 'bold', 'Color', 'w');
            end
        end
    end
    
    axis square;
    set(gca, 'Box', 'on', 'LineWidth', 1.5);
    
    % Save outputs
    saveas(gcf, fullfile(output_dir, 'STR_regional_drop_heatmap.png'));
    saveas(gcf, fullfile(output_dir, 'STR_regional_drop_heatmap.fig'));
    close(gcf);
    
    % Save matrix data
    heatmap_data = struct();
    heatmap_data.concentration_matrix = concentration_matrix;
    heatmap_data.session_count_matrix = session_count_matrix;
    heatmap_data.region_labels = all_regions;
    heatmap_data.anatomical_order = anatomical_order;
    save(fullfile(output_dir, 'STR_heatmap_data.mat'), 'heatmap_data');
    
    fprintf('  Heatmap saved to: %s\n\n', output_dir);
end


%% =========================================================================
%  FUNCTION 2: STR CONCENTRATION BAR PLOT
%  =========================================================================

function create_STR_concentration_barplot(output_dir, anatomical_order)
% CREATE_STR_CONCENTRATION_BARPLOT Creates grouped bar plot comparing STR with partners
%
% VISUALIZATION:
% For each STR-target pathway, displays paired bars showing:
%   - STR's encoding concentration when paired with target
%   - Target region's encoding concentration when paired with STR
%
% INPUTS:
%   output_dir       - Directory containing sensitivity_*.mat files
%   anatomical_order - Cell array defining canonical region ordering

    fprintf('Creating STR concentration bar plot...\n');
    
    % Define STR pathways of interest
    STR_pathways = {
        % Tier 1: Cortical → STR pathways
        'mPFC_STR',   'mPFC';
        'ORB_STR',    'ORB';
        'MOp_STR',    'MOp';
        'MOs_STR',    'MOs';
        'OLF_STR',    'OLF';
        % Tier 2: STR ↔ Subcortical pathways
        'STR_STRv',   'STRv';
        'STR_LP',     'LP';
        'STR_VALVM',  'VALVM';
        'STR_VPMPO',  'VPMPO';
        'STR_ILM',    'ILM';
        'STR_HY',     'HY'
    };
    n_pathways = size(STR_pathways, 1);
    
    % Initialize storage
    STR_concentrations = nan(n_pathways, 1);
    target_concentrations = nan(n_pathways, 1);
    STR_se = nan(n_pathways, 1);
    target_se = nan(n_pathways, 1);
    n_sessions_per_pathway = zeros(n_pathways, 1);
    
    % Load sensitivity files
    sensitivity_files = dir(fullfile(output_dir, 'sensitivity_*.mat'));
    
    for f = 1:length(sensitivity_files)
        file_path = fullfile(output_dir, sensitivity_files(f).name);
        data = load(file_path);
        
        if ~isfield(data, 'sensitivity_results')
            continue;
        end
        
        sens = data.sensitivity_results;
        
        if ~isfield(sens, 'sessions') || length(sens.sessions) < 2
            continue;
        end
        
        region1 = sens.region1;
        region2 = sens.region2;
        
        % Identify which pathway this belongs to
        pathway_idx = find_STR_pathway_index(region1, region2, STR_pathways);
        
        if isempty(pathway_idx)
            continue;
        end
        
        % Find 50% removal index
        removal_pct = sens.removal_percentages;
        idx_50 = find(removal_pct >= 50, 1, 'first');
        
        if isempty(idx_50)
            continue;
        end
        
        % Extract R² values at 50% removal
        n_sess = length(sens.sessions);
        STR_vals = zeros(n_sess, 1);
        target_vals = zeros(n_sess, 1);
        
        % Determine which region is STR
        is_region1_STR = contains(region1, 'STR', 'IgnoreCase', true) && ...
                         ~contains(region1, 'STRv', 'IgnoreCase', true);
        
        for s = 1:n_sess
            session = sens.sessions{s};
            if ~isfield(session, 'region1_toprank') || ...
               ~isfield(session, 'region2_toprank')
                continue;
            end
            
            if is_region1_STR
                STR_vals(s) = session.region1_toprank(idx_50);
                target_vals(s) = session.region2_toprank(idx_50);
            else
                STR_vals(s) = session.region2_toprank(idx_50);
                target_vals(s) = session.region1_toprank(idx_50);
            end
        end
        
        % Store statistics
        STR_concentrations(pathway_idx) = mean(STR_vals);
        target_concentrations(pathway_idx) = mean(target_vals);
        STR_se(pathway_idx) = std(STR_vals) / sqrt(n_sess);
        target_se(pathway_idx) = std(target_vals) / sqrt(n_sess);
        n_sessions_per_pathway(pathway_idx) = n_sess;
        
        fprintf('  Pathway %s: STR=%.3f, Target=%.3f (n=%d)\n', ...
            STR_pathways{pathway_idx, 1}, STR_concentrations(pathway_idx), ...
            target_concentrations(pathway_idx), n_sess);
    end
    
    % Create bar plot
    figure('Position', [100, 100, 1200, 600]);
    
    % Filter to valid pathways
    valid_idx = ~isnan(STR_concentrations);
    if ~any(valid_idx)
        fprintf('  Warning: No valid pathway data found\n');
        close(gcf);
        return;
    end
    
    pathway_names = STR_pathways(valid_idx, 1);
    STR_vals_plot = STR_concentrations(valid_idx);
    target_vals_plot = target_concentrations(valid_idx);
    STR_se_plot = STR_se(valid_idx);
    target_se_plot = target_se(valid_idx);
    n_sessions_plot = n_sessions_per_pathway(valid_idx);
    n_valid = sum(valid_idx);
    
    x = 1:n_valid;
    bar_width = 0.35;
    
    % Create grouped bars
    b1 = bar(x - bar_width/2, STR_vals_plot, bar_width, ...
        'FaceColor', [0.8, 0.4, 0.2], 'EdgeColor', 'none');
    hold on;
    b2 = bar(x + bar_width/2, target_vals_plot, bar_width, ...
        'FaceColor', [0.2, 0.4, 0.8], 'EdgeColor', 'none');
    
    % Add error bars
    errorbar(x - bar_width/2, STR_vals_plot, STR_se_plot, ...
        'k', 'LineStyle', 'none', 'LineWidth', 1.5, 'CapSize', 6);
    errorbar(x + bar_width/2, target_vals_plot, target_se_plot, ...
        'k', 'LineStyle', 'none', 'LineWidth', 1.5, 'CapSize', 6);
    
    % Add session count annotations
    y_max = max([STR_vals_plot + STR_se_plot; target_vals_plot + target_se_plot]);
    for i = 1:n_valid
        text(x(i), y_max * 1.05, sprintf('n=%d', n_sessions_plot(i)), ...
            'HorizontalAlignment', 'center', 'FontSize', 12);
    end
    
    % Configure axes
    set(gca, 'XTick', x, 'XTickLabel', pathway_names);
    set(gca, 'FontSize', 14);
    xtickangle(30);
    
    xlabel('Pathway', 'FontSize', 16, 'FontWeight', 'bold');
    ylabel('R^2 at 50% Neuron Removal', 'FontSize', 16, 'FontWeight', 'bold');
    title('STR Pathway Encoding Concentration Comparison', ...
        'FontSize', 18, 'FontWeight', 'bold');
    
    legend([b1, b2], {'STR', 'Partner Region'}, ...
        'Location', 'northeast', 'FontSize', 12);
    
    ylim([0, y_max * 1.15]);
    grid on;
    set(gca, 'GridLineStyle', ':', 'GridAlpha', 0.5);
    ax = gca;
    ax.Box = 'off';
    
    % Save outputs
    saveas(gcf, fullfile(output_dir, 'STR_concentration_barplot.png'));
    saveas(gcf, fullfile(output_dir, 'STR_concentration_barplot.fig'));
    close(gcf);
    
    fprintf('  Bar plot saved to: %s\n\n', output_dir);
end


%% =========================================================================
%  FUNCTION 3: STR PATHWAY BOXPLOT WITH PAIRED STATISTICS
%  =========================================================================



function create_STR_pathway_boxplot(output_dir, anatomical_order)
% CREATE_STR_PATHWAY_BOXPLOT Creates boxplot with paired Wilcoxon signed-rank tests
%
% STATISTICAL FRAMEWORK:
% For each pathway, we test the null hypothesis:
%
%   $H_0: \text{median}(R^2_{\text{STR}}) = \text{median}(R^2_{\text{target}})$
%
% using the Wilcoxon signed-rank test, which is appropriate for paired,
% non-normally distributed observations. Effect sizes are reported as
% median differences.
%
% VISUALIZATION APPROACH:
% Paired session-level observations are connected by lines, enabling
% visualization of individual session trajectories. For session $s$, the
% trajectory is defined as:
%
%   $\mathbf{v}_s = (R^2_{\text{STR},s}, R^2_{\text{partner},s})$
%
% Line slopes reveal systematic encoding differences between regions.
%
% ANATOMICAL ORDERING CONSTRAINT:
% Boxplots are ordered according to the canonical anatomical hierarchy
% provided in anatomical_order. Partner regions appear in their anatomical
% sequence (e.g., rostral-to-caudal for cortical regions, medial-to-lateral
% for thalamic nuclei), ensuring consistent visualization across analyses.
%
% INPUTS:
%   output_dir       - Directory containing sensitivity_*.mat files
%   anatomical_order - Cell array defining canonical region ordering
%                      (e.g., {'mPFC','ORB','MOp','MOs','LP','MD',...})

    fprintf('Creating STR pathway boxplot with paired statistics...\n');
    
    % Define STR pathways
    STR_pathways = {
        % Tier 1: Cortical → STR pathways
        'mPFC_STR',   'mPFC';
        'ORB_STR',    'ORB';
        'MOp_STR',    'MOp';
        'MOs_STR',    'MOs';
        % Tier 2: STR ↔ Subcortical pathways
        'STR_STRv',   'STRv';
        'STR_LP',     'LP';
        'STR_VALVM',  'VALVM';
        'STR_VPMPO',  'VPMPO';
        'STR_ILM',    'ILM';
        'STR_HY',     'HY'
    };
    
    % Initialize storage for session-level data
    pathway_session_data = struct();
    pathway_stats = struct();
    pathway_names = {};
    n_pathways_found = 0;
    
    % Load sensitivity files
    sensitivity_files = dir(fullfile(output_dir, 'sensitivity_*.mat'));
    
    for f = 1:length(sensitivity_files)
        file_path = fullfile(output_dir, sensitivity_files(f).name);
        data = load(file_path);
        
        if ~isfield(data, 'sensitivity_results')
            continue;
        end
        
        sens = data.sensitivity_results;
        
        if ~isfield(sens, 'sessions') || length(sens.sessions) < 3
            continue;
        end
        
        region1 = sens.region1;
        region2 = sens.region2;
        
        % Identify pathway
        pathway_idx = find_STR_pathway_index(region1, region2, STR_pathways);
        
        if isempty(pathway_idx)
            continue;
        end
        
        % Find 50% removal index
        removal_pct = sens.removal_percentages;
        idx_50 = find(removal_pct >= 50, 1, 'first');
        
        if isempty(idx_50)
            continue;
        end
        
        % Extract session-level R² values
        n_sess = length(sens.sessions);
        STR_vals = zeros(n_sess, 1);
        target_vals = zeros(n_sess, 1);
        valid_count = 0;
        
        is_region1_STR = contains(region1, 'STR', 'IgnoreCase', true) && ...
                         ~contains(region1, 'STRv', 'IgnoreCase', true);
        
        for s = 1:n_sess
            session = sens.sessions{s};
            if ~isfield(session, 'region1_toprank') || ...
               ~isfield(session, 'region2_toprank')
                continue;
            end
            
            valid_count = valid_count + 1;
            if is_region1_STR
                STR_vals(valid_count) = session.region1_toprank(idx_50);
                target_vals(valid_count) = session.region2_toprank(idx_50);
            else
                STR_vals(valid_count) = session.region2_toprank(idx_50);
                target_vals(valid_count) = session.region1_toprank(idx_50);
            end
        end
        
        STR_vals = STR_vals(1:valid_count);
        target_vals = target_vals(1:valid_count);
        
        if valid_count < 3
            continue;
        end
        
        % Store pathway data
        n_pathways_found = n_pathways_found + 1;
        pathway_name = STR_pathways{pathway_idx, 2};
        pathway_names{n_pathways_found} = pathway_name;
        
        field_name = sprintf('pathway_%d', n_pathways_found);
        pathway_session_data.(field_name).STR = STR_vals;
        pathway_session_data.(field_name).target = target_vals;
        pathway_session_data.(field_name).name = pathway_name;
        
        % Perform Wilcoxon signed-rank test
        [p_val, ~, wilcox_stats] = signrank(STR_vals, target_vals);
        
        % Calculate descriptive statistics
        stats_struct = struct();
        stats_struct.name = pathway_name;
        stats_struct.n_sessions = valid_count;
        stats_struct.STR_mean = mean(STR_vals);
        stats_struct.STR_std = std(STR_vals);
        stats_struct.STR_median = median(STR_vals);
        stats_struct.target_mean = mean(target_vals);
        stats_struct.target_std = std(target_vals);
        stats_struct.target_median = median(target_vals);
        stats_struct.median_difference = median(STR_vals - target_vals);
        stats_struct.p_value = p_val;
        stats_struct.wilcoxon_stats = wilcox_stats;
        
        pathway_stats.(field_name) = stats_struct;
    end
    
    if n_pathways_found == 0
        fprintf('  Warning: No valid pathway data found for boxplot\n');
        return;
    end
    
    % ========================================================================
    % ANATOMICAL ORDERING: Reorder pathways according to anatomical_order
    % ========================================================================
    [pathway_session_data, pathway_stats, pathway_names] = ...
        reorder_pathways_by_anatomy(pathway_session_data, pathway_stats, ...
                                     pathway_names, anatomical_order);
    
    % Update pathway count after potential reordering
    n_pathways_found = length(pathway_names);
    
    % Create boxplot figure
    figure('Position', [100, 100, 1400, 700]);
    
    % Create positions for paired boxplots
    positions = [];
    tick_positions = [];
    tick_labels = {};
    
    current_pos = 1;
    for p = 1:n_pathways_found
        positions = [positions, current_pos, current_pos + 0.6];
        tick_positions = [tick_positions, current_pos + 0.3];
        tick_labels{p} = pathway_names{p};
        current_pos = current_pos + 2;
    end
    
    % Plot individual boxplots with paired session lines
    hold on;
    
    STR_color = [0.8, 0.4, 0.2];
    target_color = [0.2, 0.4, 0.8];
    line_color = [0.6, 0.6, 0.6];  % Gray for connecting lines
    
    for p = 1:n_pathways_found
        field_name = sprintf('pathway_%d', p);
        pdata = pathway_session_data.(field_name);
        
        STR_pos = positions(2*p - 1);
        target_pos = positions(2*p);
        
        % ====================================================================
        % STEP 1: Draw paired session connecting lines (behind boxplots)
        % ====================================================================
        n_sessions = length(pdata.STR);
        
        for s = 1:n_sessions
            x_coords = [STR_pos, target_pos];
            y_coords = [pdata.STR(s), pdata.target(s)];
            
            plot(x_coords, y_coords, '-', ...
                'Color', [line_color, 0.4], ...  % Semi-transparent gray
                'LineWidth', 1.0);
        end
        
        % ====================================================================
        % STEP 2: Draw boxplots (on top of connecting lines)
        % ====================================================================
        % STR boxplot
        boxplot_single(pdata.STR, STR_pos, 0.4, STR_color);
        
        % Target boxplot
        boxplot_single(pdata.target, target_pos, 0.4, target_color);
        
        % ====================================================================
        % STEP 3: Overlay individual session data points
        % ====================================================================
        % Add small horizontal jitter to prevent overlapping points
        jitter_amount = 0.05;
        
        % STR data points
        x_jitter_STR = STR_pos + (rand(n_sessions, 1) - 0.5) * jitter_amount;
        scatter(x_jitter_STR, pdata.STR, 40, STR_color, 'filled', ...
            'MarkerFaceAlpha', 0.6, 'MarkerEdgeColor', 'k', 'LineWidth', 0.5);
        
        % Target data points
        x_jitter_target = target_pos + (rand(n_sessions, 1) - 0.5) * jitter_amount;
        scatter(x_jitter_target, pdata.target, 40, target_color, 'filled', ...
            'MarkerFaceAlpha', 0.6, 'MarkerEdgeColor', 'k', 'LineWidth', 0.5);
        
        % ====================================================================
        % STEP 4: Add significance annotation
        % ====================================================================
        stats = pathway_stats.(field_name);
        p_val = stats.p_value;
        
        annotation_height = max([pdata.STR; pdata.target]) * 1.15;
        
        % Draw bracket
        plot([STR_pos, target_pos], [annotation_height, annotation_height], ...
            'k-', 'LineWidth', 1.2);
        
        % Add p-value with stars
        text((STR_pos + target_pos)/2, annotation_height * 1.03, ...
            sprintf('p = %.3f%s', p_val, get_significance_stars(p_val)), ...
            'HorizontalAlignment', 'center', 'FontSize', 12, 'FontWeight', 'bold');
        
        % Add sample size
        text((STR_pos + target_pos)/2, annotation_height * 0.93, ...
            sprintf('n = %d', stats.n_sessions), ...
            'HorizontalAlignment', 'center', 'FontSize', 11);
    end
    
    % Configure axes
    set(gca, 'XTick', tick_positions, 'XTickLabel', tick_labels);
    set(gca, 'FontSize', 14);
    
    if n_pathways_found > 3
        xtickangle(35);
    end
    
    xlabel('Partner Region', 'FontSize', 16, 'FontWeight', 'bold');
    ylabel('R^2 at 50% Neuron Removal', 'FontSize', 16, 'FontWeight', 'bold');
    % title('STR Pathway Paired Comparison (Wilcoxon Signed-Rank Test)', ...
    %     'FontSize', 18, 'FontWeight', 'bold');
    
    % Add legend
    h1 = patch(nan, nan, STR_color, 'EdgeColor', 'none');
    h2 = patch(nan, nan, target_color, 'EdgeColor', 'none');
    h3 = plot(nan, nan, '-', 'Color', line_color, 'LineWidth', 1.0);
    legend([h1, h2, h3], {'STR', 'Partner', 'Session Pair'}, ...
        'Location', 'northeast', 'FontSize', 12);
    
    grid on;
    set(gca, 'GridLineStyle', ':', 'GridAlpha', 0.5);
    ax = gca;
    ax.Box = 'off';
    ylim([0,0.5]);
    % Save outputs
    saveas(gcf, fullfile(output_dir, 'STR_pathway_boxplot.png'));
    saveas(gcf, fullfile(output_dir, 'STR_pathway_boxplot.fig'));
    close(gcf);
    
    % Save comprehensive statistics
    comprehensive_stats = struct();
    comprehensive_stats.pathway_session_data = pathway_session_data;
    comprehensive_stats.pathway_stats = pathway_stats;
    comprehensive_stats.pathway_names = pathway_names;
    comprehensive_stats.test_type = 'Wilcoxon signed-rank test (paired observations)';
    comprehensive_stats.anatomical_order_applied = true;
    
    save(fullfile(output_dir, 'STR_pathway_paired_stats.mat'), 'comprehensive_stats');
    
    % Print statistical summary
    fprintf('\n=== STR Pathway Paired Analysis Results ===\n');
    fprintf('Total Pathways: %d (ordered anatomically)\n\n', n_pathways_found);
    
    for p = 1:n_pathways_found
        field_name = sprintf('pathway_%d', p);
        stats = pathway_stats.(field_name);
        fprintf('Pathway: STR ↔ %s (n=%d sessions)\n', stats.name, stats.n_sessions);
        fprintf('  STR:    %.3f ± %.3f (median: %.3f)\n', ...
            stats.STR_mean, stats.STR_std, stats.STR_median);
        fprintf('  %s:  %.3f ± %.3f (median: %.3f)\n', ...
            stats.name, stats.target_mean, stats.target_std, stats.target_median);
        fprintf('  Δ(STR - %s): %.3f\n', stats.name, stats.median_difference);
        fprintf('  Significance: p = %.4f %s\n\n', ...
            stats.p_value, get_significance_stars(stats.p_value));
    end
    
    fprintf('  Boxplot saved to: %s\n\n', output_dir);
end

function [reordered_data, reordered_stats, reordered_names] = ...
    reorder_pathways_by_anatomy(pathway_data, pathway_stats, pathway_names, anatomical_order)
% REORDER_PATHWAYS_BY_ANATOMY Reorders STR pathways according to anatomical hierarchy
%
% MATHEMATICAL FORMULATION:
% Given a set of partner regions $\mathcal{P} = \{P_1, P_2, \ldots, P_n\}$
% and an anatomical ordering $\mathcal{A} = (A_1, A_2, \ldots, A_m)$, we
% define the ordering function:
%
%   $\pi: \mathcal{P} \to \mathbb{N}$ where $\pi(P_i) = \min\{j : P_i \in A_j\}$
%
% Pathways are then sorted such that $\pi(P_i) < \pi(P_j) \implies i < j$
%
% INPUTS:
%   pathway_data     - Structure containing session-level data
%   pathway_stats    - Structure containing statistical summaries
%   pathway_names    - Cell array of partner region names
%   anatomical_order - Cell array defining canonical ordering
%
% OUTPUTS:
%   reordered_data   - Pathway data in anatomical order
%   reordered_stats  - Pathway statistics in anatomical order
%   reordered_names  - Pathway names in anatomical order

    n_pathways = length(pathway_names);
    
    % Create anatomical position mapping
    % $\pi(P_i)$ = position of region $P_i$ in anatomical_order
    anatomical_positions = zeros(n_pathways, 1);
    
    for p = 1:n_pathways
        region_name = pathway_names{p};
        
        % Find position in anatomical_order
        pos = find(strcmpi(anatomical_order, region_name), 1);
        
        if isempty(pos)
            % If not found in anatomical_order, assign large value (place at end)
            anatomical_positions(p) = length(anatomical_order) + p;
            fprintf('  Warning: Region "%s" not found in anatomical_order, placing at end\n', ...
                region_name);
        else
            anatomical_positions(p) = pos;
        end
    end
    
    % Sort pathways by anatomical position
    [~, sort_indices] = sort(anatomical_positions);
    
    % Reorder pathway names
    reordered_names = pathway_names(sort_indices);
    
    % Reorder pathway data and stats
    reordered_data = struct();
    reordered_stats = struct();
    
    for new_idx = 1:n_pathways
        old_idx = sort_indices(new_idx);
        
        old_field = sprintf('pathway_%d', old_idx);
        new_field = sprintf('pathway_%d', new_idx);
        
        reordered_data.(new_field) = pathway_data.(old_field);
        reordered_stats.(new_field) = pathway_stats.(old_field);
    end
    
    fprintf('  Pathways reordered according to anatomical hierarchy:\n');
    for p = 1:n_pathways
        fprintf('    %d. %s\n', p, reordered_names{p});
    end
    fprintf('\n');
end

% function create_STR_pathway_boxplot(output_dir, anatomical_order)
% % CREATE_STR_PATHWAY_BOXPLOT Creates boxplot with paired Wilcoxon signed-rank tests
% %
% % STATISTICAL FRAMEWORK:
% % For each pathway, we test the null hypothesis:
% %
% %   $H_0: \text{median}(R^2_{\text{STR}}) = \text{median}(R^2_{\text{target}})$
% %
% % using the Wilcoxon signed-rank test, which is appropriate for paired,
% % non-normally distributed observations. Effect sizes are reported as
% % median differences.
% %
% % INPUTS:
% %   output_dir       - Directory containing sensitivity_*.mat files
% %   anatomical_order - Cell array defining canonical region ordering
% 
%     fprintf('Creating STR pathway boxplot with paired statistics...\n');
% 
%     % Define STR pathways
%     STR_pathways = {
%         'STR-mPFC', 'mPFC';
%         'STR-ORB',  'ORB';
%         'STR-MOp',  'MOp';
%         'STR-MOs',  'MOs';
%         'STR-LP',   'LP';
%         'STR-MD',   'MD'
%     };
% 
%     % Initialize storage for session-level data
%     pathway_session_data = struct();
%     pathway_stats = struct();
%     pathway_names = {};
%     n_pathways_found = 0;
% 
%     % Load sensitivity files
%     sensitivity_files = dir(fullfile(output_dir, 'sensitivity_*.mat'));
% 
%     for f = 1:length(sensitivity_files)
%         file_path = fullfile(output_dir, sensitivity_files(f).name);
%         data = load(file_path);
% 
%         if ~isfield(data, 'sensitivity_results')
%             continue;
%         end
% 
%         sens = data.sensitivity_results;
% 
%         if ~isfield(sens, 'sessions') || length(sens.sessions) < 3
%             continue;
%         end
% 
%         region1 = sens.region1;
%         region2 = sens.region2;
% 
%         % Identify pathway
%         pathway_idx = find_STR_pathway_index(region1, region2, STR_pathways);
% 
%         if isempty(pathway_idx)
%             continue;
%         end
% 
%         % Find 50% removal index
%         removal_pct = sens.removal_percentages;
%         idx_50 = find(removal_pct >= 50, 1, 'first');
% 
%         if isempty(idx_50)
%             continue;
%         end
% 
%         % Extract session-level R² values
%         n_sess = length(sens.sessions);
%         STR_vals = zeros(n_sess, 1);
%         target_vals = zeros(n_sess, 1);
%         valid_count = 0;
% 
%         is_region1_STR = contains(region1, 'STR', 'IgnoreCase', true) && ...
%                          ~contains(region1, 'STRv', 'IgnoreCase', true);
% 
%         for s = 1:n_sess
%             session = sens.sessions{s};
%             if ~isfield(session, 'region1_toprank') || ...
%                ~isfield(session, 'region2_toprank')
%                 continue;
%             end
% 
%             valid_count = valid_count + 1;
%             if is_region1_STR
%                 STR_vals(valid_count) = session.region1_toprank(idx_50);
%                 target_vals(valid_count) = session.region2_toprank(idx_50);
%             else
%                 STR_vals(valid_count) = session.region2_toprank(idx_50);
%                 target_vals(valid_count) = session.region1_toprank(idx_50);
%             end
%         end
% 
%         STR_vals = STR_vals(1:valid_count);
%         target_vals = target_vals(1:valid_count);
% 
%         if valid_count < 3
%             continue;
%         end
% 
%         % Store pathway data
%         n_pathways_found = n_pathways_found + 1;
%         pathway_name = STR_pathways{pathway_idx, 2};
%         pathway_names{n_pathways_found} = pathway_name;
% 
%         field_name = sprintf('pathway_%d', n_pathways_found);
%         pathway_session_data.(field_name).STR = STR_vals;
%         pathway_session_data.(field_name).target = target_vals;
%         pathway_session_data.(field_name).name = pathway_name;
% 
%         % Perform Wilcoxon signed-rank test
%         [p_val, ~, wilcox_stats] = signrank(STR_vals, target_vals);
% 
%         % Calculate descriptive statistics
%         stats_struct = struct();
%         stats_struct.name = pathway_name;
%         stats_struct.n_sessions = valid_count;
%         stats_struct.STR_mean = mean(STR_vals);
%         stats_struct.STR_std = std(STR_vals);
%         stats_struct.STR_median = median(STR_vals);
%         stats_struct.target_mean = mean(target_vals);
%         stats_struct.target_std = std(target_vals);
%         stats_struct.target_median = median(target_vals);
%         stats_struct.median_difference = median(STR_vals - target_vals);
%         stats_struct.p_value = p_val;
%         stats_struct.wilcoxon_stats = wilcox_stats;
% 
%         pathway_stats.(field_name) = stats_struct;
%     end
% 
%     if n_pathways_found == 0
%         fprintf('  Warning: No valid pathway data found for boxplot\n');
%         return;
%     end
% 
%     % Create boxplot figure
%     figure('Position', [100, 100, 1400, 700]);
% 
%     % Prepare data for boxplot
%     all_concentrations = [];
%     group_labels = {};
%     pathway_labels = {};
% 
%     for p = 1:n_pathways_found
%         field_name = sprintf('pathway_%d', p);
%         pdata = pathway_session_data.(field_name);
% 
%         % Add STR data
%         all_concentrations = [all_concentrations; pdata.STR];
%         n_STR = length(pdata.STR);
%         group_labels = [group_labels; repmat({'STR'}, n_STR, 1)];
%         pathway_labels = [pathway_labels; repmat({pdata.name}, n_STR, 1)];
% 
%         % Add target data
%         all_concentrations = [all_concentrations; pdata.target];
%         n_target = length(pdata.target);
%         group_labels = [group_labels; repmat({pdata.name}, n_target, 1)];
%         pathway_labels = [pathway_labels; repmat({pdata.name}, n_target, 1)];
%     end
% 
%     % Create positions for paired boxplots
%     positions = [];
%     tick_positions = [];
%     tick_labels = {};
% 
%     current_pos = 1;
%     for p = 1:n_pathways_found
%         positions = [positions, current_pos, current_pos + 0.6];
%         tick_positions = [tick_positions, current_pos + 0.3];
%         tick_labels{p} = pathway_names{p};
%         current_pos = current_pos + 2;
%     end
% 
%     % Plot individual boxplots
%     hold on;
% 
%     STR_color = [0.8, 0.4, 0.2];
%     target_color = [0.2, 0.4, 0.8];
% 
%     for p = 1:n_pathways_found
%         field_name = sprintf('pathway_%d', p);
%         pdata = pathway_session_data.(field_name);
% 
%         STR_pos = positions(2*p - 1);
%         target_pos = positions(2*p);
% 
%         % STR boxplot
%         boxplot_single(pdata.STR, STR_pos, 0.4, STR_color);
% 
%         % Target boxplot
%         boxplot_single(pdata.target, target_pos, 0.4, target_color);
% 
%         % Add significance annotation
%         stats = pathway_stats.(field_name);
%         p_val = stats.p_value;
% 
%         annotation_height = max([pdata.STR; pdata.target]) * 1.15;
% 
%         % Draw bracket
%         plot([STR_pos, target_pos], [annotation_height, annotation_height], ...
%             'k-', 'LineWidth', 1.2);
% 
%         % Add p-value with stars
%         text((STR_pos + target_pos)/2, annotation_height * 1.03, ...
%             sprintf('p = %.3f%s', p_val, get_significance_stars(p_val)), ...
%             'HorizontalAlignment', 'center', 'FontSize', 12, 'FontWeight', 'bold');
% 
%         % Add sample size
%         text((STR_pos + target_pos)/2, annotation_height * 0.93, ...
%             sprintf('n = %d', stats.n_sessions), ...
%             'HorizontalAlignment', 'center', 'FontSize', 11);
%     end
% 
%     % Configure axes
%     set(gca, 'XTick', tick_positions, 'XTickLabel', tick_labels);
%     set(gca, 'FontSize', 14);
% 
%     if n_pathways_found > 3
%         xtickangle(35);
%     end
% 
%     xlabel('Partner Region', 'FontSize', 16, 'FontWeight', 'bold');
%     ylabel('R^2 at 50% Neuron Removal', 'FontSize', 16, 'FontWeight', 'bold');
%     % title('STR Pathway Paired Comparison (Wilcoxon Signed-Rank Test)', ...
%     %     'FontSize', 18, 'FontWeight', 'bold');
% 
%     % Add legend
%     h1 = patch(nan, nan, STR_color, 'EdgeColor', 'none');
%     h2 = patch(nan, nan, target_color, 'EdgeColor', 'none');
%     legend([h1, h2], {'STR', 'Partner'}, 'Location', 'northeast', 'FontSize', 12);
% 
%     grid on;
%     set(gca, 'GridLineStyle', ':', 'GridAlpha', 0.5);
%     ax = gca;
%     ax.Box = 'off';
% 
%     % Save outputs
%     saveas(gcf, fullfile(output_dir, 'STR_pathway_boxplot.png'));
%     saveas(gcf, fullfile(output_dir, 'STR_pathway_boxplot.fig'));
%     close(gcf);
% 
%     % Save comprehensive statistics
%     comprehensive_stats = struct();
%     comprehensive_stats.pathway_session_data = pathway_session_data;
%     comprehensive_stats.pathway_stats = pathway_stats;
%     comprehensive_stats.pathway_names = pathway_names;
%     comprehensive_stats.test_type = 'Wilcoxon signed-rank test (paired observations)';
% 
%     save(fullfile(output_dir, 'STR_pathway_paired_stats.mat'), 'comprehensive_stats');
% 
%     % Print statistical summary
%     fprintf('\n=== STR Pathway Paired Analysis Results ===\n');
%     fprintf('Total Pathways: %d\n\n', n_pathways_found);
% 
%     for p = 1:n_pathways_found
%         field_name = sprintf('pathway_%d', p);
%         stats = pathway_stats.(field_name);
%         fprintf('Pathway: STR ↔ %s (n=%d sessions)\n', stats.name, stats.n_sessions);
%         fprintf('  STR:    %.3f ± %.3f (median: %.3f)\n', ...
%             stats.STR_mean, stats.STR_std, stats.STR_median);
%         fprintf('  %s:  %.3f ± %.3f (median: %.3f)\n', ...
%             stats.name, stats.target_mean, stats.target_std, stats.target_median);
%         fprintf('  Δ(STR - %s): %.3f\n', stats.name, stats.median_difference);
%         fprintf('  Significance: p = %.4f %s\n\n', ...
%             stats.p_value, get_significance_stars(stats.p_value));
%     end
% 
%     fprintf('  Boxplot saved to: %s\n\n', output_dir);
% end


%% =========================================================================
%  HELPER FUNCTIONS
%  =========================================================================

function boxplot_single(data, position, width, color)
% BOXPLOT_SINGLE Creates a single boxplot at specified position
%
% VISUALIZATION:
% Renders box (IQR), whiskers (1.5×IQR), median line, and outliers
%
% INPUTS:
%   data     - Vector of observations
%   position - X-axis position for boxplot
%   width    - Width of the box
%   color    - RGB color vector [r, g, b]

    q1 = quantile(data, 0.25);
    q2 = median(data);
    q3 = quantile(data, 0.75);
    iqr_val = q3 - q1;
    
    whisker_low = max(min(data), q1 - 1.5 * iqr_val);
    whisker_high = min(max(data), q3 + 1.5 * iqr_val);
    
    % Draw box
    rectangle('Position', [position - width/2, q1, width, q3 - q1], ...
        'FaceColor', color, 'EdgeColor', 'k', 'LineWidth', 1);
    
    % Draw median
    plot([position - width/2, position + width/2], [q2, q2], ...
        'k-', 'LineWidth', 2);
    
    % Draw whiskers
    plot([position, position], [q1, whisker_low], 'k-', 'LineWidth', 1);
    plot([position, position], [q3, whisker_high], 'k-', 'LineWidth', 1);
    plot([position - width/4, position + width/4], [whisker_low, whisker_low], ...
        'k-', 'LineWidth', 1);
    plot([position - width/4, position + width/4], [whisker_high, whisker_high], ...
        'k-', 'LineWidth', 1);
    
    % Draw outliers
    outliers = data(data < whisker_low | data > whisker_high);
    if ~isempty(outliers)
        scatter(repmat(position, length(outliers), 1), outliers, 30, 'k', 'filled');
    end
end


function [is_STR_pair, idx1, idx2] = check_STR_pairing(region1, region2, all_regions)
% CHECK_STR_PAIRING Determines if region pair involves STR
%
% CRITICAL: Must distinguish STR from STRv (ventral striatum)
%
% INPUTS:
%   region1, region2 - Region identifiers from sensitivity file
%   all_regions      - Cell array of regions to match against
%
% OUTPUTS:
%   is_STR_pair - Boolean indicating STR involvement
%   idx1, idx2  - Indices in all_regions array

    is_STR_pair = false;
    idx1 = [];
    idx2 = [];
    
    % Check if either region contains STR (but not STRv)
    contains_STR_1 = contains(region1, 'STR', 'IgnoreCase', true) && ...
                     ~contains(region1, 'STRv', 'IgnoreCase', true);
    contains_STR_2 = contains(region2, 'STR', 'IgnoreCase', true) && ...
                     ~contains(region2, 'STRv', 'IgnoreCase', true);
    
    if ~contains_STR_1 && ~contains_STR_2
        return;
    end
    
    % Find indices in all_regions
    idx1 = find_region_index(region1, all_regions);
    idx2 = find_region_index(region2, all_regions);
    
    if ~isempty(idx1) && ~isempty(idx2)
        is_STR_pair = true;
    end
end


function idx = find_region_index(region_name, all_regions)
% FIND_REGION_INDEX Locates region in standardized region list
%
% Handles naming variations and abbreviations through:
%   1. Direct string matching
%   2. Special STR handling (distinguishing from STRv)
%   3. Substring containment matching
%
% INPUTS:
%   region_name - Region identifier string
%   all_regions - Cell array of canonical region names
%
% OUTPUTS:
%   idx - Index in all_regions, or empty if not found

    idx = [];
    
    % Direct match
    for i = 1:length(all_regions)
        if strcmpi(region_name, all_regions{i})
            idx = i;
            return;
        end
    end
    
    % Special handling for STR (exclude STRv)
    if contains(region_name, 'STR', 'IgnoreCase', true) && ...
       ~contains(region_name, 'STRv', 'IgnoreCase', true)
        % Find exact 'STR' in all_regions
        for i = 1:length(all_regions)
            if strcmpi(all_regions{i}, 'STR')
                idx = i;
                return;
            end
        end
    end
    
    % Partial match (substring search)
    for i = 1:length(all_regions)
        if contains(region_name, all_regions{i}, 'IgnoreCase', true) || ...
           contains(all_regions{i}, region_name, 'IgnoreCase', true)
            idx = i;
            return;
        end
    end
end


function idx = find_STR_pathway_index(region1, region2, STR_pathways)
% FIND_STR_PATHWAY_INDEX Identifies which STR pathway a region pair represents
%
% INPUTS:
%   region1, region2 - Region identifiers
%   STR_pathways     - Cell array {pathway_name, short_name; ...}
%
% OUTPUTS:
%   idx - Row index in STR_pathways, or empty if not an STR pathway

    idx = [];
    
    for i = 1:size(STR_pathways, 1)
        pathway_short = STR_pathways{i, 2};
        
        % Check if one region is STR (not STRv) and other is target
        is_region1_STR = contains(region1, 'STR', 'IgnoreCase', true) && ...
                         ~contains(region1, 'STRv', 'IgnoreCase', true);
        is_region2_STR = contains(region2, 'STR', 'IgnoreCase', true) && ...
                         ~contains(region2, 'STRv', 'IgnoreCase', true);
        
        if (is_region1_STR && contains(region2, pathway_short, 'IgnoreCase', true)) || ...
           (is_region2_STR && contains(region1, pathway_short, 'IgnoreCase', true))
            idx = i;
            return;
        end
    end
end


function stars = get_significance_stars(p_value)
% GET_SIGNIFICANCE_STARS Converts p-values to conventional notation
%
% THRESHOLDS (following standard conventions):
%   *** : $p < 0.001$
%   **  : $p < 0.01$
%   *   : $p < 0.05$
%   n.s.: $p \geq 0.05$ (not significant)
%
% INPUTS:
%   p_value - Statistical test p-value
%
% OUTPUTS:
%   stars - String with significance annotation

    if p_value < 0.001
        stars = ' ***';
    elseif p_value < 0.01
        stars = ' **';
    elseif p_value < 0.05
        stars = ' *';
    else
        stars = ' n.s.';
    end
end


%% =========================================================================
%  SCRIPT EXECUTION
%  =========================================================================
% Uncomment the line below to run the analysis
% oxford_glm_summary_analysis_STR()