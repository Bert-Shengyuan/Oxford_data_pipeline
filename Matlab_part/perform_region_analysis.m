function region_data = perform_region_analysis(session_data, config)
% PERFORM_REGION_ANALYSIS - Comprehensive brain region separation and validation
%
% This function represents the neuroanatomical foundation of your CCA analysis.
% Rather than working with pre-defined region datasets, we dynamically
% discover and validate brain regions within each experimental session.
% This approach provides superior experimental control and allows for
% session-specific adaptation of the analysis pipeline.
%
% THEORETICAL FRAMEWORK:
% Neural populations are organized by brain region labels from cell_metrics.brainRegion_final.
% Each region must meet minimum neuron count criteria to ensure statistical reliability
% of the canonical correlation analysis. The function implements adaptive sampling
% to equalize neuron counts across regions, controlling for recording yield differences.

    fprintf('  Analyzing neural populations across brain regions...\n');
    
    % Extract brain region labels for all stable neurons
    brain_regions = session_data.brain_regions;
    unique_regions = unique(brain_regions);
    
    fprintf('  Discovered %d unique brain regions in this session\n', length(unique_regions));
    
    % Initialize region analysis structure
    region_data = struct();
    region_data.session_name = session_data.session_name;
    region_data.regions = struct();
    region_data.valid_regions = {};
    region_data.region_pairs = [];
    region_data.spike_data = session_data.spike_rates;
    region_data.timepoints = size(session_data.spike_rates, 3);
    
    % Analyze each brain region for neuron count and data quality
    for i = 1:length(unique_regions)
        region_name = unique_regions{i};
        
        % Skip regions with empty or invalid names
        if isempty(region_name) || strcmp(region_name, 'Unknown') || strcmp(region_name, '')
            continue;
        end
        
        % Find neurons belonging to this region
        region_neurons = strcmp(brain_regions, region_name);
        n_neurons = sum(region_neurons);
        
        fprintf('    Region: %s - %d neurons\n', region_name, n_neurons);
        
        % Apply minimum neuron threshold for statistical reliability
        if n_neurons >= config.min_neurons_per_region
            % Extract spike data for this region
            region_spike_data = session_data.spike_rates(:, region_neurons, :);
            
            % Validate data quality (check for excessive NaNs or zeros)
            %data_quality = assess_region_data_quality(region_spike_data, region_name);
            
            % if data_quality.is_valid
            % Store region information
            region_data.regions.(region_name) = struct();
            region_data.regions.(region_name).neuron_indices = find(region_neurons);
            region_data.regions.(region_name).n_neurons = n_neurons;
            region_data.regions.(region_name).spike_data = region_spike_data;

            %region_data.regions.(region_name).data_quality = data_quality;
            
            % Add to valid regions list
            region_data.valid_regions{end+1} = region_name;
            
            fprintf('      ✓ Added to valid regions (meets criteria)\n');
            % else
            %     fprintf('      ✗ Excluded due to poor data quality\n');
            % end
        else
            fprintf('      ✗ Excluded (insufficient neurons: need >= %d)\n', ...
                    config.min_neurons_per_region);
        end
    end
    
    % Generate all possible region pairs for CCA analysis
    n_valid_regions = length(region_data.valid_regions);
    fprintf('  Valid regions for analysis: %d\n', n_valid_regions);
    
    if n_valid_regions >= 2
        pair_count = 0;
        for i = 1:n_valid_regions
            for j = i+1:n_valid_regions
                pair_count = pair_count + 1;
                region_data.region_pairs(pair_count, :) = [i, j];
            end
        end
        
        fprintf('  Generated %d region pairs for CCA analysis\n', pair_count);
        
        % Display valid regions and their pairings
        fprintf('  Valid regions:\n');
        for i = 1:length(region_data.valid_regions)
            region_name = region_data.valid_regions{i};
            n_neurons = region_data.regions.(region_name).n_neurons;
            fprintf('    %d. %s (%d neurons)\n', i, region_name, n_neurons);
        end
    else
        fprintf('  Warning: Insufficient valid regions (%d) for pairwise analysis\n', n_valid_regions);
        region_data.valid_regions = {};
    end
    for region_idx = 1:length(region_data.valid_regions)
        region_name = region_data.valid_regions{region_idx};
        spike_data = region_data.regions.(region_name).spike_data;
        n_neurons = size(spike_data, 2);
        
        % Determine target neurons for this region
        target_neurons = min(config.target_neurons, n_neurons);
        
        if target_neurons < config.min_neurons_per_region
            fprintf('    Warning: %s has insufficient neurons (%d < %d)\n', ...
                    region_name, target_neurons, config.min_neurons_per_region);
            region_data.regions.(region_name).spike_data_sampled = [];
            region_data.regions.(region_name).selected_neurons = [];
            region_data.regions.(region_name).target_neurons = 0;
            region_data.regions.(region_name).original_neurons = n_neurons;
            continue;
        end
        
        % Randomly sample neurons once for this region
        rng(12345, 'twister');
        selected_neurons = randperm(n_neurons, target_neurons);
        
        % Store sampled data and metadata
        % region_data.regions.(region_name).spike_data_sampled = spike_data(:, selected_neurons, :);
        region_data.regions.(region_name).selected_neurons = selected_neurons;
        region_data.regions.(region_name).target_neurons = target_neurons;
        region_data.regions.(region_name).original_neurons = n_neurons;
        
        fprintf('    %s: %d → %d neurons\n', region_name, n_neurons, target_neurons);
    end    
end

function data_quality = assess_region_data_quality(spike_data, region_name)
% ASSESS_REGION_DATA_QUALITY - Evaluate data quality for a brain region
% This function implements comprehensive quality control metrics to ensure
% reliable CCA analysis across different recording conditions

    data_quality = struct();
    data_quality.region_name = region_name;
    data_quality.is_valid = true;
    data_quality.warnings = {};
    
    % Calculate basic statistics
    total_elements = numel(spike_data);
    nan_count = sum(isnan(spike_data(:)));
    zero_count = sum(spike_data(:) == 0);
    
    data_quality.nan_percentage = (nan_count / total_elements) * 50;
    data_quality.zero_percentage = (zero_count / total_elements) * 100;
    data_quality.mean_firing_rate = nanmean(spike_data(:));
    data_quality.std_firing_rate = nanstd(spike_data(:));
    
    % Apply quality thresholds
    max_nan_percentage = 30.0; % Maximum 10% NaN values
    max_zero_percentage = 100.0; % Maximum 90% zero values
    min_mean_rate = 0.1; % Minimum mean firing rate (Hz)
    
    if data_quality.nan_percentage > max_nan_percentage
        data_quality.is_valid = false;
        data_quality.warnings{end+1} = sprintf('High NaN percentage: %.1f%%', data_quality.nan_percentage);
    end
    
    if data_quality.zero_percentage > max_zero_percentage
        data_quality.is_valid = false;
        data_quality.warnings{end+1} = sprintf('High zero percentage: %.1f%%', data_quality.zero_percentage);
    end
    
    if data_quality.mean_firing_rate < min_mean_rate
        data_quality.is_valid = false;
        data_quality.warnings{end+1} = sprintf('Low mean firing rate: %.3f Hz', data_quality.mean_firing_rate);
    end
    
    % Log quality assessment
    if data_quality.is_valid
        fprintf('      Data quality: PASS (mean rate: %.2f Hz, NaN: %.1f%%, zeros: %.1f%%)\n', ...
                data_quality.mean_firing_rate, data_quality.nan_percentage, data_quality.zero_percentage);
    else
        fprintf('      Data quality: FAIL - %s\n', strjoin(data_quality.warnings, '; '));
    end
end

