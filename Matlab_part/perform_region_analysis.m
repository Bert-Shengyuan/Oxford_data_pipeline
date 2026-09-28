function region_data = perform_region_analysis(session_data, config)
% PERFORM_REGION_ANALYSIS - Comprehensive brain region separation
%
% This function represents the neuroanatomical foundation of your CCA analysis.
% Rather than working with pre-defined region datasets, we dynamically
% discover brain regions within each experimental session.
%
% THEORETICAL FRAMEWORK:
% Neural populations are organized by brain region labels from cell_metrics.brainRegion_final.
% EVERY named region is kept in region_data, regardless of its neuron count.
% The minimum-neuron exclusion is NOT applied here; it is applied later, and
% only when a PCA/kernel computation is requested, by
% restrict_region_data_to_sample_size.m.
%
% Per-region fields:
%   .neuron_indices, .n_neurons, .spike_data, .subregion_labels (if available)
%   .meets_min_neurons  - n_neurons >= config.min_neurons_per_region
%   .selected_neurons   - regions meeting the threshold: random sample of
%                         min(target_neurons, n_neurons) neurons (rng 12345);
%                         regions below it: all neurons (1:n_neurons)
%   .target_neurons     - numel(selected_neurons)
%   .original_neurons   - n_neurons

    fprintf('  Analyzing neural populations across brain regions...\n');
    
    % Extract brain region labels for all stable neurons
    brain_regions = session_data.brain_regions;
    unique_regions = unique(brain_regions);

    % Subregion labels (finer-grained than brain_regions), if available
    if isfield(session_data, 'subregions')
        subregions = session_data.subregions;
    else
        subregions = [];
    end
    
    fprintf('  Discovered %d unique brain regions in this session\n', length(unique_regions));
    
    % Initialize region analysis structure
    region_data = struct();
    region_data.session_name = session_data.session_name;
    region_data.regions = struct();
    region_data.valid_regions = {};
    region_data.region_pairs = [];
    region_data.spike_data = session_data.spike_rates;
    region_data.timepoints = size(session_data.spike_rates, 3);
    % Marks region_data built with every region kept (no neuron-count
    % exclusion); caches without this flag were built by the old filtered code.
    region_data.all_regions_kept = true;
    region_data.min_neurons_per_region = config.min_neurons_per_region;
    region_data.target_neurons = config.target_neurons;
    
    % Store every named brain region
    for i = 1:length(unique_regions)
        region_name = unique_regions{i};
        
        % Skip regions with empty or invalid names
        if isempty(region_name) || strcmp(region_name, 'Unknown') || strcmp(region_name, '')
            continue;
        end
        
        % Find neurons belonging to this region
        region_neurons = strcmp(brain_regions, region_name);
        n_neurons = sum(region_neurons);
        meets_min = n_neurons >= config.min_neurons_per_region;
        
        region_data.regions.(region_name) = struct();
        region_data.regions.(region_name).neuron_indices = find(region_neurons);
        region_data.regions.(region_name).n_neurons = n_neurons;
        region_data.regions.(region_name).spike_data = session_data.spike_rates(:, region_neurons, :);
        if ~isempty(subregions)
            region_data.regions.(region_name).subregion_labels = subregions(region_neurons);
        end
        region_data.regions.(region_name).meets_min_neurons = meets_min;

        if meets_min
            % Randomly sample neurons once for this region (used by PCA/kernels)
            target_neurons = min(config.target_neurons, n_neurons);
            rng(12345, 'twister');
            selected_neurons = randperm(n_neurons, target_neurons);
            fprintf('    Region: %s - %d neurons (sampled %d for kernels)\n', ...
                    region_name, n_neurons, target_neurons);
        else
            % Below the sample size: keep all neurons; excluded only at kernel time
            selected_neurons = 1:n_neurons;
            fprintf('    Region: %s - %d neurons (kept; below kernel threshold %d)\n', ...
                    region_name, n_neurons, config.min_neurons_per_region);
        end

        region_data.regions.(region_name).selected_neurons = selected_neurons;
        region_data.regions.(region_name).target_neurons = numel(selected_neurons);
        region_data.regions.(region_name).original_neurons = n_neurons;

        region_data.valid_regions{end+1} = region_name;
    end
    
    % All pairs over every kept region; restrict_region_data_to_sample_size
    % rebuilds the pairs over the eligible subset for kernel analyses.
    region_data.region_pairs = build_region_pairs(numel(region_data.valid_regions));

    n_eligible = sum(cellfun(@(r) region_data.regions.(r).meets_min_neurons, region_data.valid_regions));
    fprintf('  Regions kept: %d (%d meet the %d-neuron kernel threshold)\n', ...
            numel(region_data.valid_regions), n_eligible, config.min_neurons_per_region);
end

function pairs = build_region_pairs(n_regions)
    pairs = zeros(0, 2);
    for i = 1:n_regions
        for j = i+1:n_regions
            pairs(end+1, :) = [i, j]; %#ok<AGROW>
        end
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

