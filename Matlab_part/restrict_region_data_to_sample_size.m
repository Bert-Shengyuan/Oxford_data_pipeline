function kernel_data = restrict_region_data_to_sample_size(region_data, config)
% RESTRICT_REGION_DATA_TO_SAMPLE_SIZE - Apply the neuron-count exclusion for kernels
%
% region_data (from perform_region_analysis.m) keeps every brain region. This
% function builds the view used by PCA / CCA / pCCA / tkCCA: regions with
% fewer than config.min_neurons_per_region neurons are dropped, each remaining
% region is restricted to min(config.target_neurons, n_neurons) sampled
% neurons (.selected_neurons), and .region_pairs is rebuilt over the kept
% regions. The saved region_data itself is not modified.

    kernel_data = region_data;
    kernel_data.regions = struct();
    kernel_data.valid_regions = {};

    fprintf('  Restricting regions to sample size (min %d, target %d neurons)...\n', ...
            config.min_neurons_per_region, config.target_neurons);

    for i = 1:numel(region_data.valid_regions)
        region_name = region_data.valid_regions{i};
        region = region_data.regions.(region_name);
        n_neurons = size(region.spike_data, 2);

        if n_neurons < config.min_neurons_per_region
            fprintf('    ✗ %s excluded (%d < %d neurons)\n', ...
                    region_name, n_neurons, config.min_neurons_per_region);
            continue;
        end

        % Reuse the stored sample when it matches the requested size;
        % otherwise resample with the same seed as perform_region_analysis.
        target_neurons = min(config.target_neurons, n_neurons);
        if ~isfield(region, 'selected_neurons') || numel(region.selected_neurons) ~= target_neurons
            rng(12345, 'twister');
            region.selected_neurons = randperm(n_neurons, target_neurons);
        end
        region.target_neurons = target_neurons;
        region.original_neurons = n_neurons;

        kernel_data.regions.(region_name) = region;
        kernel_data.valid_regions{end+1} = region_name;
        fprintf('    ✓ %s: %d → %d neurons\n', region_name, n_neurons, target_neurons);
    end

    n_regions = numel(kernel_data.valid_regions);
    kernel_data.region_pairs = zeros(0, 2);
    for i = 1:n_regions
        for j = i+1:n_regions
            kernel_data.region_pairs(end+1, :) = [i, j];
        end
    end

    fprintf('  Kernel regions: %d of %d (%d pairs)\n', n_regions, ...
            numel(region_data.valid_regions), size(kernel_data.region_pairs, 1));
end
