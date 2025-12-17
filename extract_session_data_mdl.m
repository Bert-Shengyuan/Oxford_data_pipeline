function session_data = extract_session_data_mdl(session_id, date_str, config, t_approach)
% EXTRACT_SESSION_DATA_MDL - Extract and preprocess neural data from MDL format
%
% This function implements the updated data extraction pipeline for the Oxford dataset,
% handling the transition from pre-segmented trial files to continuous MDL data with
% behavioral alignment via t_approach timestamps.
%
% PIPELINE STAGES:
% 1. Load MDL data containing continuous firing rates (mdl.predictor.firingrate)
% 2. Load cell metrics to obtain brain region assignments (brainRegion_final)
% 3. Segment continuous data into trials using t_approach timestamps
% 4. Apply quality control filtering (stable units, valid trials)
% 5. Construct unified session data structure for downstream CCA/PCA analysis
%
% DATA TRANSFORMATION:
% The MDL firing rate matrix F ∈ ℝ^{N_neurons × T_total} is transformed to
% F_trial ∈ ℝ^{N_trials × N_neurons × N_timepoints} using t_approach alignment.
%
% INPUTS:
%   session_id  - Animal identifier (e.g., 'yp020')
%   date_str    - Recording date in YYMMDD format (e.g., '220401')
%   config      - Configuration struct containing:
%                 .local_base_dir - Base directory for downloaded data
%                 .time_window    - [start, end] in seconds (default: [-1.5, 3.0])
%                 .min_neurons_per_region - Minimum neurons for valid region
%   t_approach  - Table from get_tapproach.m with columns including:
%                 {animal_id, session_date, session_name, ..., start_time, label}
%
% OUTPUTS:
%   session_data - Struct containing:
%                  .session_name    - Session identifier string
%                  .spike_rates     - [N_trials × N_neurons × N_timepoints] tensor
%                  .brain_regions   - Cell array of region names per neuron
%                  .cell_metrics    - Combined cell metrics from all probes
%                  .trial_info      - Trial metadata from segmentation
%                  .n_trials        - Number of valid trials
%                  .n_neurons       - Number of neurons (before region filtering)
%
% EXAMPLE USAGE:
%   config.local_base_dir = '/Users/shengyuancai/Downloads/Oxford_dataset/';
%   config.time_window = [-1.5, 3.0];
%   t_approach = load('t_approach.mat').t_approach;
%   session_data = extract_session_data_mdl('yp020', '220401', config, t_approach);
%
% NOTES:
% - Requires MDL file: {session}.mdl.mat in session directory
% - Requires cell metrics: {session}_npxX.cell_metrics.cellinfo.mat in probe subdirs
% - Trial selection based on label == 'cued hit long' in t_approach

    session_name = sprintf('%s_%s', session_id, date_str);
    session_path = fullfile(config.local_base_dir, 'proc', session_id, session_name);
    
    fprintf('  Extracting MDL data from: %s\n', session_path);
    
    try
        %% Stage 1: Load MDL Data
        % The MDL file contains continuous firing rates without trial structure
        mdl_file = fullfile(session_path, sprintf('%s.mdl.mat', session_name));
        
        if ~exist(mdl_file, 'file')
            fprintf('  Error: MDL file not found: %s\n', mdl_file);
            session_data = [];
            return;
        end
        
        fprintf('  Loading MDL data...\n');
        mdl_data = load(mdl_file);
        
        % Validate MDL data structure
        if isfield(mdl_data, 'mdl')
            mdl_struct = mdl_data.mdl;
        else
            % Assume the loaded data IS the mdl struct
            mdl_struct = mdl_data;
        end
        
        if ~isfield(mdl_struct, 'predictor') || ~isfield(mdl_struct.predictor, 'firingrate')
            fprintf('  Error: mdl.predictor.firingrate field not found\n');
            session_data = [];
            return;
        end
        
        [n_neurons_mdl, n_total_bins] = size(mdl_struct.predictor.firingrate);
        fprintf('  MDL firing rate dimensions: %d neurons × %d bins\n', n_neurons_mdl, n_total_bins);
        
        %% Stage 2: Load and Combine Cell Metrics
        % Cell metrics provide brain region assignments essential for CCA analysis
        
        % Find all Neuropixels probe directories
        npx_pattern = fullfile(session_path, '*_npx*');
        npx_dirs = dir(npx_pattern);
        npx_dirs = npx_dirs([npx_dirs.isdir]);
        
        if isempty(npx_dirs)
            fprintf('  Error: No Neuropixels directories found\n');
            session_data = [];
            return;
        end
        
        fprintf('  Found %d Neuropixels probes\n', length(npx_dirs));
        
        % Prepare paths for Cell Explorer batch loading
        probe_paths = cell(length(npx_dirs), 1);
        for i = 1:length(npx_dirs)
            probe_paths{i} = fullfile(session_path, npx_dirs(i).name);
        end
        
        fprintf('  Loading and combining cell metrics from all probes...\n');
        
        % Attempt to use Cell Explorer's batch loading function
        try
            cell_metrics = loadCellMetricsBatch('basepaths', probe_paths);
            fprintf('  Successfully loaded cell metrics via loadCellMetricsBatch\n');
        catch ME
            fprintf('  Warning: loadCellMetricsBatch failed: %s\n', ME.message);
            fprintf('  Attempting manual cell metrics combination...\n');
            
            % Fallback: manually load and combine cell metrics files
            cell_metrics = load_and_combine_cell_metrics(probe_paths, session_name);
            
            if isempty(cell_metrics)
                fprintf('  Error: Could not load cell metrics from any probe\n');
                session_data = [];
                return;
            end
        end
        
        % Validate cell metrics contain required fields
        if ~isfield(cell_metrics, 'brainRegion_final')
            % Try alternative field names
            if isfield(cell_metrics, 'brainRegion')
                cell_metrics.brainRegion_final = cell_metrics.brainRegion;
                fprintf('  Using brainRegion field as brainRegion_final\n');
            else
                fprintf('  Error: brainRegion_final field not found in cell metrics\n');
                session_data = [];
                return;
            end
        end
        
        n_neurons_cm = length(cell_metrics.brainRegion_final);
        fprintf('  Cell metrics contain %d neurons\n', n_neurons_cm);
        
        % Validate neuron count consistency between MDL and cell metrics
        if n_neurons_mdl ~= n_neurons_cm
            warning('Neuron count mismatch: MDL has %d, cell metrics has %d', n_neurons_mdl, n_neurons_cm);
            % Use the minimum to be safe
            n_neurons = min(n_neurons_mdl, n_neurons_cm);
            fprintf('  Using %d neurons (minimum of both sources)\n', n_neurons);
        else
            n_neurons = n_neurons_mdl;
        end
        
        %% Stage 3: Segment Continuous Data into Trials
        % Use t_approach timestamps to extract trial epochs from continuous data
        
        fprintf('  Segmenting continuous data using t_approach...\n');
        
        [trial_firing_rates, trial_info] = segment_mdl_to_trials(mdl_data, t_approach, ...
                                                                  session_id, date_str, config);
        
        if isempty(trial_firing_rates) || trial_info.n_trials == 0
            fprintf('  Error: No valid trials could be segmented\n');
            session_data = [];
            return;
        end
        
        fprintf('  Segmented %d trials successfully\n', trial_info.n_trials);
        
        %% Stage 4: Quality Control - Apply Stable Unit Filtering (if available)
        % Note: With MDL format, stable_unit information may be in cell_metrics or mdl
        
        stable_units = [];
        predictor_part = mdl_struct.predictor;
        % Check for stable_unit in cell_metrics
        if isfield(predictor_part, 'stable_unit')
            stable_units = predictor_part.stable_unit;
            fprintf('  Found stable_unit field in MDL structure\n');
        else
            fprintf('  Error! Can find stable unit label!\n');
        end
        
        if ~isempty(stable_units)
            if islogical(stable_units)
                stable_mask = stable_units(:);
            else
                stable_mask = stable_units(:) == 1;
            end
            
            % Ensure mask length matches neuron count
            if length(stable_mask) > n_neurons
                stable_mask = stable_mask(1:n_neurons);
            elseif length(stable_mask) < n_neurons
                % Pad with true (assume stable if not specified)
                stable_mask = [stable_mask; true(n_neurons - length(stable_mask), 1)];
            end
            
            n_stable = sum(stable_mask);
            fprintf('  Stable units: %d / %d (%.1f%%)\n', n_stable, n_neurons, 100*n_stable/n_neurons);
            
            % Apply stable unit filter to firing rates and brain regions
            trial_firing_rates = trial_firing_rates(:, stable_mask, :);
            brain_regions = cell_metrics.brainRegion_final(stable_mask);
        else
            fprintf('  No stable_unit information found - using all neurons\n');
            stable_mask = true(n_neurons, 1);
            brain_regions = cell_metrics.brainRegion_final(1:n_neurons);
        end
        
        %% Stage 5: Construct Session Data Structure
        % Package all extracted and processed data for downstream analysis
        
        session_data = struct();
        session_data.session_name = session_name;
        session_data.spike_rates = trial_firing_rates;  % [N_trials × N_neurons × N_timepoints]
        session_data.brain_regions = brain_regions;
        session_data.cell_metrics = cell_metrics;
        session_data.stable_units = stable_mask;
        session_data.trial_info = trial_info;
        session_data.n_trials = trial_info.n_trials;
        session_data.n_neurons = size(trial_firing_rates, 2);
        session_data.n_timepoints = size(trial_firing_rates, 3);
        session_data.time_axis = trial_info.time_axis;
        session_data.time_window = trial_info.time_window;
        session_data.data_source = 'mdl';
        
        % Summary output
        fprintf('  Session data extraction completed successfully\n');
        fprintf('    Final dimensions: %d trials × %d neurons × %d timepoints\n', ...
                session_data.n_trials, session_data.n_neurons, session_data.n_timepoints);
        fprintf('    Time window: [%.1fs, %.1fs]\n', session_data.time_window(1), session_data.time_window(2));
        fprintf('    Unique brain regions: %d\n', length(unique(session_data.brain_regions)));
        
    catch ME
        fprintf('  Critical error in MDL data extraction: %s\n', ME.message);
        fprintf('  Stack trace:\n');
        for i = 1:length(ME.stack)
            fprintf('    Line %d in %s: %s\n', ME.stack(i).line, ME.stack(i).name, ME.stack(i).file);
        end
        session_data = [];
    end
end


function cell_metrics = load_and_combine_cell_metrics(probe_paths, session_name)
% LOAD_AND_COMBINE_CELL_METRICS - Manually load and combine cell metrics from multiple probes
%
% This fallback function is used when loadCellMetricsBatch is unavailable.
% It manually loads cell_metrics.cellinfo.mat files from each probe directory
% and concatenates the relevant fields.

    fprintf('    Manual cell metrics loading for %d probes...\n', length(probe_paths));
    
    cell_metrics = [];
    
    for i = 1:length(probe_paths)
        probe_path = probe_paths{i};
        [~, probe_name] = fileparts(probe_path);
        
        % Find cell metrics file in this probe directory
        cm_pattern = fullfile(probe_path, '*.cell_metrics.cellinfo.mat');
        cm_files = dir(cm_pattern);
        
        if isempty(cm_files)
            fprintf('    Warning: No cell metrics file found in %s\n', probe_path);
            continue;
        end
        
        cm_file = fullfile(probe_path, cm_files(1).name);
        fprintf('    Loading: %s\n', cm_files(1).name);
        
        try
            probe_data = load(cm_file);
            
            % Extract cell_metrics structure (may be nested)
            if isfield(probe_data, 'cell_metrics')
                probe_cm = probe_data.cell_metrics;
            else
                % Assume the loaded data IS the cell_metrics
                probe_cm = probe_data;
            end
            
            if isempty(cell_metrics)
                % First probe - initialize combined structure
                cell_metrics = probe_cm;
                cell_metrics.probe_assignment = repmat({probe_name}, 1, length(probe_cm.brainRegion_final));
            else
                % Subsequent probes - concatenate fields
                cell_metrics = concatenate_cell_metrics(cell_metrics, probe_cm, probe_name);
            end
            
        catch ME
            fprintf('    Error loading %s: %s\n', cm_files(1).name, ME.message);
        end
    end
    
    if ~isempty(cell_metrics)
        fprintf('    Combined cell metrics: %d total neurons\n', length(cell_metrics.brainRegion_final));
    end
end


function combined = concatenate_cell_metrics(existing, new_probe, probe_name)
% CONCATENATE_CELL_METRICS - Combine cell metrics from two probes
%
% This function concatenates the fields of two cell_metrics structures,
% handling both cell array and numeric array fields appropriately.

    combined = existing;
    
    % Get list of fields to concatenate
    field_names = fieldnames(new_probe);
    n_new_neurons = length(new_probe.brainRegion_final);
    
    for i = 1:length(field_names)
        fn = field_names{i};
        
        if ~isfield(combined, fn)
            % New field not in existing - skip or initialize
            continue;
        end
        
        existing_val = combined.(fn);
        new_val = new_probe.(fn);
        
        % Handle different data types
        if iscell(existing_val) && iscell(new_val)
            % Concatenate cell arrays
            if isrow(existing_val) && isrow(new_val)
                combined.(fn) = [existing_val, new_val];
            else
                combined.(fn) = [existing_val(:); new_val(:)];
            end
            
        elseif isnumeric(existing_val) && isnumeric(new_val)
            % Concatenate numeric arrays along the neuron dimension
            if isvector(existing_val) && isvector(new_val)
                if isrow(existing_val)
                    combined.(fn) = [existing_val, new_val(:)'];
                else
                    combined.(fn) = [existing_val; new_val(:)];
                end
            else
                % For matrices, assume first dimension is neurons
                combined.(fn) = [existing_val; new_val];
            end
            
        elseif islogical(existing_val) && islogical(new_val)
            % Concatenate logical arrays
            combined.(fn) = [existing_val(:); new_val(:)];
        end
        % Skip struct and other complex types
    end
    
    % Update probe assignment tracking
    if isfield(combined, 'probe_assignment')
        combined.probe_assignment = [combined.probe_assignment, repmat({probe_name}, 1, n_new_neurons)];
    end
end
