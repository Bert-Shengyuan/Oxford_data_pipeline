function [trial_firing_rates, trial_info] = segment_mdl_to_trials(mdl_data, t_approach, session_id, date_str, config)
% SEGMENT_MDL_TO_TRIALS - Extract trial-segmented firing rates from continuous MDL data
%
% This function implements the critical transformation from continuous neural recordings
% to trial-aligned epochs using behavioral timestamps from the t_approach table. This
% represents a fundamental paradigm shift in the data processing pipeline.
%
% MATHEMATICAL FRAMEWORK:
% The continuous firing rate matrix F_mdl ∈ ℝ^{N_neurons × T_total} is segmented into
% K trials using the alignment times τ_k from t_approach:
%
%   F_trial(k) = F_mdl[:, τ_k - 75 : τ_k + 150]
%
% where τ_k is the start_time bin for trial k. This yields:
%   - Pre-event window: 75 bins × 20ms/bin = 1.5 seconds
%   - Post-event window: 150 bins × 20ms/bin = 3.0 seconds
%   - Total window: 225 bins × 20ms/bin = 4.5 seconds
%
% TRIAL SELECTION CRITERION:
% Only trials with label == 'cued hit long' are extracted, starting from the first
% trial of each session. This ensures consistent behavioral state across analyses.
%
% INPUTS:
%   mdl_data    - Struct containing mdl.predictor.firingrate [N_neurons × T_total]
%   t_approach  - Table with columns: {animal_id, session_date, session_name, ..., start_time, label}
%                 Note: start_time should be in bins (not seconds)
%   session_id  - Animal identifier (e.g., 'yp020')
%   date_str    - Recording date in YYMMDD format (e.g., '220401')
%   config      - Configuration struct with time_window field
%
% OUTPUTS:
%   trial_firing_rates - [N_trials × N_neurons × N_timepoints] tensor
%   trial_info         - Struct containing trial metadata:
%                        .n_trials, .n_neurons, .n_timepoints
%                        .trial_labels, .start_times, .valid_trial_indices
%
% EXAMPLE USAGE:
%   mdl = load('yp020_220401.mdl.mat');
%   t_approach = load('t_approach.mat');
%   [trial_fr, info] = segment_mdl_to_trials(mdl, t_approach.t_approach, 'yp020', '220401', config);
%
% NOTES:
% - t_approach is based on get_tapproach.m from global_movement-master_test/extract
% - The label column contains: "cued hit long", "other", "spont short"
% - Assumes 50Hz binning (20ms bins) in the MDL firing rate data

    fprintf('  Segmenting continuous MDL data into trials...\n');
    
    % Define bin parameters for 50Hz data (20ms bins)
    % Time window: [-1.5s, 3.0s] → bins: [-75, +150] relative to start_time
    pre_event_bins = 75;    % 1.5 seconds before event
    post_event_bins = 150;  % 3.0 seconds after event
    total_bins = pre_event_bins + post_event_bins + 1;  % 226 bins total (including start bin)
    
    % Validate time window consistency with config
    expected_pre = abs(config.time_window(1)) * 50;   % Convert seconds to bins at 50Hz
    expected_post = config.time_window(2) * 50;
    if abs(expected_pre - pre_event_bins) > 1 || abs(expected_post - post_event_bins) > 1
        warning('Time window parameters may not match config. Using [-1.5s, 3.0s] as specified.');
    end
    
    %% Step 1: Extract firing rate matrix from MDL structure
    % The MDL data contains firing rates in mdl.predictor.firingrate
    if isfield(mdl_data, 'mdl')
        % Handle case where mdl_data is the loaded struct containing 'mdl' field
        firing_rate_continuous = mdl_data.mdl.predictor.firingrate;
    elseif isfield(mdl_data, 'predictor')
        % Handle case where mdl_data IS the mdl struct directly
        firing_rate_continuous = mdl_data.predictor.firingrate;
    else
        error('MDL data structure not recognized. Expected mdl.predictor.firingrate');
    end
    
    [n_neurons, n_total_bins] = size(firing_rate_continuous);
    fprintf('    Continuous data dimensions: %d neurons × %d bins (%.1f seconds)\n', ...
            n_neurons, n_total_bins, n_total_bins/50);
    
    %% Step 2: Filter t_approach for current session
    % Match session based on animal_id, session_date, and session_name
    % First three columns of t_approach contain: animal_id, session_date, session_name
    
    session_name = sprintf('%s_%s', session_id, date_str);
    
    % Get column names to handle different table formats
    if istable(t_approach)
        col_names = t_approach.Properties.VariableNames;
    else
        error('t_approach must be a MATLAB table');
    end
    
    % Identify session matching columns (first three columns)
    % Typical structure: animal_id | session_date | session_name | ... | start_time | label
    
    % Convert session identifiers for matching
    % Handle different possible column name conventions
    if ismember('animal_id', col_names)
        animal_col = 'animal_id';
    elseif ismember('animalID', col_names)
        animal_col = 'animalID';
    else
        % Assume first column is animal identifier
        animal_col = col_names{1};
        fprintf('    Using column "%s" as animal identifier\n', animal_col);
    end
    
    if ismember('session_date', col_names)
        date_col = 'session_date';
    elseif ismember('sessionDate', col_names)
        date_col = 'sessionDate';
    else
        % Assume second column is session date
        date_col = col_names{2};
        fprintf('    Using column "%s" as session date\n', date_col);
    end
    
    % Create session mask
    % Handle both string and cell array column types
    animal_values = t_approach.(animal_col);
    if iscell(animal_values)
        animal_mask = strcmp(animal_values, session_id);
    else
        animal_mask = animal_values == string(session_id);
    end
    
    date_values = t_approach.(date_col);
    if iscell(date_values)
        % Date might be stored as string 'YYMMDD' or numeric
        date_mask = strcmp(date_values, date_str) | strcmp(date_values, ['20' date_str]);
    elseif isnumeric(date_values)
        date_mask = date_values == str2double(date_str) | date_values == str2double(['20' date_str]);
    else
        date_mask = date_values == string(date_str) | date_values == string(['20' date_str]);
    end
    
    session_mask = animal_mask & date_mask;
    fprintf('    Found %d trials matching session %s in t_approach\n', sum(session_mask), session_name);
    
    %% Step 3: Filter for 'cued hit long' trials only
    % The label column contains: "cued hit long", "other", "spont short"
    
    if ~ismember('label', col_names)
        error('t_approach table must contain a "label" column');
    end
    
    label_values = t_approach.label;
    if iscell(label_values)
        label_mask = strcmp(label_values, 'cued hit long');
    else
        label_mask = label_values == "cued hit long";
    end
    
    % Combine session and label masks
    valid_trial_mask = session_mask & label_mask;
    n_valid_trials = sum(valid_trial_mask);
    
    fprintf('    Trials with "cued hit long" label: %d\n', n_valid_trials);
    
    if n_valid_trials == 0
        warning('No valid trials found for session %s with label "cued hit long"', session_name);
        trial_firing_rates = [];
        trial_info = struct('n_trials', 0, 'error', 'No valid trials');
        return;
    end
    
    %% Step 4: Extract start_time for valid trials
    % start_time should be in bins (at 50Hz, corresponding to MDL firing rate indexing)
    
    if ~ismember('start_time', col_names)
        error('t_approach table must contain a "start_time" column');
    end
    
    % Get the subset of t_approach for valid trials
    valid_trials_table = t_approach(valid_trial_mask, :);
    start_times = valid_trials_table.start_time;
    
    % Ensure start_times are numeric (bin indices)
    if iscell(start_times)
        start_times = cell2mat(start_times);
    end
    
    % Convert to integer bin indices if they appear to be in seconds
    % Heuristic: if max start_time is small relative to n_total_bins, it's likely in seconds
    if max(start_times) < n_total_bins / 50
        fprintf('    WARNING: start_time appears to be in seconds, converting to bins...\n');
        start_times = round(start_times * 50);  % Convert seconds to bins at 50Hz
    end
    
    start_times = round(start_times);  % Ensure integer bin indices
    
    fprintf('    Start time range: bins %d to %d\n', min(start_times), max(start_times));
    
    %% Step 5: Segment continuous data into trial epochs
    % For each valid trial, extract: [start_bin - 75, start_bin + 150]
    
    % Preallocate output tensor
    trial_firing_rates = zeros(n_valid_trials, n_neurons, total_bins);
    valid_trial_indices = zeros(n_valid_trials, 1);
    skipped_trials = 0;
    
    trial_idx = 0;
    for i = 1:n_valid_trials
        start_bin = start_times(i);
        
        % Calculate epoch boundaries
        epoch_start = start_bin - pre_event_bins;
        epoch_end = start_bin + post_event_bins;
        
        % Validate boundaries (ensure we don't exceed data limits)
        if epoch_start < 1
            fprintf('    Skipping trial %d: start bin %d requires pre-event data before recording start\n', i, start_bin);
            skipped_trials = skipped_trials + 1;
            continue;
        end
        
        if epoch_end > n_total_bins
            fprintf('    Skipping trial %d: end bin %d exceeds recording length %d\n', i, epoch_end, n_total_bins);
            skipped_trials = skipped_trials + 1;
            continue;
        end
        
        % Extract epoch data
        trial_idx = trial_idx + 1;
        trial_firing_rates(trial_idx, :, :) = firing_rate_continuous(:, epoch_start:epoch_end);
        valid_trial_indices(trial_idx) = i;
    end
    
    % Trim preallocated arrays to actual number of valid trials
    if trial_idx < n_valid_trials
        trial_firing_rates = trial_firing_rates(1:trial_idx, :, :);
        valid_trial_indices = valid_trial_indices(1:trial_idx);
    end
    
    actual_n_trials = trial_idx;
    
    %% Step 6: Construct output metadata structure
    trial_info = struct();
    trial_info.n_trials = actual_n_trials;
    trial_info.n_neurons = n_neurons;
    trial_info.n_timepoints = total_bins;
    trial_info.session_name = session_name;
    trial_info.trial_labels = repmat({'cued hit long'}, actual_n_trials, 1);
    trial_info.start_times = start_times(valid_trial_indices);
    trial_info.valid_trial_indices = valid_trial_indices;
    trial_info.skipped_trials = skipped_trials;
    trial_info.time_window = [-1.5, 3.0];  % seconds
    trial_info.bin_window = [-pre_event_bins, post_event_bins];  % bins relative to start
    trial_info.sampling_rate_hz = 50;
    trial_info.time_axis = linspace(-1.5, 3.0, total_bins);  % Time axis in seconds
    
    % Summary statistics
    fprintf('    Trial segmentation complete:\n');
    fprintf('      Successfully extracted: %d trials\n', actual_n_trials);
    fprintf('      Skipped (boundary issues): %d trials\n', skipped_trials);
    fprintf('      Output dimensions: [%d trials × %d neurons × %d timepoints]\n', ...
            actual_n_trials, n_neurons, total_bins);
    fprintf('      Time window: [%.1fs, %.1fs] relative to start_time\n', ...
            trial_info.time_window(1), trial_info.time_window(2));
    
    %% Validation: Check for NaN or Inf values
    if any(isnan(trial_firing_rates(:)))
        warning('NaN values detected in segmented firing rates');
        trial_info.has_nan = true;
    else
        trial_info.has_nan = false;
    end
    
    if any(isinf(trial_firing_rates(:)))
        warning('Inf values detected in segmented firing rates');
        trial_info.has_inf = true;
    else
        trial_info.has_inf = false;
    end
    
end
