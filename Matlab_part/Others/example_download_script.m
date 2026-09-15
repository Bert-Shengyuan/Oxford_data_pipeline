%% Oxford Data Download Example Script
% This script demonstrates how to download neurophysiology data from the
% Oxford server using the download_oxford_data function.
%
% Author: Computational Neuroscience Lab
% Purpose: Automated data acquisition for CCA analysis pipeline

%% Configuration Parameters
% These parameters define your server connection and local storage setup

% Server configuration
server_host = 'hpc-login-1.cubi.bihealth.org';
username = 'shca10_c';  % Replace with your actual username
server_base_dir = '/data/cephfs-2/unmirrored/groups/peng/YP_Oxford/';

% Local storage configuration
local_base_dir = '/Users/shengyuancai/Downloads/Oxford_dataset/';

%% Single Session Download Example

fprintf('Starting download \n\n');

% % Call the download function
% download_oxford_data('yp014', '220212', local_base_dir, server_base_dir, server_host, username);
% 
% %% Verification of Downloaded Files
% % After download, let's verify the files are where we expect them
% 
% session_dir = 'yp014_220212';
% local_session_path = fullfile(local_base_dir, 'proc', 'yp014', session_dir);
% 
% fprintf('\n=== Post-Download Verification ===\n');
% 
% % Check trial data
% trial_file = fullfile(local_session_path, [session_dir, '.cue_dlc_bar_off.trial.mat']);
% if exist(trial_file, 'file')
%     fprintf('✓ Trial data found: %s\n', trial_file);
% 
%     % Load and inspect trial data structure
%     try
%         trial_data = load(trial_file);
%         fprintf('  Variables in trial data: %s\n', strjoin(fieldnames(trial_data), ', '));
%     catch ME
%         fprintf('  Could not load trial data: %s\n', ME.message);
%     end
% else
%     fprintf('✗ Trial data not found at: %s\n', trial_file);
% end
% 
% % Check for cell metrics files
% fprintf('\nChecking for cell metrics files...\n');
% npx_pattern = fullfile(local_session_path, '*_npx*');
% npx_dirs = dir(npx_pattern);
% npx_dirs = npx_dirs([npx_dirs.isdir]);
% 
% if ~isempty(npx_dirs)
%     for i = 1:length(npx_dirs)
%         probe_name = npx_dirs(i).name;
%         cell_metrics_file = fullfile(local_session_path, probe_name, [probe_name, '.cell_metrics.cellinfo.mat']);
% 
%         if exist(cell_metrics_file, 'file')
%             fprintf('✓ Cell metrics found for %s\n', probe_name);
% 
%             % Load and inspect cell metrics structure
%             try
%                 cell_data = load(cell_metrics_file);%
%                 if isfield(cell_data, 'cell_metrics')
%                     n_cells = length(cell_data.cell_metrics.cellID);
%                     fprintf('  Number of cells: %d\n', n_cells);
%                 end
%             catch ME
%                 fprintf('  Could not load cell metrics: %s\n', ME.message);
%             end
%         else
%             fprintf('✗ Cell metrics not found for %s\n', probe_name);
%         end
%     end
% else
%     fprintf('No Neuropixels directories found locally.\n');
% end

%% Example: Batch Download Multiple Sessions
% If you need to download multiple sessions, you can use the batch function

% Define multiple sessions to download
session_list = {
    {'yp014', '220212'};  %   % Hypothetical additional sessions
    {'yp021', '220407'};  % Add more as needed
};

% Server configuration structure
server_config = struct();
server_config.host = server_host;
server_config.username = username;
server_config.base_dir = server_base_dir;

% Uncomment the following lines to run batch download
fprintf('\n=== Starting Batch Download ===\n');
batch_download_oxford_data(session_list, local_base_dir, server_config);

%% Integration with Your CCA Analysis Pipeline
% This section shows how the downloaded data integrates with your existing
% analysis scripts (LL_CCA_infering_both.m and RR_CCA_infering_both.m)

fprintf('\n=== Integration with CCA Analysis ===\n');

% The downloaded trial data can be used to extract behavioral variables
if exist(trial_file, 'file')
    fprintf('Trial data is ready for behavioral analysis.\n');
    fprintf('Typical variables might include:\n');
    fprintf('  - Stimulus onset times\n');
    fprintf('  - Choice outcomes\n');
    fprintf('  - Reaction times\n');
    fprintf('  - Trial types\n');
end

% The downloaded cell metrics can be used for neuron selection
if ~isempty(npx_dirs)
    fprintf('\nCell metrics data is ready for neuron filtering.\n');
    fprintf('Typical filtering criteria might include:\n');
    fprintf('  - Spike waveform quality\n');
    fprintf('  - Firing rate thresholds\n');
    fprintf('  - Isolation distance\n');
    fprintf('  - Contamination percentage\n');
end

fprintf('\nNext steps:\n');
fprintf('1. Process trial data to extract task variables\n');
fprintf('2. Apply cell metrics filters to select high-quality neurons\n');
fprintf('3. Align neural data to behavioral events\n');
fprintf('4. Run CCA analysis using your existing scripts\n');

fprintf('\n=== Download Protocol Complete ===\n');
