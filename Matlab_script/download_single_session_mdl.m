function success = download_single_session_mdl(session_id, date_str, local_base_dir, server_config)
% DOWNLOAD_SINGLE_SESSION - Download a single experimental session if not already present
% This function first checks if data already exists locally before attempting download

    % Construct session path
    session_name = sprintf('%s_%s', session_id, date_str);
    session_path = fullfile(local_base_dir, 'proc', session_id, session_name);
    
    % Check if the session directory already exists
    if exist(session_path, 'dir')
        % Check if the trial file exists (primary indicator of successful previous download)
        trial_file = fullfile(session_path, [session_name, '.mdl.mat']);
        
        if exist(trial_file, 'file')
            % Additional check for at least one Neuropixels probe directory
            npx_pattern = fullfile(session_path, '*_npx*');
            npx_dirs = dir(npx_pattern);
            npx_dirs = npx_dirs([npx_dirs.isdir]);
            
            if ~isempty(npx_dirs)
                fprintf('  Session data already exists for %s. Skipping download.\n', session_name);
                success = true;
                return;
            end
        end
    end
    
    % If we reach here, data is not available locally or is incomplete - perform download
    try
        fprintf('  Data not found locally. Downloading from server...\n');
        download_oxford_mdl_data(session_id, date_str, local_base_dir, ...
                           server_config.base_dir, server_config.host, ...
                           server_config.username);
        success = true;
    catch ME
        fprintf('  Download error: %s\n', ME.message);
        success = false;
    end
end


function download_oxford_mdl_data(session_id, date_str, local_base_dir, server_base_dir, server_host, username)
% DOWNLOAD_OXFORD_MDL_DATA - Download MDL and cell metrics data from Oxford HPC server
%
% This function has been updated to download the MDL (Model Data Layer) files
% instead of the previous trial-segmented format. The MDL files contain 
% continuous firing rate data in mdl.predictor.firing_rate, which must be
% subsequently segmented using t_approach timestamps.
%
% DATA ORGANIZATION CHANGE:
% Previous format: {session}.cue_dlc_bar_off.trial.mat (pre-segmented trials)
% Current format:  {session}.mdl.mat (continuous firing rates)
%
% The firing rate matrix in MDL format is:
%   mdl.predictor.firing_rate: [N_neurons × T_total] continuous data
%
% INPUTS:
%   session_id     - Animal/session identifier (e.g., 'yp020')
%   date_str       - Recording date in YYMMDD format (e.g., '220401')
%   local_base_dir - Local base directory for downloads
%   server_base_dir- Server base directory path (default: YP_Oxford)
%   server_host    - Server hostname
%   username       - SSH username
%
% EXAMPLE USAGE:
%   download_oxford_mdl_data('yp020', '220401', ...
%       '/Users/shengyuancai/Downloads/Oxford_dataset/', ...
%       '/data/cephfs-2/unmirrored/groups/peng/YP_Oxford/', ...
%       'hpc-login-1.cubi.bihealth.org', 'shca10_c');
%
% HPC FILE LOCATIONS:
%   MDL data: /data/cephfs-2/unmirrored/groups/peng/YP_Oxford/proc/{animal}/{session}/{session}.mdl.mat
%   Cell metrics: /data/cephfs-2/unmirrored/groups/peng/YP_Oxford/proc/{animal}/{session}/{session}_npxX/{session}_npxX.cell_metrics.cellinfo.mat

    % Construct session-specific paths following Oxford naming convention
    session_dir = sprintf('%s_%s', session_id, date_str);
    server_session_path = fullfile(server_base_dir, 'proc', session_id, session_dir);
    local_session_path = fullfile(local_base_dir, 'proc', session_id, session_dir);
    
    fprintf('=== Oxford MDL Data Download Protocol ===\n');
    fprintf('Session: %s\n', session_dir);
    fprintf('Server path: %s\n', server_session_path);
    fprintf('Local path: %s\n', local_session_path);
    fprintf('=========================================\n\n');
    
    %% Step 1: Create local directory structure
    % This ensures we maintain the hierarchical organization of the data
    if ~exist(local_session_path, 'dir')
        fprintf('Creating local directory structure...\n');
        mkdir(local_session_path);
        fprintf('✓ Created: %s\n\n', local_session_path);
    else
        fprintf('Local directory already exists: %s\n\n', local_session_path);
    end
    
    %% Step 2: Download MDL data file (NEW: replaces trial.mat download)
    % The MDL file contains continuous firing rate data that must be
    % segmented using t_approach timestamps
    mdl_filename = sprintf('%s.mdl.mat', session_dir);
    server_mdl_path = sprintf('%s:%s/%s', server_host, server_session_path, mdl_filename);
    local_mdl_path = fullfile(local_session_path, mdl_filename);
    
    fprintf('Downloading MDL data file...\n');
    fprintf('Source: %s\n', server_mdl_path);
    fprintf('Target: %s\n', local_mdl_path);
    
    % Construct and execute SCP command for MDL data
    scp_cmd = sprintf('scp %s@%s "%s"', username, server_mdl_path, local_mdl_path);
    [status, result] = system(scp_cmd);
    
    if status == 0
        fprintf('✓ MDL data downloaded successfully\n\n');
    else
        fprintf('✗ Error downloading MDL data:\n%s\n\n', result);
        return; % Exit if critical MDL data cannot be obtained
    end
    
    %% Step 3: Discover and download cell metrics files
    % Cell metrics files contain single-unit characterization data including
    % brainRegion_final which is essential for regional assignment
    fprintf('Discovering Neuropixels probe directories...\n');
    
    % List directories on the server to find all _npxX folders
    list_cmd = sprintf('ssh %s@%s "ls -d %s/*_npx*/"', username, server_host, server_session_path);
    [status, npx_dirs] = system(list_cmd);
    
    if status ~= 0
        fprintf('✗ Error listing Neuropixels directories:\n%s\n', npx_dirs);
        return;
    end
    
    % Parse the directory listing
    npx_dirs = strtrim(npx_dirs);
    if isempty(npx_dirs)
        fprintf('No Neuropixels directories found.\n');
        return;
    end
    
    % Split into individual directory paths
    npx_dir_list = strsplit(npx_dirs, '\n');
    npx_dir_list = npx_dir_list(~cellfun(@isempty, npx_dir_list));
    
    fprintf('Found %d Neuropixels probe directories:\n', length(npx_dir_list));
    for i = 1:length(npx_dir_list)
        fprintf('  - %s\n', npx_dir_list{i});
    end
    fprintf('\n');
    
    %% Step 4: Download cell metrics for each probe
    for i = 1:length(npx_dir_list)
        npx_dir = strtrim(npx_dir_list{i});
        
        % Extract the probe directory name from full path
        probe_dir_name = sprintf('%s_npx%d', session_dir, i);
        
        % Construct cell metrics filename following the naming convention
        cell_metrics_filename = sprintf('%s.cell_metrics.cellinfo.mat', probe_dir_name);
        
        % Define server and local paths for this cell metrics file
        server_cell_metrics_path = sprintf('%s:%s/%s', server_host, npx_dir, cell_metrics_filename);
        local_probe_dir = fullfile(local_session_path, probe_dir_name);
        local_cell_metrics_path = fullfile(local_probe_dir, cell_metrics_filename);
        
        fprintf('Processing probe: %s\n', probe_dir_name);
        fprintf('Cell metrics file: %s\n', cell_metrics_filename);
        
        % Create local probe directory
        if ~exist(local_probe_dir, 'dir')
            mkdir(local_probe_dir);
            fprintf('✓ Created probe directory: %s\n', local_probe_dir);
        end
        
        % Download cell metrics file
        fprintf('Downloading cell metrics...\n');
        fprintf('Source: %s\n', server_cell_metrics_path);
        fprintf('Target: %s\n', local_cell_metrics_path);
        
        scp_cmd = sprintf('scp %s@%s "%s"', username, server_cell_metrics_path, local_cell_metrics_path);
        [status, result] = system(scp_cmd);
        
        if status == 0
            fprintf('✓ Cell metrics downloaded successfully\n');
        else
            fprintf('✗ Error downloading cell metrics:\n%s\n', result);
            fprintf('Continuing with next probe...\n');
        end
        fprintf('\n');
    end
    
    %% Step 5: Verification and summary
    fprintf('=== Download Summary ===\n');
    
    % Verify MDL data
    if exist(local_mdl_path, 'file')
        mdl_info = dir(local_mdl_path);
        fprintf('✓ MDL data: %s (%.2f MB)\n', mdl_filename, mdl_info.bytes/1024/1024);
    else
        fprintf('✗ MDL data: Missing\n');
    end
    
    % Verify cell metrics files
    cell_metrics_count = 0;
    total_size = 0;
    for i = 1:length(npx_dir_list)
        probe_dir_name = sprintf('%s_npx%d', session_dir, i);
        cell_metrics_filename = sprintf('%s.cell_metrics.cellinfo.mat', probe_dir_name);
        local_cell_metrics_path = fullfile(local_session_path, probe_dir_name, cell_metrics_filename);
        
        if exist(local_cell_metrics_path, 'file')
            cell_info = dir(local_cell_metrics_path);
            total_size = total_size + cell_info.bytes;
            cell_metrics_count = cell_metrics_count + 1;
            fprintf('✓ Cell metrics (%s): %.2f MB\n', probe_dir_name, cell_info.bytes/1024/1024);
        else
            fprintf('✗ Cell metrics (%s): Missing\n', probe_dir_name);
        end
    end
    
    % Calculate total size including MDL file
    if exist(local_mdl_path, 'file')
        mdl_info = dir(local_mdl_path);
        total_with_mdl = total_size + mdl_info.bytes;
    else
        total_with_mdl = total_size;
    end
    
    fprintf('\nTotal files downloaded: %d\n', cell_metrics_count + 1);
    fprintf('Total data size: %.2f MB\n', total_with_mdl/1024/1024);
    fprintf('========================\n');
end
