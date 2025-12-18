function cleanup_session_mdl_files(session_id, date_str, local_base_dir, verbose)
% CLEANUP_SESSION_MDL_FILES - Remove raw MDL data and cell metrics files after processing
%
% This function implements the selective cleanup strategy for the updated pipeline,
% removing the large raw data files (MDL and cell_metrics) while preserving the
% processed CCA results and PSTH data. This approach ensures efficient disk usage
% during batch processing of multiple sessions.
%
% FILES REMOVED:
% 1. MDL data: {session}.mdl.mat - Contains continuous firing rate data
% 2. Cell metrics: {session}_npxX.cell_metrics.cellinfo.mat - One per probe
%
% RATIONALE:
% After successful extraction and segmentation, the trial-level firing rate data
% is stored within the CCA results file (region_data.regions.{region}.spike_data).
% The raw continuous MDL data and cell metrics are no longer needed and can be
% safely deleted to reclaim disk space for subsequent sessions.
%
% INPUTS:
%   session_id    - Animal identifier (e.g., 'yp020')
%   date_str      - Recording date in YYMMDD format (e.g., '220401')
%   local_base_dir- Base directory containing the proc folder
%   verbose       - (Optional) Boolean for detailed output (default: true)
%
% EXAMPLE USAGE:
%   cleanup_session_mdl_files('yp020', '220401', '/Users/user/Oxford_dataset/', true);
%
% NOTES:
% - This function should be called AFTER save_session_results has successfully
%   completed to ensure all processed data is preserved
% - Only deletes specific file types, not the entire session directory
% - Provides detailed logging of deletion status for audit purposes

    if nargin < 4
        verbose = true;
    end
    
    session_name = sprintf('%s_%s', session_id, date_str);
    session_path = fullfile(local_base_dir, 'proc', session_id, session_name);
    
    if verbose
        fprintf('  Cleaning up raw data files for session: %s\n', session_name);
    end
    
    if ~exist(session_path, 'dir')
        if verbose
            fprintf('  Session directory does not exist: %s\n', session_path);
        end
        return;
    end
    
    files_deleted = 0;
    bytes_freed = 0;
    
    %% Step 1: Delete MDL data file
    % The MDL file is typically the largest file and contains continuous firing rates
    
    mdl_file = fullfile(session_path, sprintf('%s.mdl.mat', session_name));
    
    if exist(mdl_file, 'file')
        try
            file_info = dir(mdl_file);
            delete(mdl_file);
            files_deleted = files_deleted + 1;
            bytes_freed = bytes_freed + file_info.bytes;
            
            if verbose
                fprintf('    ✓ Deleted MDL file: %s (%.2f MB)\n', ...
                        sprintf('%s.mdl.mat', session_name), file_info.bytes/1024/1024);
            end
        catch ME
            if verbose
                fprintf('    ✗ Failed to delete MDL file: %s\n', ME.message);
            end
        end
    else
        if verbose
            fprintf('    - MDL file not found (may already be deleted)\n');
        end
    end
    
    %% Step 2: Delete cell metrics files from all probe directories
    % Each Neuropixels probe has its own cell_metrics.cellinfo.mat file
    
    npx_pattern = fullfile(session_path, sprintf('%s_npx*', session_name));
    npx_dirs = dir(npx_pattern);
    npx_dirs = npx_dirs([npx_dirs.isdir]);
    
    if verbose && ~isempty(npx_dirs)
        fprintf('    Found %d Neuropixels probe directories to clean\n', length(npx_dirs));
    end
    
    for i = 1:length(npx_dirs)
        probe_dir = fullfile(session_path, npx_dirs(i).name);
        probe_name = npx_dirs(i).name;
        
        % Find cell metrics file in this probe directory
        cm_pattern = fullfile(probe_dir, '*.cell_metrics.cellinfo.mat');
        cm_files = dir(cm_pattern);
        
        for j = 1:length(cm_files)
            cm_file_path = fullfile(probe_dir, cm_files(j).name);
            
            try
                file_info = dir(cm_file_path);
                delete(cm_file_path);
                files_deleted = files_deleted + 1;
                bytes_freed = bytes_freed + file_info.bytes;
                
                if verbose
                    fprintf('    ✓ Deleted cell metrics: %s/%s (%.2f MB)\n', ...
                            probe_name, cm_files(j).name, file_info.bytes/1024/1024);
                end
            catch ME
                if verbose
                    fprintf('    ✗ Failed to delete %s: %s\n', cm_files(j).name, ME.message);
                end
            end
        end
        
        % Optionally remove the empty probe directory
        % Check if directory is now empty
        remaining_files = dir(fullfile(probe_dir, '*'));
        remaining_files = remaining_files(~ismember({remaining_files.name}, {'.', '..'}));
        
        if isempty(remaining_files)
            try
                rmdir(probe_dir);
                if verbose
                    fprintf('    ✓ Removed empty probe directory: %s\n', probe_name);
                end
            catch ME
                if verbose
                    fprintf('    - Could not remove probe directory: %s\n', ME.message);
                end
            end
        end
    end
    
    %% Step 3: Check if session directory can be removed
    % If only metadata or empty directories remain, clean up the session directory
    
    remaining_items = dir(session_path);
    remaining_items = remaining_items(~ismember({remaining_items.name}, {'.', '..'}));
    
    if isempty(remaining_items)
        try
            rmdir(session_path);
            if verbose
                fprintf('    ✓ Removed empty session directory\n');
            end
            
            % Also check if the animal directory is now empty
            animal_dir = fullfile(local_base_dir, 'proc', session_id);
            animal_contents = dir(animal_dir);
            animal_contents = animal_contents(~ismember({animal_contents.name}, {'.', '..'}));
            
            if isempty(animal_contents)
                rmdir(animal_dir);
                if verbose
                    fprintf('    ✓ Removed empty animal directory: %s\n', session_id);
                end
            end
        catch ME
            if verbose
                fprintf('    - Could not remove session directory: %s\n', ME.message);
            end
        end
    end
    
    %% Summary
    if verbose
        fprintf('  Cleanup summary: %d files deleted, %.2f MB freed\n', ...
                files_deleted, bytes_freed/1024/1024);
    end
end


function cleanup_all_session_raw_data(session_list, local_base_dir, verbose)
% CLEANUP_ALL_SESSION_RAW_DATA - Batch cleanup of raw data for multiple sessions
%
% This helper function applies cleanup to all sessions in a list.
% Useful for post-processing cleanup after an entire batch has been analyzed.
%
% INPUTS:
%   session_list  - Cell array of {animal_id, date_str} pairs
%   local_base_dir- Base directory containing proc folder
%   verbose       - Boolean for detailed output

    if nargin < 3
        verbose = true;
    end
    
    fprintf('=== Batch Cleanup of Raw Session Data ===\n');
    fprintf('Processing %d sessions...\n\n', length(session_list));
    
    total_freed = 0;
    
    for i = 1:length(session_list)
        session_info = session_list{i};
        session_id = session_info{1};
        date_str = session_info{2};
        
        fprintf('Session %d/%d: %s_%s\n', i, length(session_list), session_id, date_str);
        
        % Get initial disk usage for this session
        session_path = fullfile(local_base_dir, 'proc', session_id, sprintf('%s_%s', session_id, date_str));
        if exist(session_path, 'dir')
            [~, result] = system(sprintf('du -sh "%s" 2>/dev/null', session_path));
            if verbose
                fprintf('  Initial size: %s', strtrim(result));
            end
        end
        
        cleanup_session_mdl_files(session_id, date_str, local_base_dir, verbose);
        fprintf('\n');
    end
    
    fprintf('=== Batch Cleanup Complete ===\n');
end
