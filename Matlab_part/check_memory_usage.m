function memory_usage = check_memory_usage()
% CHECK_MEMORY_USAGE - Monitor memory consumption during processing
% This function provides memory usage feedback for pipeline optimization

    if ispc
        [~, memory_info] = system('wmic OS get TotalVisibleMemorySize,FreePhysicalMemory /value');
        % Parse Windows memory info
        memory_usage.total_gb = 'Unknown';
        memory_usage.free_gb = 'Unknown';
        memory_usage.used_percent = 'Unknown';
    elseif ismac || isunix
        [~, memory_info] = system('vm_stat');
        % Parse Unix/Mac memory info - simplified
        memory_usage.total_gb = 'System dependent';
        memory_usage.free_gb = 'System dependent';
        memory_usage.used_percent = 'Check system monitor';
    else
        memory_usage.total_gb = 'Unknown system';
        memory_usage.free_gb = 'Unknown';
        memory_usage.used_percent = 'Unknown';
    end
    
    % MATLAB memory information
    matlab_memory = memory;
    memory_usage.matlab_used_gb = matlab_memory.MemUsedMATLAB / 1024^3;
    memory_usage.matlab_available_gb = matlab_memory.MemAvailableAllArrays / 1024^3;
end
