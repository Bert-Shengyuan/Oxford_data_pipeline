function convert_tcell_table_to_struct(input_file, output_file)
% CONVERT_TCELL_TABLE_TO_STRUCT - Convert a table-backed cell-metrics .mat
% file into a scalar "struct-of-arrays" .mat file that is easy to load
% from Python (e.g. with h5py, pymatreader, or mat73).
%
% Each table variable becomes one struct field holding the full column
% (not one struct element per row), so downstream Python code can read
% every field straight into a numpy array / pandas DataFrame column.
%
% Usage:
%   convert_tcell_table_to_struct()
%   convert_tcell_table_to_struct(input_file, output_file)

    if nargin < 1 || isempty(input_file)
        input_file = '/Users/shengyuancai/Downloads/Oxford_dataset/tcell_cellmetrics.mat';
    end
    if nargin < 2 || isempty(output_file)
        [in_dir, in_name] = fileparts(input_file);
        output_file = fullfile(in_dir, [in_name '_struct.mat']);
    end

    fprintf('Loading %s ...\n', input_file);
    S_in = load(input_file);

    % Find the table variable regardless of its name.
    var_names = fieldnames(S_in);
    table_name = '';
    for i = 1:numel(var_names)
        if istable(S_in.(var_names{i}))
            table_name = var_names{i};
            break
        end
    end
    if isempty(table_name)
        error('convert_tcell_table_to_struct:noTable', ...
            'No table variable found in %s', input_file);
    end

    T = S_in.(table_name);
    fprintf('Found table "%s": %d rows x %d columns\n', table_name, height(T), width(T));

    % Convert categorical / string columns to cell arrays of char so that
    % they save as plain text (cellstr) rather than MATLAB-specific
    % object classes, which are painful to read from Python.
    col_names = T.Properties.VariableNames;
    for i = 1:numel(col_names)
        col = T.(col_names{i});
        if iscategorical(col)
            T.(col_names{i}) = cellstr(col);
        elseif isstring(col)
            T.(col_names{i}) = cellstr(col);
        end
    end

    % 'ToScalar' keeps each table variable as one struct field holding
    % the whole column, instead of building a 1xN struct array.
    tcell_struct = table2struct(T, 'ToScalar', true);
    tcell_struct.row_names = T.Properties.RowNames;
    tcell_struct.variable_names = col_names;

    fprintf('Saving struct to %s ...\n', output_file);
    save(output_file, 'tcell_struct', '-v7.3');
    fprintf('Done.\n');
end
