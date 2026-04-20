% camargo_mat2csv.m
% ------------------------------------------------------------------
% Converts all .mat files in the Camargo dataset to .csv, keeping the
% original folder layout. Handles MATLAB `table` class (MCOS) and falls
% back to struct->CSV for other formats.
%
% Usage:
%   1. Open MATLAB and cd to the dataset root
%      (the folder that contains AB06/, AB07/, AB08/, ...)
%   2. Run:
%         camargo_mat2csv                     % convert imu + conditions
%         camargo_mat2csv('sensors', {'imu','conditions','gon'})
%         camargo_mat2csv('root', 'D:/path/to/dataset')
%
% Output:
%   For every .mat found, a .csv is written next to it. Existing .csv
%   files are skipped (resumable).
%
% Tested on MATLAB R2020a+. Requires Statistics & ML Toolbox only if
% the source tables use categorical types (Camargo data does not).
% ------------------------------------------------------------------

function camargo_mat2csv(varargin)

    % ---- parse options ----
    p = inputParser;
    addParameter(p, 'root',    pwd,                          @ischar);
    addParameter(p, 'sensors', {'imu', 'conditions'},        @iscell);
    addParameter(p, 'overwrite', false,                      @islogical);
    parse(p, varargin{:});
    root      = p.Results.root;
    sensors   = p.Results.sensors;
    overwrite = p.Results.overwrite;

    fprintf('Root: %s\n', root);
    fprintf('Sensors: %s\n', strjoin(sensors, ', '));

    subjects = dir(fullfile(root, 'AB*'));
    subjects = subjects([subjects.isdir]);
    if isempty(subjects)
        error('No AB* subject folders found under %s', root);
    end

    total_mat   = 0;
    total_done  = 0;
    total_skip  = 0;
    total_fail  = 0;

    for s = 1:length(subjects)
        subj_dir  = fullfile(root, subjects(s).name);
        date_dirs = dir(subj_dir);
        date_dirs = date_dirs([date_dirs.isdir] & ~startsWith({date_dirs.name}, '.'));

        for d = 1:length(date_dirs)
            date_dir = fullfile(subj_dir, date_dirs(d).name);
            modes    = {'levelground','ramp','stair','treadmill'};

            for m = 1:length(modes)
                mode_dir = fullfile(date_dir, modes{m});
                if ~exist(mode_dir, 'dir'), continue; end

                for si = 1:length(sensors)
                    sens_dir = fullfile(mode_dir, sensors{si});
                    if ~exist(sens_dir, 'dir'), continue; end

                    files = dir(fullfile(sens_dir, '*.mat'));
                    for f = 1:length(files)
                        in  = fullfile(sens_dir, files(f).name);
                        out = strrep(in, '.mat', '.csv');
                        total_mat = total_mat + 1;

                        if exist(out, 'file') && ~overwrite
                            total_skip = total_skip + 1;
                            continue;
                        end

                        try
                            convert_one(in, out);
                            total_done = total_done + 1;
                        catch ME
                            total_fail = total_fail + 1;
                            fprintf('  FAIL %s : %s\n', in, ME.message);
                        end
                    end
                end
            end
        end
        fprintf('[%2d/%2d] %s : converted=%d skipped=%d failed=%d (of %d)\n', ...
                s, length(subjects), subjects(s).name, ...
                total_done, total_skip, total_fail, total_mat);
    end

    fprintf('\nFinished.  total=%d  converted=%d  skipped=%d  failed=%d\n', ...
            total_mat, total_done, total_skip, total_fail);
end


% --- converter: load one .mat and write one .csv ---
function convert_one(in_path, out_path)
    S  = load(in_path);
    fn = fieldnames(S);

    % 1) prefer a MATLAB table field
    for k = 1:length(fn)
        v = S.(fn{k});
        if istable(v)
            writetable(v, out_path);
            return;
        end
    end

    % 2) next, look for a struct with a Header field + numeric columns
    %    (some sensors are stored as struct-of-arrays rather than table)
    for k = 1:length(fn)
        v = S.(fn{k});
        if isstruct(v) && isfield(v, 'Header')
            T = struct2table(v);
            writetable(T, out_path);
            return;
        end
    end

    % 3) fallback: try the whole loaded struct as one table
    try
        T = struct2table(S);
        writetable(T, out_path);
        return;
    catch
    end

    error('no convertible field found in %s (keys: %s)', ...
          in_path, strjoin(fn, ', '));
end
