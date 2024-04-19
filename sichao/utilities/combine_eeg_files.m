function combine_eeg_files(source_dir, save_loc)

% PARAMETERS

% source_dir - location of EEG files to be combined; all files must have
% the same number of rows; all files must contain a field named 'y' and
% that field must contain an EEG data matrix of arbitrary size

% save_loc = location and file name in which to save aggregate EEG file

%-------------------------------------------------------------------------%

% RETURNS

% N/A

%-------------------------------------------------------------------------%

% This function can be used to stitch together EEG .mat files in the passed
% directory and save the stitched-together file. The files are combined
% horizontally, so the number of rows remains consistent but the number of
% columns grows. Every file in the directory needs a field named 'y' and
% every file must also have the same number of rows.

% Initialize empty array
temp = [];

% Get files to be combined
file_pattern = fullfile(source_dir, '*.mat'); % Regex-ish equivalent
files = dir(file_pattern); % Get all files from above directory matching pattern

% Concatenate files horizontally
for f = 1:length(files)
    load(files(f).name, 'y')
    temp = [temp, y];
end

% Assign to 'y' variable for .mat file consistency
y = temp;

% Save new array to specified save location
save(save_loc, 'y')