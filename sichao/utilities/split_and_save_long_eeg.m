function split_and_save_long_eeg(source_dir, save_dir, fs, rows)

% PARAMETERS

% source_dir - location of EEG files to be split into trials; all files must contain a field named 'y' and
% that field must contain an EEG data matrix of arbitrary size

% save_dir - location in which to save individual trial .mat files

% fs - sampling rate of the EEG waveform data (Hz)

% rows - vector of row indices to load from files in source_dir; these may
% differ from sensor indices as they are dependent on the format of the EEG
% data files

%-------------------------------------------------------------------------%

% RETURNS

% N/A

%-------------------------------------------------------------------------%

% This function reads from a folder containing very long EEG data files. It
% splits those files into individual experimental trials and saves them
% separately.

% The mat files must have a single field named 'y' and that field must
% contain the EEG experimental data matrix.

% Passed parameter rows denotes the rows the user wishes to be loaded in

% The file naming convention must match the one established by the joint
% CWRU and KTH research team.

% This function should be ran only if the user desires to separate out
% EVERY single trial from every file in the source_dir directory. Other
% functions, like make_eeg_filer_banks, do not require every trial to be
% separated in order to function properly and so do NOT require this
% function to be ran first.

%-------------------------------------------------------------------------%

% Begin function execution timing
tic

% Empty destination directory
% while ~isempty(dir(append(save_loc, '*.mat')))
%     delete(append(save_loc, '*.mat'))
% end
% disp("Save location cleared")

% Get files to be split
file_pattern = fullfile(source_dir, '*.mat');
files = dir(file_pattern);

% Variable to track how many files have been saved; ensures unique
% identifier for every saved file
tot_count = 1;

% Loop through all files in source_loc directory and split them
for f = 1:length(files)
    
    % Load in file to be split and get only desired rows from data matrix
    file = files(f).name;
    eeg_data = load_eeg(fullfile(source_dir, file), fs, rows);
    
    % Get length of each trial in file
    [trial_length_seconds, ~] = get_trial_length_seconds(file);
    
    % Get everything preceding '-#.mat' in file name
    command_and_timing = regexp(files(f).name, '\w+(?=-)', 'match');
    
    % Split larger signals into trials. Each row of this tensor is a
    % different sensor, each column a step in time for each trial, and each
    % increment in depth is another trial.
    trials = split_data_into_trials(eeg_data, fs, trial_length_seconds);
    
    % For each trial, get data for all desired rows and save
    for i = 1:size(trials, 3)
        
        % Select trial to be saved
        y = trials(:, :, i);
        
        % Prepare save filename Has form <three letter command
        % code>-<timing code>-<index>.mat
        save_filename = fullfile(save_dir, strcat(command_and_timing{:},...
            '-', num2str(tot_count), '.mat'));
        
        % Save trial? this file contains all 16 sensors' worth of data for
        % the given trial
        save(save_filename, 'y')
        
        tot_count = tot_count + 1; % Increment # of files saved
        
    end
end

% End execution timing
toc

end