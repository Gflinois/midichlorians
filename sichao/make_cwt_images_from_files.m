function make_cwt_images_from_files(source_loc, save_dir, fs)

% TODO: Finish this

% This function makes CWT images from a directory of mat files. The files
% must all contain exactly one field named 'y'. That field must contain a
% matrix with n rows (one for each EEG sensor.) The mat files must use the
% file naming convention established by the joint CWRU and KTH research
% team.

%-------------------------------------------------------------------------%

% Begin function execution timing
tic

% Empty destination directory
% while ~isempty(dir(append(save_loc, '*.mat')))
%     delete(append(save_loc, '*.mat'))
% end
% disp("Save location cleared")

% Get files to be trimmed
file_pattern = fullfile(source_loc, '*.mat'); % Regex-ish equivalent
files = dir(file_pattern); % Get all files from above directory matching pattern

% Load command words and codes from vocabulary file
load("matlab_code/signal_vocabulary.mat",...
    'subject_codes', 'subject_words',...
    'predicate_codes', 'predicate_words',...
    'object_codes', 'object_words')

% Variable to track how many files have been saved; ensures unique
% identifier for every saved file
tot_count = 1;

% Loop through every sensor
for f = 1:length(files)
    
    % Load in file and get field 'y'
    load(files(f).name, 'y')
    signals = y;
    
    % Get three letter command code
    command = regexp(files(f).name, '\w+(?=_)', 'match');
    disp(command)
    
    for row = 1:size(signals, 1)
        
        % Get CWT image matrix
        y = make_cwt_img(signals(row, :), fs);
        
        % Prepare save filename
        % Has form <sensor>-<one letter command code>-<tot_count>.mat
        save_filename = fullfile(save_dir, strcat(num2str(row), '-', ...
            command, '-', num2str(tot_count), '.mat'));
        
        % Save to file with name <sensor>-<one letter command code>-
        % <tot_count>.mat
        % This .mat file contains a single field named y
        save(save_filename, 'y');
        
        tot_count = tot_count + 1; % Increment # of files saved
        
    end
end

% End execution timing
toc

end