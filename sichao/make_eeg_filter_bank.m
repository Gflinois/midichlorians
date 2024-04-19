function make_eeg_filter_bank(source_dir, save_loc, fs)

% PARAMETERS

% source_dir - directory containing EEG data files which will be parsed and
% included in the filter bank; ; every file must contain a field named 'y'
% and that field must contain an EEG data matrix of arbitrary size

% save_loc - the save location of the filter bank, down to the .mat file
% name (and including the .mat extension)

% fs - sampling rate of the EEG data files (Hz)

%-------------------------------------------------------------------------%

% RETURNS

% N/A

%-------------------------------------------------------------------------%

% This function makes a single filter bank file containing a filter bank
% struct. That struct has n rows (one for each EEG sensor) and one column
% for each command vocabulary word. Each entry in the bank is a tensor of
% templates for the word-sensor combination at that entry.

% This function has the capability of splitting longer files into shorter
% trials before extracting template wavforms. It should therefore be
% pointed at a directory with a few long EEG files instead of a directory
% of several thousand single-trial files. Pointing it at the latter will be
% excruciatingly slow due to the thousands of needed load operations.

%-------------------------------------------------------------------------%

%% Get bank generation files
file_pattern = fullfile(source_dir, '*.mat'); % Regex-ish equivalent. return file paths 
files = dir(file_pattern); % Get all files from above directory matching pattern

%% Hardcode in loading all 16 sensors
rows = 2:17;

%% Initialize filter banks

% Ensure that vocabulary file exists by running generation script
make_command_vocabulary_file()

% Load in a .mat file containing all words and word codes
load("matlab_code/signal_vocabulary.mat",...
    'subject_codes', 'subject_words',...
    'predicate_codes', 'predicate_words',...
    'object_codes', 'object_words')

% Make a struct object
% Struct has row_stop - row_start rows (one for each sensor)
% Struct has one column ("field") for each command word
% Each struct cell therefore stores all of the "template" waveforms for
%   sensor <row> and word <col> (storage is a "stack" in the 3rd dimension)
% Each cell of struct is initialized to be an empty cell array
% Cell arrays MUST be used to accomodate different word durations

filter_banks(numel(rows)) = struct();

all_words = [subject_words, predicate_words, object_words];% human robot .... cyliner 
all_codes = [subject_codes, predicate_codes, object_codes];% h robot .....c 

% Loop through all words to initialize struct columns
for word = all_words
    [filter_banks.(word{1})] = deal({});
end

tot_filters = 0;

%% Populate filter banks

% Loops through all files in source_dir
for f = 1:length(files)
    
    % Load file, extract array of EEG data only
    file = files(f).name;
    eeg_data = load_eeg(fullfile(source_dir, file), fs, rows);% read eeg signals of the 'file', and size=16*2500
    
    % Parse file name to get three-letter command code (cell array)
    % file=hlb_1111123-136, command_code='hlb'
    command_code = cell2mat(regexp(file, '\w+(?=_)', 'match'));% 
    
    % Get length of each trial in file
    [trial_length_seconds, parse_scheme] = get_trial_length_seconds(file);
    
    % Split the EEG data into n equal-length trials; trials=16*2500*(eeg_data/2500)
    trials = split_data_into_trials(eeg_data, fs, trial_length_seconds);
    
    % Loop through every trial present in file
    for i = 1:size(trials, 3) % number of reshaped files
        
        % Select trial currently being processed
        trial = trials(:, :, i);
        
        % Get S, P, and O signals from trial for all sensors at once; 1*3
        % cell;16*250,16*250,16*500 in each cell
        spo_array = get_spo(trial, fs, parse_scheme);
        
        for j = 1:numel(spo_array)  % Loop through trial and get S, P, O signals
            code = command_code(j);  % Grab single-letter word code
            
            % Depending on the class of the current code, get array of words
            % and codes for that class
            if any(strcmp(code, subject_codes))% compare whether two strs equal: code?=subject
                codes = subject_codes;
            elseif any(strcmp(code, predicate_codes))% code?=suject
                words = predicate_words;
                codes = predicate_codes;
            elseif any(strcmp(code, object_codes))
                words = object_words;
                codes = object_codes;
            end
            
            % Get single-letter code for current field;
            word = words{strcmp(code, codes)};
            
            % Loop through every EEG channel and assign word signal to bank
            % filter_banks(sensor).(word){1}=1*250;spo_array{j}(sensor,
            % :)=1*250;
            for sensor = 1:size(spo_array{j}, 1)
                filter_banks(sensor).(word){end + 1} = spo_array{j}(sensor, :);
                tot_filters = tot_filters + 1;
            end
            
        end
        
    end
    
end

%% Clean up struct and vocabulary file if a word wasn't used

max_filter_depth = Inf;

for i = 1:length(all_words)
    
    if ~any(size(filter_banks(1).(all_words{i})))
        filter_banks = rmfield(filter_banks, all_words{i});% delete all_words{i} from struct filter_banks
        
        subject_words(ismember(subject_words, all_words{i})) = [];% ismember 
        subject_codes(ismember(subject_codes, all_codes{i})) = [];
        
        predicate_words(ismember(predicate_words, all_words{i})) = [];
        predicate_codes(ismember(predicate_codes, all_codes{i})) = [];
        
        object_words(ismember(object_words, all_words{i})) = [];
        object_codes(ismember(object_codes, all_codes{i})) = [];
        
    elseif numel(filter_banks(1).(all_words{i})) < max_filter_depth
        max_filter_depth = numel(filter_banks(1).(all_words{i}));
    end
    
end

save("matlab_code/signal_vocabulary.mat", "subject_words", ...
    "subject_codes", "predicate_words", "predicate_codes", ...
    "object_words", "object_codes")

%% Enforce that all filter banks are the same size
for i = 1:length(all_words)
    
    if isfield(filter_banks(1), all_words{i})
        for sensor = 1:size(eeg_data, 1)
            filter_banks(sensor).(all_words{i}) = filter_banks(sensor).(all_words{i})(1:max_filter_depth);
        end
    end
    
end
%% Save final filter bank struct

save(save_loc, 'filter_banks')

fprintf("%d total filters successfully saved to bank.\n\n", tot_filters)

end