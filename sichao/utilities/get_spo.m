function spo_array = get_spo(trial_matrix, fs, parse_scheme)

% PARAMETERS

% trial_matrix - EEG data matrix from single n second trial, of arbitrary
% number of rows

% fs - sampling rate of the EEG data files (Hz)

% parse_scheme - array representing parse scheme of EEG data matrix; must
% follow CWRU-KTH convention

%-------------------------------------------------------------------------%

% RETURNS

% spo_array - three element cell array containing subject, predicate, and
% object waveform arrays, respectively

%-------------------------------------------------------------------------%

% This function extracts the subject, predicate, and object from a matrix
% of EEG data. Each row in the matrix should correspond to a different
% sensor.

% This function returns a cell array with three entries. Each entry is a
% matrix with as many rows as there are sensors in trial_matrix and with as
% many columns as are necessary to contain the entirety of each S, P, or O.

%-------------------------------------------------------------------------%

% Initialize empty cell array
spo_array = {};

for j = 1:3  % Loop through trial and get S, P, O signals
    
    % Set boundaries of word
    word_start = fs * sum(parse_scheme(1:2*j-1));
    word_end = word_start + fs * parse_scheme(2*j); %% here, object should be 2 seconds, not one second.
    
    % Extract word waveform
    word_signals = trial_matrix(:, word_start:word_end-1);
    
    % Assign word waveform to appropriate location of spo_array
    spo_array{end + 1} = word_signals;
    
end