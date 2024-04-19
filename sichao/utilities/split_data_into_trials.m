function reshaped_eeg = split_data_into_trials(data_mat, fs, trial_length_seconds)

% PARAMETERS

% data_mat - data matrix of excessive length in the 2nd dimension (columns)

% fs - sampling rate of the EEG waveform data (Hz)

% trial_length_seconds - the duration of one trial (seconds)

%-------------------------------------------------------------------------%

% RETURNS

% reshaped_eeg - tensor with a 1st dimension equal to data_mat, with a
% consistent 2nd dimension, and with a 3rd dimension equal to the maximum
% number of equally-sized submatrices that can be made from data_mat given
% that each submatrix must have the same 1st and 2nd dimension

%-------------------------------------------------------------------------%

% This function recieves a data matrix of arbitrtary height and width and
% splits that matrix into submatrices with the same number of rows as the
% original matrix and with sampling_rate * trial_length_seconds columns.

% The function returns a tensor with the above width and height, as well as
% whatever depth is needed in order to contain all of the submatrices.

% If the number of columns in the original data matrix is not evenly
% divisible by sampling_rate * trial_length_seconds, columns are removed
% from the original matrix until this division returns an integer.

%-------------------------------------------------------------------------%

% Determine how long one trial is in samples
trial_length_samples = fs * trial_length_seconds;

% Cut data off end of file so that file length is an exact multiple of
% trial_length_samples
samples_off_back = mod(size(data_mat, 2), trial_length_samples);% return the remainder of size(data_mat, 2) dividing trial_length_samples.
trimmed_data = data_mat(:, 1:end-samples_off_back);

% Get number of matrix rows here to shorten below expression
num_rows = size(trimmed_data, 1);

% Reshape EEG to be a 3D stack of trials and return this stack, reshape
% trimmed_data as [row, trial_lenghth_samples, deepth ]
reshaped_eeg = reshape(trimmed_data, num_rows, trial_length_samples, []);
    
end
