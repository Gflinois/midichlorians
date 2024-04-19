function processed_eeg_signals = load_eeg(filename, fs, rows)

% PARAMETERS

% filename - .mat file named using CWRU-KTH naming convention; this file
% must contain a field named 'y' and that field must contain an EEG data
% matrix of arbitrary size

% fs - sampling rate of the bank waveforms (Hz)

% rows - vector of row indices to load from file specified by filename;
% these may differ from sensor indices as they are dependent on the format
% of the EEG data files

%-------------------------------------------------------------------------%

% RETURNS

% N/A

%-------------------------------------------------------------------------%

% This function loads in a mat file containing EEG data and extracts
% user-selected rows from that file. It also conducts independent signal
% processing operations on every row of the resulting matrix.

% The mat file must have a field named 'y' and that field must contain a
% matrix with an abritrary number of rows.

% The user selects which rows of the data matrix to extract using the
% rows parameter, which must be a vector of integers representing the rows
% of the data matrix to be loaded in.

%example: filename='hlb_1111123-136';processed_eeg_signals=rows_size*2500
%eeg signals, aned it is also a 10-second signals
%-------------------------------------------------------------------------%

% Load in file to be tested
load(filename, 'y');
raw_eeg_signals = y(rows, :);

% Process using processing function
processed_eeg_signals = process_mat(raw_eeg_signals, fs);

    function processed_mat = process_mat(mat, fs)
        
        % This subfunction can perform row-wise signal processing operations on
        % any matrix it is passed.
        
        % This function must be directly edited in order to implement the desired
        % pre-processing operations.
        
        %-------------------------------------------------------------------------%
        
        processed_mat = mat;
        % processed_mat = normalize(processed_mat, 2, 'range', [-1, 1]);
        
        freq_range = [12, 50];
        if max(freq_range) > fs/2
            error("Frequency range exceeds Nyquist limit.")
        end
        
        % One-liners %
        if true
            % Bandpass
            %     processed_mat = bandpass(processed_mat', freq_range, fs)';
            
            % PCA
            %     [~,score] = pca(processed_mat');
            %     processed_mat = score';
        end
        
        % Need for loops %
        for row = 1:size(processed_mat, 1)
            
            % CWT
            %     [wt, f] = cwt(processed_mat(row, :), fs);
            %     processed_mat(row, :) = icwt(wt, f, freq_range);
            
            % EMD
            %     imf_max = 1;
            %     decomp = emd(processed_mat(row, :), 'MaxNumIMF', imf_max, 'Display', 0);
            %     processed_mat(row, :) = decomp(:, imf_max)';
            
        end
        
    end

end