function [cwt_img, hz] = get_cwt_img(signal, wavelet_name, fs)

% PARAMETERS

% signal - 1D waveform to be turned into time-frequency image

% fs - sampling rate of the waveform (Hz)

%-------------------------------------------------------------------------%

% RETURNS

% cwt_img - 2D array containing time frequency image with time steps in the
% 1st dimension and frequency steps in the 2nd dimension

%-------------------------------------------------------------------------%

% This function performs a CWT on the provided signal and returns the
% matrix representing the CWT image

%-------------------------------------------------------------------------%

% Do CWT
[wt, hz] = cwt(signal, wavelet_name, fs);

% Make image array
cwt_img = abs(wt);

end