
clear;
clc;
load 'hlb_1111123-136.mat'
y=[y y];
save hlb_1111123-136.mat y
filename='hlb_1111123-136'
rows=1:16;
[trial_length_seconds, parse_scheme] = get_trial_length_seconds(filename);
processed_eeg_signals = load_eeg(filename, 250,rows);
trials=split_data_into_trials(processed_eeg_signals, 250,10);
spo_array = get_spo(trials(:,:,1), 250, parse_scheme);