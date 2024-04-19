%% This script monitors a user-specified folder for new EEG signal files
% being added. This script must be running before new files appear in the
% monitored directory.

% When a new file appears, the script:
%   1. Gets the parse scheme from the file name
%   2. Extracts the S, P, and O waveforms from the file for the user-specified sensors
%   3. Makes time-frequency images out of every waveform extracted in #2
%   4. Saves the time-frequency images to a user specified directory

% The .mat files in the monitored directory MUST be named using the
% CWRU-KTH temporal parse scheme (e.g. <anything>-1111123.mat). The names do not
% need to contain the ground truth commands. Additionally, every file must
% contain a single field named 'y', which contains all data from a given
% trial.

%%-------------------------------------------------------------------------%

clc
% General class code array
class_codes = 'spo';

% Directory that is monitored for new files
monitored_dir = 'eeg_from_headset';

% Directory that will contain time-frequency images made from incoming
% files in monitored_dir
save_dir = 'images_from_headset';

% The 'rows' variable should be the rows in the EEG data matrix that are
% desired for data processing. THIS MAY NOT BE ALIGNED WITH THE SENSOR
% NUMBERS. For example, if the data matrix has time in row 1 and then 16
% rows of EEG data, the 'rows' variable should contain 2 in order to access
% the first EEG channel.
rows = [16];

% Sampling rate
fs = 250;

% Establish current contents of monitored directory so that script reacts
% only to files added after script is launched.
old_contents = dir(fullfile(monitored_dir, '*.mat'));
% Intentionally infine loop; must Ctrl-C to escape
while true
    
    % Get contents of monitored directory again and check if anything new
    % has appeared
    new_contents = dir(fullfile(monitored_dir, '*.mat'));
    pause(10)
    dir_difference = setdiff({new_contents.name}, {old_contents.name});
    
    % If new files have been added, proceess each file by making
    % time-frequency images of their subjects, predicates, and objects
    if ~isempty(dir_difference)    
    t1=clock;
        % Loop through every new file
        for f = 1:numel(dir_difference)
            
            % Shortcut to get file name
            file = dir_difference{f};
            
            % Use file name to get parse scheme
            [~, parse_scheme] = get_trial_length_seconds(file);
            
            % Load in file
            eeg_data = load_eeg(fullfile(monitored_dir, file), fs, rows);
            
            % Split file into S, P, and O submatrices
            spo_array = get_spo(eeg_data, fs, parse_scheme);
            
            % Keep track of how many subplots have been made
            plotted = 1;
            
            % Loop through S, P, and O
            for element = 1:numel(spo_array)
                
                % Shortcut to get current matrix of interest (S, P, or O)
                current_class = spo_array{element};
                
                % Loop through every sensor in matrix and save CWT image
                for row = 1:size(current_class, 1)
                    
                    % Get CWT image
                    y = get_cwt_img(current_class(row, :),'amor', fs);
                    
                    % Plot CWT image in subplot
                    % Each row of plot is a word class
                    % Each column is row specified by 'rows' vector above,
                    % e.g. 3, 6, 11, 12
                    
                    figure(plotted)
                    %plot( numel(rows)/250, plotted)
                    %subplot(3, numel(rows), plotted)
                    imagesc(y)
                    axis tight
                    set(gcf,'color','w','position',[250.2000 288.6000 560 420]);% [490.2000 170.2000 560 420] %[4349 45 560 420]
                    set(gca,'position',[0.2200 0.240 0.50 0.6])
                    set(gca,'YScale','log')
                    yticks([10, 100])
                    xlabel('Sample','Fontname', 'Arial');
                    ylabel('Frequency (Hz)','Fontname', 'Arial');
                    plotted = plotted + 1;
                    pause(1.5)
                    
                    % Note that this plotting functionality will be less
                    % useful if multiple files are added to the monitored
                    % directory at once.
                    
                    % Save to file with name <sensor>-<class
                    % code>-<original file name>
                    save_filename = fullfile(save_dir, strcat(num2str(rows(row)), ...
                        '-', class_codes(element), '-', file));
                    
                    save(save_filename, 'y');
                    
                    fprintf("Saved %s\n", save_filename)
                    
                    % At this point the Python classification script
                    % should detect the new image and classify it
                    
                end
            end
        end
       t2=clock; 
       %disp(['runtime: ',num2str(etime(t2,t1))])
    else
        pause(1)
    end
    
    old_contents = new_contents;
end

