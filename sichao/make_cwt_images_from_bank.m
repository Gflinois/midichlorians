function make_cwt_images_from_bank(bank_loc, save_dir, fs)

% PARAMETERS

% bank_loc - location of bank to be used for image creation; every waveform
% in bank will be converted to a time-frequency image and saved elsewhere;
% bank should be created using make_eeg_filter_bank.m

% save_dir = save directory of time-frequency images generated from
% waveforms in bank

% fs - sampling rate of the bank waveforms (Hz)

%-------------------------------------------------------------------------%

% RETURNS

% N/A

%-------------------------------------------------------------------------%

% This function makes CWT images from an EEG filter bank m-file. This
% approach has the huge advantage of having a single load call instead of
% potentially thousands. The bank must have the same format as specified by
% make_eeg_filter_banks.m MATLAB function. You should have ran that
% function before running this one.

%-------------------------------------------------------------------------%

% Begin function execution timing
tic

% Empty destination directory
% while ~isempty(dir(append(save_loc, '*.mat')))
%     delete(append(save_loc, '*.mat'))
% end
% disp("Save location cleared")

% Load filter_banks variable from bank file; get field names
load(bank_loc, 'filter_banks')
fields = fieldnames(filter_banks); % 9×1 cell array; fields=word

% Load command words and codes from vocabulary file
load("matlab_code/signal_vocabulary.mat",...
    'subject_codes', 'subject_words',...
    'predicate_codes', 'predicate_words',...
    'object_codes', 'object_words')

% Variable to track how many files have been saved; ensures unique
% identifier for every saved file
tot_count = 1;

% Loop through every sensor
for sensor = 1:numel(filter_banks)% filter_banks is 1×16 struct with 9 fields
    
    % Loop through every word stored in sensor
    for f = 1:numel(fields)
        
        % Select name for current loop and store in standalone variable 
        field = fields{f};
        
        % Depending on the class of the current field, get array of words
        % and codes for that class
        if any(strcmp(field, subject_words))
            words = subject_words;
            codes = subject_codes;
        elseif any(strcmp(field, predicate_words))
            words = predicate_words;
            codes = predicate_codes;
        elseif any(strcmp(field, object_words))
            words = object_words;
            codes = object_codes;
        end
        
        % Get single-letter code for current field
        template_code = codes{strcmp(field, words)};
        
        % Select "stack" of templates using current field name
        template_stack = filter_banks(sensor).(field); % data of sensor ? with field (word) 
        
        % For every template in the "stack" do CWT and save image
        for template_index = 1:numel(template_stack)
            
            template = template_stack{template_index}; % Get current template
            
            % Get CWT image matrix
            y = get_cwt_img(template, fs);
            
            % Prepare save filename; has form <sensor>-<one letter command
            % code>-<tot_count>.mat
            save_filename = fullfile(save_dir, strcat(num2str(sensor), ...
                '-', template_code, '-', num2str(tot_count), '.mat'));
            
            % Save to file with name <sensor>-<one letter command code>-
            % <tot_count>.mat This .mat file contains a single field named
            % y
            save(save_filename, 'y');
            
            tot_count = tot_count + 1; % Increment # of files saved
            
        end
    end
end

% End execution timing
toc

fprintf("%d images successfully saved to %s.\n\n", tot_count - 1, save_dir)

end