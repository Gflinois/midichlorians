function make_cwt_image_bank(waveform_bank_loc, fs)

% TODO: Update documentation

% PARAMETERS

% waveform_bank_loc - location of bank of EEG waveforms to be used for
% time-frequency image creation; every waveform in bank will be converted
% to a time-frequency image and saved in another bank; waveform bank should
% be created using make_eeg_filter_bank.m

% save_loc = save directory for raw and average image banks

% fs - sampling rate of the waveform bank entries (Hz)

%-------------------------------------------------------------------------%

% RETURNS

% N/A

%-------------------------------------------------------------------------%

% This function makes CWT images from an EEG waveform bank m-file. The
% time-frequency images are saved to a bank with an identical structure to
% the bank from which waveforms are read. The waveform bank must have the
% same format as specified by make_eeg_filter_banks.m MATLAB function. You
% should have ran that function before running this one.

%-------------------------------------------------------------------------%

% Load filter_banks variable from waveform bank file; get field names
load(waveform_bank_loc, 'filter_banks')
fields = fieldnames(filter_banks);% fields = fieldnames(S) returns the 
%field names of the structure array S in a cell array.

% Copy filter bank struct to initialize bank of images for easier
% processing
raw_image_banks = filter_banks;
avg_image_banks = raw_image_banks;

% Loop through every chosen sensor
for sensor = 1:numel(filter_banks)% length of element of filte_banks
    
    % Loop through every word stored in sensor
    for f = 1:numel(fields)
        
        % Select name for current loop and store in standalone variable
        field = fields{f}; 
        
        % Select "stack" of templates using current field name
        template_stack = filter_banks(sensor).(field); % sample number of the field for sensor (channel)
        
        % For every template in the "stack" do CWT and save image
        for template_index = 1:numel(template_stack)% numel(template_stack)=100
            
            template = template_stack{template_index}; % Get current template, and template is 1*250
            
            % Get CWT image matrix
            [y, ~] = get_cwt_img(template, 'amor', fs);
            
            if ~isequal(size(y), [51, 250]) % judge the size of y equals to [51,250]
                y = imresize(y, [51, 250]); % resize y as [51,250]
            end
            
            % Assign CWT image to image stack
            raw_image_banks(sensor).(field){template_index} = y;
            
        end
        %size([filter_banks(1).(fields{1}){:}])=1*25000;
        %numel(raw_image_banks(sensor).(field))=100;
        %size(raw_image_banks(sensor).(field){1})=1*250;
        % calculate the mean of image bank of each channel with field
        % (word)in 3rd dimension, and size of the 3rd dimension will be 1.
        avg_image_banks(sensor).(field) = mean(reshape([raw_image_banks(sensor).(field){:}], [size(raw_image_banks(sensor).(field){1}), numel(raw_image_banks(sensor).(field))]), 3);
        
    end
end

% Save bank of images to save_loc
[filepath,~,~] = fileparts(waveform_bank_loc);
save(fullfile(filepath, 'raw_image_bank.mat'), 'raw_image_banks', '-v7.3')
save(fullfile(filepath, 'avg_image_bank.mat'), 'avg_image_banks', '-v7.3')

end