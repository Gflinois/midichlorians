%-------------------------------------------------------------------------%
% this fucntuon is to calculate the energy-entropy ratio of different
% wavelets

signal_bank_loc = './signal_bank.mat';
fs = 250;

load(signal_bank_loc, 'filter_banks')
fields = fieldnames(filter_banks);

wavelets = {'amor', 'morse', 'bump'};

avg_ratio_mat = zeros(51, numel(wavelets));% numel: total number of elements
var_ratio_mat = zeros(51, numel(wavelets));% numel(wavelets)=3

hz_mat = [];

for w = 1:size(avg_ratio_mat, 2) % w=1:3
    
    wave = wavelets{w};
    
    energy_mat = [];
    entropy_mat = [];
    
    for sensor = 1:numel(filter_banks)% numel(filter_banks)=16
        for f = 1:numel(fields)% numel(fields)=9 words
            signals = filter_banks(sensor).(fields{f});
            for signal = 1:numel(signals)% numel(signals)=100,which is the sample number. 
                
                % Get CWT image matrix;y is the is a matrix with the number
                %of rows equal to the length of scales and number of columns
                %equal to the length of the input signal. The k-th row of coefs
                %corresponds to the CWT coefficients for the k-th element in the scales vector.
                %Hz is frequencies of the CWT, returned as a vector, and
                %size is the same as the size of the scale
                
                [y, hz] = get_cwt_img(signals{signal}, wave, fs);% 

             
                
                if numel(hz) == 51
                    hz_mat = [hz_mat, hz];
                end
                
                if ~isequal(size(y), [51, 250])
                    % returns image y that has the number of rows and columns
                    %specified by the two-element vector [numrows numcols].
                    y = imresize(y, [51, 250]);
                end
                %compute the sum along the second dimension and total
                %energy in sliding window W at j scale y=[51,250];
                img_energy = sum(y.^2, 2); % img_energy=[51,1]
                % total energy in sliding window at all scale
                p = y.^2 ./ img_energy; % p_size=51*250
                % wavelet energy Shannon entropy
                img_entropy = -sum(p .* log2(p), 2); % img_entropy_size=51*1
                
                energy_mat = [energy_mat, img_energy];% energy_mat_size=51*(1*signal*f*sensor); 
                entropy_mat = [entropy_mat, img_entropy]; % entropy_mat_size=51*(1*signal*f*sensor); 
                  
            end
        end
    end
    
    ratio_mat = energy_mat ./ entropy_mat;% ratio_mat_size=51*(1*signal*f*sensor);
    
    avg_ratio_mat(:, w) = mean(ratio_mat, 2); % avg_ratio_mat_size=51*3
    ratio_sd = std(ratio_mat, [], 2);
    lower_ebars = avg_ratio_mat(:, w) - min(ratio_mat, [], 2);
    upper_ebars = avg_ratio_mat(:, w) + min(ratio_mat, [], 2);
    
    hold on
    %a scatter plot with circular markers at the locations specified by the vectors x and y
    scatter(hz_mat(:, 1), avg_ratio_mat(:, w), 75, 'Filled', 'o')
    
end

xlabel("Frequency (Hz)")
ylabel("Energy-to-Entropy Ratio")
legend("Morlet", "Morse", "Bump")
set(gca, 'FontSize', 12)
grid()