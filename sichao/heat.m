%% environement initialisation
clear all;
clc;
%

%% file fetching and data preparation
signal_bank_loc = './signal_bank.mat';
load(signal_bank_loc, 'filter_banks')
fields = fieldnames(filter_banks);

% extraction of one data
q=1;
sensor = 3; %16 = numb of filters in filterbanks????requires explanation maybe
f = 1; % 9 words ????? requires explanation maybe
signal = 1; % 100 which is sample number ?????requires explanation maybe
fs = 250; %maybe sampling frequency?





%% variables initialisation
framewidth = 30;
wave = "amor";%using morlet waves
signals = filter_banks(sensor).(fields{f});
sig = signals{signal};

%% morlet transform of the data
[y, hz] = get_cwt_img(sig, wave, fs);% 
limits= [round(min(min(y(1:30,:)))) round(max(max(y(1:30,:))))];




%% heatmap plotting
%for currentframe = 1:250-framewidth
for currentframe = 1:240
y_limited = y(1:30,:);
current_y = y_limited(:,currentframe:currentframe+30);
h = heatmap(current_y);
h.ColorLimits = limits;
h.Colormap = hot;
drawnow;
end
