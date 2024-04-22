%% environement initialisation
%clear all;
%clc;
%

%% file fetching and data preparation
% signal_bank_loc = './signal_bank.mat';
% load(signal_bank_loc, 'filter_banks')
% fields = fieldnames(filter_banks);
% 
% % extraction of one data
% q=1;
% sensor = 3; %16 = numb of filters in filterbanks????requires explanation maybe
% f = 1; % 9 words ????? requires explanation maybe
% signal = 1; % 100 which is sample number ?????requires explanation maybe
% fs = 250; %maybe sampling frequency?





%% variables initialisation
framewidth = 30;
wave = "amor";%using morlet waves
% signals = filter_banks(sensor).(fields{f});
% sig = signals{signal};
%data = o.data(181443:181643,:);
data = o.data(15583-40:15583+40+40+50+40,:);
sig = data(:,3);

%% morlet transform of the data
[wt,~] = cwt(sig, wave, fs);
y = abs(wt);% 

limits= [round(min(min(y(1:35,:)))) round(max(max(y(1:35,:))))];

%wave starts 40 frames (200ms) after the beguinng of the task and stops 50
%later


%% heatmap plotting
[s,~] = size(sig);
for currentframe = 1:s-framewidth
%for currentframe = 1:20
if (130>=currentframe) && (currentframe>=80)
    currentframe
end
y_limited = y(1:35,:);
current_y = y_limited(:,currentframe:currentframe+30);
h = heatmap(current_y);
h.ColorLimits = limits;
h.Colormap = hot;
drawnow;
end
