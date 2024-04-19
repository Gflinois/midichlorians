
function [trial_length_seconds, parse_scheme] = get_trial_length_seconds(filename)

% PARAMETERS

% filename - file name using CWRU-KTH naming convention

%-------------------------------------------------------------------------%

% RETURNS

% trial_length_seconds - sum of integers in parse scheme

% parse_scheme - array of integers representing subject, predicate, and
% object timing in file; follows CWRU-KTH convention

%-------------------------------------------------------------------------%

% This function uses the CWRU-KTH EEG file naming convention in order to
% retrieve a files temporal parse scheme as well as the total number of
% seconds each trial in said file lasts (which is the sum of the integers
% in the parse scheme.)

% example: filename='hlb_1111123-136', trial_length_seconds=10;
% parse_scheme=[1,1,1,1,1,2,3]
%-------------------------------------------------------------------------%

% Parse file name to get timing parse scheme (vector)
parse_scheme = regexp(filename, '\d+(?=-)', 'match');
parse_scheme = cell2mat(parse_scheme);
parse_scheme = split(parse_scheme, "");% split each chacter
parse_scheme = parse_scheme(2:end-1); % the first and last ones are null
parse_scheme = cell2mat(parse_scheme); % covert cell into matrix
parse_scheme = str2num(parse_scheme)'; % convert str to num

% By the CWRU-KTH naming convention, one trial's duration is the sum of the
% integers in the parse scheme
trial_length_seconds = sum(parse_scheme);



end
