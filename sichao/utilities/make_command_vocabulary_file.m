function make_command_vocabulary_file()

% PARAMETERS

% N/A

%-------------------------------------------------------------------------%

% RETURNS

% N/A

%-------------------------------------------------------------------------%

% This function generates a master vocabulary mat file of all possible
% robot commands and their corresponding encodings. This script MUST be
% updated when new words are added to commands.

subject_words = {'human', 'robot'};
subject_codes = {'h', 'r'};

predicate_words = {'colassy', 'insert', 'pick', 'place'};
predicate_codes = {'a', 'i' 'p', 'l'};

object_words = {'block', 'cylinder', 'tube'};
object_codes = {'b', 'c', 't'};

save("matlab_code/signal_vocabulary.mat", "subject_words", "subject_codes", ...
     "predicate_words", "predicate_codes", "object_words", "object_codes")