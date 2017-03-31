function [datatr, datate] = splitData(randseed, data, split)

%Error checking
if(nargin < 3)
    error('splitData requires a random seed, a dataset, and a split percent');
end
if(max(size(randseed)) ~= 1)
    error('The random seed must be a single non-negative integer. If the value is not an integer, it will be rounded to the nearest integer');
end
if(or(split < 0.0, split > 1.0))
    error('The percent to train must be a real number between 0.0 and 1.0 inclusive');
end

backup_rng = rng;

%Overwrite the random number generator (defaults to Mersenne Twister)
rng(randseed);

numData = size(data, 1);

%Shuffle
data = data(randperm(numData), :);

%Split
%+1 ... -2 adjusts so that 0% train --> index 1, 100% train --> end-1.
train_end = 1 + round((numData - 2) * split); 

datatr = data(1:train_end, :);
datate = data(train_end+1:end, :);

%Restore backed up random number generator
rng(backup_rng);
end
