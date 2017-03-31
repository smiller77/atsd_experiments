function [model] = blackbox(datatr, varargin)

args = struct('classifier','svm', 'freeparams',[], 'dokfold',false);
argNames = fieldnames(args);

for pair = reshape(varargin, 2, [])
    arg = lower(pair{1});
    if any(strcmp(arg, argNames))
        args.(arg) = pair{2};
    else
       error('Unknown parameter %s', arg);
    end
end

if(args.dokfold)
    c = cvpartition(size(datatr, 1), 'KFold', 10);
else
    c = [];
end


if(strcmp(args.classifier, 'svm'))
    model = fitcsvm(datatr(:, 1:end-1), datatr(:, end), ...
		'BoxConstraint', args.freeparams(1), ...
		'KernelFunction', 'rbf', ...
		'KernelScale', args.freeparams(2), ...
		'Solver', 'SMO', ...
		'KKTTolerance', 1e-4, ...
		'IterationLimit', 1e5, ...
		'CVPartition', c);
end

if(strcmp(args.classifier, 'knn'))
    model = fitcknn(datatr(:, 1:end-1), datatr(:, end), ...
		'NumNeighbors', round(args.freeparams(1)), ...
		'Distance', 'euclidean', ...
		'CVPartition', c);
end

if(strcmp(args.classifier, 'dtree'))
    model = fitctree(datatr(:, 1:end-1), datatr(:, end), ...
		'MaxNumSplits', round(args.freeparams(1)), ...
		'MinLeafSize', round(args.freeparams(2)), ...
		'SplitCriterion', 'gdi', ...
		'CVPartition', c);
end
%{
if(strcmp(args.classifier, 'rforest'))
    options = statset('UseParallel', true);
    forest = TreeBagger(round(args.freeparams(1)), datatr(:, 1:end-1), datatr(:, end), ...
        'MinLeafSize', round(args.freeparams(2)), ...
        'Method', 'classification', ...
        'Options', options);
    yhat = str2double(predict(forest, datate(:, 1:end-1)));
end
%}
end
