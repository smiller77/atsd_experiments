function [] = matlab_main_experiment(datasets, params)

classifier = params.classifier;
numDatasets = length(datasets);
numRuns = params.numRuns;

% initialize result arrays
timers_mat = zeros(numDatasets, 1);
all_errors_mat = zeros(numDatasets, 1);
count_errors_mat = zeros(numDatasets, 1);
all_fms_mat = zeros(numDatasets, 1);

for i = 1:numDatasets
	disp(['  -> Running ', datasets{i}])
  
    for n = 1:numRuns
		disp(['    Average ', num2str(n), ' of ', num2str(numRuns)]);
        
		data = load([datasets{i}, '.csv']);
		if length(unique(data(:, end))) ~= 2
			error('Must be a two class problem.');
		end
		[datatr, datate] = splitData(n+10, data, params.split);
        
        tic;
        x = matlab_search(datatr, params);
        timers_mat(i) = timers_mat(i) + toc;
               
        % run desired classifier
        model = blackbox(datatr, ...
            'classifier', classifier, ...
            'freeparams', x);
        yhat = predict(model, datate(:, 1:end-1));
		err = calcError(yhat, datate(:, end));

        stats = calcStats(datate(:, end), yhat);
        fms_best = mean(stats.fscore);
      
        all_fms_mat(i) = all_fms_mat(i) + fms_best;
        all_errors_mat(i) = all_errors_mat(i) + err;
        count_errors_mat(i) = count_errors_mat(i) + 1;
        
	save(['outputs/raw_outputs/', classifier, '_matlab_optimizer.mat']);
    end
end

save(['outputs/raw_outputs/', classifier, '_matlab_optimizer.mat']);
end
