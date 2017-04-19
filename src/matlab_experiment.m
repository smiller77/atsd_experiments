function [errors, timers] = matlab__experiment(datasets, params)
    classifier = params.classifier;
    numDatasets = length(datasets);
    numRuns = params.numRuns;

    % initialize result arrays
    timers = zeros(numDatasets, 1);
    errors = zeros(numDatasets, 1);
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
            timers(i) = timers(i) + toc;

            % run desired classifier
            model = blackbox(datatr, ...
                'classifier', classifier, ...
                'freeparams', x);
            yhat = predict(model, datate(:, 1:end-1));
            err = calcError(yhat, datate(:, end));

            stats = calcStats(datate(:, end), yhat);
            fms_best = mean(stats.fscore);

            all_fms_mat(i) = all_fms_mat(i) + fms_best;
            errors(i) = errors(i) + err;

			results.matlab_errors = errors./n;
			results.matlab_timers = timers./n;
            save(['outputs/raw_outputs/', classifier, '_matlab_optimizer.mat'], results);
        end
    end

    results.matlab_errors = errors./numRuns;
    results.matlab_timers = timers./numRuns;
    save(['outputs/raw_outputs/', classifier, '_matlab_optimizer.mat'], results);
end
