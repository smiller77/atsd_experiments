function [] = atsd_main_experiment(datasets, params)
    classifier = params.classifier;
    numDatasets = length(datasets);
    numRuns = params.numRuns;
    ftypes = params.ftypes;

    % initialize result arrays
    timers_moo = zeros(numDatasets, ftypes);
    all_errors_moo = zeros(numDatasets, ftypes);
    count_errors_moo = zeros(numDatasets, ftypes);
    all_fms_moo = zeros(numDatasets, ftypes);

    for i = 1:numDatasets
        disp(['  -> Running ', datasets{i}]);
        for ftype = 1:ftypes
            disp(['    Ftype ', num2str(ftype)]);
            for n = 1:numRuns
                disp(['    Average ', num2str(n), ' of ', num2str(numRuns)]);

                data = load([datasets{i}, '.csv']);
                if length(unique(data(:, end))) ~= 2
                    error('Must be a two class problem.');
                end
                [datatr, datate] = splitData(n, data, params.split);

                tic;
                x = anti_training(datatr, ftype, params);
                timers_moo(i, ftype) = timers_moo(i, ftype) + toc;

                err_best = intmax;

                for j = 1:size(x, 1)
                    % run desired classifier
                    model = blackbox(datatr, ...
                        'classifier', classifier, ...
                        'freeparams', x(j, :), ...
                        'dokfold', false);

                    yhat = predict(model, datate(:, 1:end-1));
                    err = calcError(yhat, datate(:, end));

                    if err < err_best
                        err_best = err;
                        stats = calcStats(datate(:, end), yhat);
                        fms_best = mean(stats.fscore);
                    end
                end

                all_fms_moo(i, ftype) = all_fms_moo(i, ftype) + fms_best;
                all_errors_moo(i, ftype) = all_errors_moo(i, ftype) + err_best;
                count_errors_moo(i, ftype) = count_errors_moo(i, ftype) + 1;

                save(['outputs/raw_outputs/', classifier, '_atsd_optimizer.mat']);
            end
        end
    end

    save(['outputs/raw_outputs/', classifier, '_atsd_optimizer.mat']);
end
