function [] = atsd_experiment(datasets, params)
    classifier = params.classifier;
    numDatasets = length(datasets);
    numRuns = params.numRuns;
    ftypes = params.ftypes;

    % initialize result arrays
    timers = zeros(numDatasets, ftypes);

    errors_all = zeros(numDatasets, ftypes);
    errors_best = zeros(numDatasets, ftypes);
    errors_best3 = zeros(numDatasets, ftypes);

    fscores_all = zeros(numDatasets, ftypes);
    fscores_best = zeros(numDatasets, ftypes);
    fscores_best3 = zeros(numDatasets, ftypes);

    disp(['Running atsd_experiment using ', classifier]);
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
                pareto = anti_training(datatr, ftype, params);
                pareto_size = size(pareto, 1);
                timers(i, ftype) = timers(i, ftype) + toc;
                
                disp(['    Pareto front size: ', int2str(pareto_size)])

                err_all = 0;
                err_best = intmax;
                err_best3 = ones(1, min(3, pareto_size)) * double(intmax);

                fms_all = 0;
                fms_best = intmax;
                fms_best3 = ones(1, min(3, pareto_size)) * double(intmax);   

                for j = 1:pareto_size
                    % run desired classifier
                    model = blackbox(datatr, ...
                        'classifier', classifier, ...
                        'freeparams', pareto(j, :), ...
                        'dokfold', false);

                    yhat = predict(model, datate(:, 1:end-1));
                    stats = calcStats(yhat, datate(:, end));
                    err = 1 - stats.accuracy;
                    fms = stats.fscore;

                    % calc error stats
                    [err_max,idx] = max(err_best3);
                    if err < err_max
                        err_best3(idx) = err;
                    end

                    err_best = min(err_best3);
                    err_all = err_all + err;

                    % calc fscore stats
                    [fms_max,idx] = max(fms_best3);
                    if fms < fms_max
                        fms_best3(idx) = fms;
                    end

                    fms_best = min(fms_best3);
                    fms_all = fms_all + fms;
                end

                % update averages
                errors_all(i, ftype) = errors_all(i,ftype) + err_all/pareto_size;
                errors_best(i, ftype) = errors_best(i, ftype) + err_best;
                errors_best3(i, ftype) = errors_best3(i, ftype) + mean(err_best3);

                fscores_all(i, ftype) = fscores_all(i,ftype) + fms_all/pareto_size;
                fscores_best(i, ftype) = fscores_best(i, ftype) + fms_best;
                fscores_best3(i, ftype) = fscores_best3(i, ftype) + mean(fms_best3);

                % save averages
                atsd_results.errors_all = errors_all./n;
                atsd_results.errors_best = errors_best./n;
                atsd_results.errors_best3 = errors_best3./n;

                atsd_results.fscores_all = fscores_all./n;
                atsd_results.fscores_best = fscores_best./n;
                atsd_results.fscores_best3 = fscores_best3./n;

                atsd_results.timers = timers./n;
                save(['outputs/raw_outputs/', classifier, '_atsd_results.mat'], 'atsd_results');
            end
        end
    end
    
    %save results
    save(['outputs/raw_outputs/', classifier, '_atsd_results.mat'], 'atsd_results');
end
