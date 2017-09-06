function [] = atsd_experiment(datasets, params)
    classifier = params.classifier;
    numDatasets = length(datasets);
    numRuns = params.numRuns;
    ftypes = params.ftypes;

    % initialize result arrays
    timers = zeros(numDatasets, ftypes);
    errors = zeros(numDatasets, ftypes);
    all_fms_moo = zeros(numDatasets, ftypes);

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
                timers(i, ftype) = timers(i, ftype) + toc;

				disp(['    Pareto front size: ', size(pareto, 1)])

				err_all = 0;
                err_best = intmax;
				err_best3 = [intmax intmax intmax];

                for j = 1:size(pareto, 1)
                    % run desired classifier
                    model = blackbox(datatr, ...
                        'classifier', classifier, ...
                        'freeparams', pareto(j, :), ...
                        'dokfold', false);

                    yhat = predict(model, datate(:, 1:end-1));
                    err = calcError(yhat, datate(:, end));

					[err_max,idx] = max(err_best3)
					if err < err_max:
						err_best3(idx) = err;
					end

                    err_best = min(err_best3);
					err_all = err_all + err;
                end
                %all_fms_moo(i, ftype) = all_fms_moo(i, ftype) + fms_best;
				errors_all(i, ftype) = errors_all(i,ftype) + ...
											err_all/size(pareto,1);
                errors_best(i, ftype) = errors_best(i, ftype) + err_best;
				errors_best_3(i, ftype) = errors_best_3(i, ftype) + ...
											mean(err_best3);
				% save averages
				results.atsd_errors_all = errors_all./n
				results.atsd_errors_best = errors_best./n;
				results.atsd_errors_best_3 = errors_best_3./n;
				results.atsd_timers = timers./n;
                save(['outputs/raw_outputs/', classifier, ...
							'_atsd_results.mat'], results);
            end
        end
    end
    
	%save results
    save(['outputs/raw_outputs/', classifier, '_atsd_results.mat'], results);
end
