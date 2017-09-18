function [] = summarize_results(datasets, params)
    classifier = params.classifier;
    optimizers = params.ftypes;
    
    load(['outputs/raw_outputs/', classifier, '_atsd_results.mat']);
    %load(['outputs/raw_outputs/', classifier, '_matlab_results.mat']);

    outfile = fopen(['outputs/', classifier, '_best_results.txt'], 'w');

    algs = {'None', '$Fcal_1$', '$Fcal_2$', '$Fcal_3$', '$MAT$'};
    clrs = {'\cellcolor{red!50}', '\cellcolor{red!30}', '\cellcolor{red!10}', ...
                '\cellcolor{yellow!25}', '\cellcolor{yellow!10}'};

    %errors = [atsd_results.atsd_errors matlab_results.matlab_errors];
    errors = atsd_results.atsd_errors_best;

    [~, pZtest, ~, ranks] = friedman_demsar(errors, 'left', 0.1);
    mean_ranks = mean(ranks);

    fprintf(outfile, ['\\bf Data Set & \\bf Samples & \\bf Features ', ...
                        '& \\bf None & $Fcal_1$ & \\bf $Fcal_2$ & $Fcal_3$ & MAT\\\\\n']);
    for i = 1:length(datasets)
        data = load(['../ClassificationDatasets/csv/', datasets{i}, '.csv']);
        line = [datasets{i}, ' & ', num2str(size(data, 1)), ' & ', num2str(size(data, 2)-1)];

        for j = 1:optimizers
            line = [line, ' & ', clrs{floor(ranks(i,j))}, ' ', num2str(round(10000*errors(i, j))/100), ' (', num2str(ranks(i,j)), ')'];
        end
        fprintf(outfile, '%s \\\\\n', line);
    end

    line = ' & &';
    for j = 1:optimizers
        line = [line, ' & ', num2str(mean_ranks(j))];
    end
    fprintf(outfile, '%s \\\\\n', line);

    line = '\n\n\n';
    for i = 1:optimizers
        line = [line, ' & ', algs{i}];
    end
    fprintf(outfile, '%s \\\\\n', line);

    for i = 1:optimizers
        line = algs{i};
        for j = 1:optimizers
            if i == j
                line = [line, ' & --'];
            else
                line = [line, ' & ', num2str(pZtest(i,j))];
            end
        end
        fprintf(outfile, '%s \\\\\n', line);
    end
end
