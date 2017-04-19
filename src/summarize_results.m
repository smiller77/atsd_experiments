function [] = summarize_results(datasets, params)
    classifier = params.classifier;
    ftypes = params.ftypes;
    
    % ======= old file names ======= %
    load(['outputs/raw_outputs/', classifier, '_atsd_optimizer.mat']);
    load(['outputs/raw_outputs/', classifier, '_matlab_optimizer.mat']);
    % ============================== %

    outfile = fopen(['outputs/', classifier, '_results.txt'], 'w');

    algs = {'None', '$Fcal_1$', '$Fcal_2$', '$Fcal_3$', '$MAT$'};
    clrs = {'\cellcolor{red!50}', '\cellcolor{red!30}', '\cellcolor{red!10}', ...
                '\cellcolor{yellow!25}', '\cellcolor{yellow!10}'};

    %errors = [atsd_errors matlab_errors];
    errors = [all_errors_moo(:, 1:end-1) errors];
    errors = errors./10;
    
    [~, pZtest, ~, ranks] = friedman_demsar(errors, 'left', 0.1);
    mean_ranks = mean(ranks);

    fprintf(outfile, ['\\bf Data Set & \\bf Samples & \\bf Features ', ...
                        '& \\bf None & $Fcal_1$ & \\bf $Fcal_2$ & $Fcal_3$ & MAT\\\\\n']);
    for i = 1:length(datasets)
        data = load(['../ClassificationDatasets/csv/', datasets{i}, '.csv']);
        line = [datasets{i}, ' & ', num2str(size(data, 1)), ' & ', num2str(size(data, 2)-1)];

        for j = 1:ftypes
            line = [line, ' & ', clrs{floor(ranks(i,j))}, ' ', num2str(round(10000*errors(i, j))/100), ' (', num2str(ranks(i,j)), ')'];
        end
        fprintf(outfile, '%s \\\\\n', line);
    end

    line = ' & &';
    for j = 1:ftypes
        line = [line, ' & ', num2str(mean_ranks(j))];
    end
    fprintf(outfile, '%s \\\\\n', line);

    line = '\n\n\n';
    for i = 1:ftypes
        line = [line, ' & ', algs{i}];
    end
    fprintf(outfile, '%s \\\\\n', line);

    for i = 1:ftypes
        line = algs{i};
        for j = 1:ftypes
            if i == j
                line = [line, ' & --'];
            else
                line = [line, ' & ', num2str(pZtest(i,j))];
            end
        end
        fprintf(outfile, '%s \\\\\n', line);
    end
end
