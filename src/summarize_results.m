function [] = summarize_results(datasets, classifier)
    algs = {'None', '$Fcal_1$', '$Fcal_2$', '$Fcal_3$', '$MAT$'};
    clrs = {'\cellcolor{red!50}', '\cellcolor{red!40}', '\cellcolor{red!30}', ...
        '\cellcolor{red!20}', '\cellcolor{red!10}'};
    stats = {'errors', 'fscores'};
    avgs = {'best', 'best3', 'all'};
            
    load(['outputs/raw_outputs/', classifier, '_atsd_results.mat']);
    %load(['outputs/raw_outputs/', classifier, '_matlab_results.mat']);

    ftypes = size(atsd_results.timers, 2);
    %errors = [atsd_results.atsd_errors matlab_results.matlab_errors];
    errors = atsd_results.errors_best;

    for k = 1:length(stats)*length(avgs)
        experiment = [stats{mod(k, length(stats))+1}, '_', avgs{mod(k, length(avgs))+1}];
        outfile = fopen(['outputs/', classifier, '_', experiment, '.txt'], 'w');
        errors = getfield(atsd_results, experiment);
        
        disp(['Results for ', experiment, ':']);
        [~, pZtest, ~, ranks] = friedman_demsar(errors, 'left', 0.1);
        mean_ranks = mean(ranks);

        fprintf(outfile, ['\\bf Data Set & \\bf Samples & \\bf Features ', ...
                            '& \\bf None & $Fcal_1$ & \\bf $Fcal_2$ & $Fcal_3$ \\\\\n']);
        for i = 1:length(datasets)
            data = load(['../ClassificationDatasets/csv/', datasets{i}, '.csv']);
            line = [datasets{i}, ' & ', num2str(size(data, 1)), ' & ', num2str(size(data, 2)-1)];

            for j = 1:ftypes
                line = [line, ' & ', clrs{floor(ranks(i,j))}, ' ', ...
                    num2str(round(10000*errors(i, j))/100), ' (', num2str(ranks(i,j)), ')'];
            end
            fprintf(outfile, '%s \\\\\n', line);
        end

        line = ' & &';
        for j = 1:ftypes
            line = [line, ' & ', num2str(mean_ranks(j))];
        end
        fprintf(outfile, '%s \\\\\n', line);

        line = '';
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
    disp('Results completed');
end
