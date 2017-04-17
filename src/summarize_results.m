function [] = summarize_results(optimizer);

clc
clearvars -except optimizer
close all;

addpath utils/
load(['outputs/raw_outputs/', optimizer, '_atsd_optimizer.mat']);
load(['outputs/raw_outputs/', optimizer, '_matlab_optimizer.mat']);
clearvars -except all_errors_moo all_errors_mat count_errors_moo count_errors_mat datasets optimizer

outfile = fopen(['outputs/', optimizer, '_results.txt'], 'w');

algs = {'None', '$Fcal_1$', '$Fcal_2$', '$Fcal_3$', '$MAT$'};
tail = 'left';
alpha = 0.1;

clrs = {'\cellcolor{red!50}', '\cellcolor{red!30}', '\cellcolor{red!10}', '\cellcolor{yellow!25}', '\cellcolor{yellow!10}'};
% clrs = {' ', ' ', ' ', ' ', ' '};

errors = [all_errors_moo(:, 1:end-1) all_errors_mat];
counts = [count_errors_moo(:, 1:end-1) count_errors_mat];
errors = errors./counts;

[hZtest, pZtest, pFtest, ranks] = friedman_demsar(errors, tail, alpha);
mean_ranks = mean(ranks);


fprintf(outfile, '\\bf Data Set & \\bf Samples & \\bf Features & \\bf None & $Fcal_1$ & \\bf $Fcal_2$ & $Fcal_3$ & MAT\\\\\n');
for i = 1:length(datasets)
	st = datasets{i};
	data = load(['../ClassificationDatasets/csv/', datasets{i}, '.csv']);
	st = [st, ' & ', num2str(size(data, 1)), ' & ', num2str(size(data, 2)-1)];
	
	for j = 1:size(errors, 2)
		st = [st, ' & ', clrs{floor(ranks(i,j))},' ', num2str(round(10000*errors(i, j))/100), ' (', num2str(ranks(i,j)), ')'];
	end
	fprintf(outfile, '%s \\\\\n', st);
end

st = ' & &';
for j = 1:size(errors, 2)
	st = [st, ' & ', num2str(mean_ranks(j))];
end
fprintf(outfile, '%s \\\\\n\n\n\n', st);

st = [''];
for i = 1:size(errors, 2)
	st = [st, ' & ', algs{i}];
end
fprintf(outfile, '%s \\\\\n', st);

for i = 1:size(errors, 2)
	st = algs{i};
	for j = 1:size(errors, 2)
		if i == j
			st = [st, ' & --'];
		else
			st = [st, ' & ', num2str(pZtest(i,j))];
		end
	end
	fprintf(outfile, '%s \\\\\n', st);
end
