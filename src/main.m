function [] = main(optimizer);

clc;
clearvars -except optimizer;
close all;

addpath('atsd/');
addpath('utils/');
addpath('../ClassificationDatasets/csv/');

datasets = {
  %'bank';
  'blood';
  'breast-cancer-wisc-diag';
  'breast-cancer-wisc-prog';
  'breast-cancer-wisc';
  'breast-cancer';
  'congressional-voting';
  'conn-bench-sonar-mines-rocks';
  'credit-approval';
  'cylinder-bands';
  'echocardiogram';
  'fertility';
  'haberman-survival';
  'heart-hungarian';
  'hepatitis';
  'ionosphere';
  'mammographic';
  'molec-biol-promoter';
  'musk-1';
  'oocytes_merluccius_nucleus_4d';
  'oocytes_trisopterus_nucleus_2f';
  %'ozone';
  %'parkinsons';
  %'pima';
  %'pittsburg-bridges-T-OR-D';
  %'planning';
  %'ringnorm';
  %'spambase';
  %'spectf_train';
  %'statlog-australian-credit';
  %'statlog-german-credit';
  %'statlog-heart';
  %'titanic';
  %'twonorm';
  %'vertebral-column-2clases'
  };

params.classifier = optimizer;   % classifier to run experiments on
params.numRuns = 10;          % number of times to run experiment
params.split = 0.8;        % percentage of data to be used for training
params.ftypes = 5;        % number of sacrificial set types
params.moo = 1;         % multi-objecive or single objective

% determine number of free parameters
switch params.classifier
    case 'svm'
		params.nvars = 2;
		params.lb = [1e-1; 1e-2];
		params.ub = [1000; 5];
    case 'knn'
		params.nvars = 1;
		params.lb = [1];
		params.ub = [100];
    case 'dtree'
		params.nvars = 2;
		params.lb = [1; 1];
		params.ub = [50; 100];
    %{case 'rforest'
		params.nvars = 2;
		params.lb = [1; 1];
		params.ub = [200; 100];%}
    otherwise
        error('Unknown classifier %s', params.classifier);
end

% open up parallel pool
delete(gcp('nocreate'));  
parpool(15, 'IdleTimeout', 180);

disp(['Running atsd_optimizer using ', params.classifier]);
atsd_main_experiment(datasets, params);

disp(['Running matlab_optimizer using ', params.classifier]);
matlab_main_experiment(datasets, params);

disp('Run completed successfully.');
