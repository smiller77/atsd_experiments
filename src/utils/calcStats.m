function stats = calcStats(y, yhat)
% INPUT
% y = true class labels
% yhat = predicted class labels
%
% OUTPUT
% stats is a structure array
% stats.confusionMat
%               Predicted Classes
%                    p'    n'
%              ___|_____|_____| 
%       Actual  p |     |     |
%      Classes  n |     |     |
%
% stats.accuracy = (TP + TN)/(TP + FP + FN + TN) ; the average accuracy is returned
% stats.precision = TP / (TP + FP)                  % for each class label
% stats.sensitivity = TP / (TP + FN)                % for each class label
% stats.specificity = TN / (FP + TN)                % for each class label
% stats.recall = sensitivity                        % for each class label
% stats.fscore = 2*TP /(2*TP + FP + FN)       Partition      % for each class label
%
% TP: true positive, TN: true negative, 
% FP: false positive, FN: false negative
% 

[confusionMat, gorder] = confusionmat(y, yhat);

numClasses = size(confusionMat,1);
numSamples = sum(sum(confusionMat));

[TP, TN, FP, FN] = deal(zeros(numClasses,1));
for class = 1:numClasses
   TP(class) = confusionMat(class,class);
   tempMat = confusionMat([1:class-1,class+1:end], [1:class-1,class+1:end]);
   TN(class) = sum(sum(tempMat));
   FP(class) = sum(confusionMat(:,class))-TP(class);
   FN(class) = sum(confusionMat(class,:))-TP(class);
end

[accuracy, sensitivity, specificity, precision, fscore] = deal(zeros(numClasses,1));
for class = 1:numClasses
    accuracy(class)     = (TP(class) + TN(class)) / numSamples;
    sensitivity(class)  = TP(class) / (TP(class) + FN(class));
    specificity(class)  = TN(class) / (FP(class) + TN(class));
    precision(class)    = TP(class) / (TP(class) + FP(class));
    fscore(class)       = 2 * TP(class)/(2 * TP(class) + FP(class) + FN(class));
end

stats = struct('confusionMat',  confusionMat,...
                'accuracy',     accuracy,...
                'sensitivity',  sensitivity,...
                'specificity',  specificity,...
                'precision',    precision,...
                'recall',       sensitivity,...
                'fscore',       fscore);
            
if exist('gorder','var')
    stats.groupOrder = gorder;
end
    