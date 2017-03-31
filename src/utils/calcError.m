function [err] = calcError(actual, prediction);
	err = sum(actual ~= prediction)/length(prediction);
end
