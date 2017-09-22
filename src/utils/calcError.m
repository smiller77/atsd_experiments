function [err] = calcError(yhat, y)
	err = sum(y ~= yhat)/length(y);
end