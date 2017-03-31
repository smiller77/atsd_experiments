function [x, fval] = matlab_search(data, params)

numData = size(data, 1);

minfn = @(z)kfoldLoss(blackbox(data, ...
						'classifier', params.classifier, ...
						'freeparams', z, ...
						'dokfold', true));

opts = optimset('TolX', 5e-4, 'TolFun', 5e-4);

z = zeros(params.nvars, 1);
for i = 1:params.nvars
	lb = params.lb(i);
	ub = params.ub(i);
	z(i) = (ub-lb)*rand()+lb;
end
[x, fval] = fminsearch(minfn, z, opts);
