function [x, fval] = matlab_search(data, params)
	lb = params.lb;
	ub = params.ub;

    minfn = @(z)kfoldLoss(blackbox(data, ...
                            'classifier', params.classifier, ...
                            'freeparams', z, ...
                            'dokfold', true));

    opts = optimset('TolX', 5e-4, 'TolFun', 5e-4, 'Display', 'none');

    x = zeros(10, params.nvars);
	fval = zeros(10, 1);
	for j = 1:10
    	for i = 1:params.nvars
        	z(i) = (ub(i)-lb(i))*rand()+lb(i);
		end
    	[x(j,:), fval(j)] = fmincon(minfn, z, [], [], [], [], lb, ub, [], opts);
	end

	[fval, arg] = min(fval);
	x = x(arg, :);
end
