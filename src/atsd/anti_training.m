function [x, f, exitflag] = anti_training(data, ftype, params)
    popSize = 25;
    lambda = 0.5;

    if params.moo == 1
        optimoptions = gaoptimset('PopulationSize', popSize, 'UseParallel', params.parallel);
        [x, f, exitflag] = gamultiobj(@(x)(atsd_wrapper_moo(x, data, ftype, params)), ...
            params.nvars, [], [], [], [], params.lb, params.ub, optimoptions);

    % not currently implementing blackbox
    elseif params.moo == 2
        optimoptions = saoptimset();
        optimoptions = saoptimset(optimoptions, 'MaxIter', 2000); 
        [x, f, exitflag] = simulannealbnd(@(x)(atsd_wrapper_soo(x, data, lambda)), ...
            [1, .5], lb, ub, optimoptions);

    % not currently implementing blackbox
    elseif params.moo == 3
        optimoptions = gaoptimset('PopulationSize', popSize, 'UseParallel', true);
        [x, f, exitflag] = gamultiobj(@(x)(atsd_wrapper_soo(x, data, lamda)), ...
            nvar, Aineq, bineq, A, b, lb, ub, optimoptions);
    end
end
