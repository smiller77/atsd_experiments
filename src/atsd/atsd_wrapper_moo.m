function f = atsd_wrapper_moo(x, data, ftype, params)
    % split
    [datatr, datate] = splitData(10, data, params.split);

    % f_plus
    model = blackbox(datatr, ...
        'classifier', params.classifier,...
        'freeparams', x);
    yhat = predict(model, datate(:, 1:end-1));

    stats = calcStats(datate(:, end), yhat);

    fp_sen = mean(stats.sensitivity);
    fp_spe = mean(stats.specificity);
    fp_fsc = mean(stats.fscore);
    fp_err = 1-stats.accuracy(1);
    if isnan(fp_sen)
      fp_sen = 0;
    end
    if isnan(fp_spe)
      fp_spe = 0;
    end
    if isnan(fp_fsc)
      fp_fsc = 0;
    end

    % f_minus
    yhat_bad = sign(randn(size(datatr, 1), 1));
    yhat_bad(yhat_bad==0) = 1;

    data_bad = [datatr(:, 1:end-1) yhat_bad];
    model = blackbox(data_bad, ...
        'classifier', params.classifier,...
        'freeparams', x);
    yhat = predict(model, datatr(:, 1:end-1));

    stats = calcStats(yhat_bad, yhat);

    fm_sen = mean(stats.sensitivity);
    fm_spe = mean(stats.specificity);
    fm_fsc = mean(stats.fscore);
    fm_err = 1-stats.accuracy(1);
    if isnan(fm_sen)
      fm_sen = 0;
    end
    if isnan(fm_spe)
      fm_spe = 0;
    end
    if isnan(fm_fsc)
      fm_fsc = 0;
    end

    switch ftype
        case 1
            f_plus = fp_err;
            f_minus = [];
        case 2
            f_plus = fp_err;
            f_minus = abs(.5-fm_err);
        case 3
            f_plus = [1-fp_sen; 1-fp_spe; fp_err];
            f_minus = abs(.5-fm_err);
        case 4
            f_plus = [1-fp_sen; 1-fp_spe; fp_err];
            f_minus = [abs(.5-fm_err); abs(.5-fm_sen); abs(.5-fm_spe)];
        case 5
            f_plus = [1-fp_sen; 1-fp_spe; fp_err];
            f_minus = [abs(.5-fm_err); abs(.5-fm_sen); abs(.5-fm_spe)];
    end

    f = [f_plus; f_minus];
end
