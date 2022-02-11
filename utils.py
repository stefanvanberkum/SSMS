"""
This module provides utility methods.
"""
import os

import numpy as np
from matplotlib import pyplot as plt
from statsmodels.tsa.statespace.mlemodel import MLEResults

from state_space import SSMS, SSMS_alt


def print_results(results: MLEResults, save_path: str, name: str):
    """
    Pretty-prints the results for an SSMS model with n + k variables of interest (n in either sales or mu,
    k in beta equations). Assumes n > k.

    :param results: results object for an SSMS model
    :param save_path: path to save location
    :param name: model name
    :return:
    """
    model = results.model
    if not (isinstance(model, SSMS) or isinstance(model, SSMS_alt)):
        print("Can't print parameters for a non-SSMS model.")
        return
    if isinstance(model, SSMS) and np.size(model.exog, axis=1) != 1:
        print("Can't print a model that does not have exactly one c/state intercept variable.")
        return
    if isinstance(model, SSMS_alt) and np.size(model.exog, axis=1) != 2:
        print("Can't print a model that does not have exactly one d/obs intercept and one c/state intercept variable.")
        return

    # Print AIC, BIC, MSE, and MAE.
    with open(os.path.join(save_path, name + '_stats.csv'), 'w') as out:
        header = ','.join(['AIC', 'BIC', 'MSE', 'MAE'])
        stats = ','.join([str(results.aic), str(results.bic), str(results.mse), str(results.mae)])
        out.write('\n'.join([header, stats]))

    # Print fitted parameters, standard errors, and p-values.
    regions = model.group_names
    params = results.params
    ses = results.bse
    pvalues = results.pvalues

    n = len(regions)
    k = model.k
    n_cov = model.n_cov

    param_from = 0
    param_to = n
    lambda_1 = params[param_from:param_to]
    l1_se = ses[param_from:param_to]
    l1_p = pvalues[param_from:param_to]

    param_names = model.z_names
    param_from = param_to
    param_to += k
    lambda_2 = params[param_from:param_to]
    l2_se = ses[param_from:param_to]
    l2_p = pvalues[param_from:param_to]

    y = ','.join(['region', 'var (y)'])
    l1 = ','.join(['lambda_1', 'se', 'p-value'])
    mu = 'var (mu)'
    nu = 'var (nu)'
    l2 = ','.join(['param', 'lambda_2', 'se', 'p-value', 'var'])
    header = ',,'.join([y, l1, mu, nu, l2])

    param_from = param_to
    if model.cov_rest == 'GC':
        param_to += n + n_cov
        y_var = params[param_from:param_from + n]
    else:
        param_to += n
        y_var = params[param_from:param_to]

    param_from = param_to
    param_to += n
    mu_var = params[param_from:param_to]

    param_from = param_to
    param_to += n
    nu_var = params[param_from:param_to]

    param_from = param_to
    param_to += k
    param_var = params[param_from:]

    with open(os.path.join(save_path, name + '_params.csv'), 'w') as out:
        out.write(header + '\n')

        for i in range(n):
            y = ','.join([regions[i], str(y_var[i])])
            mu = str(mu_var[i])
            nu = str(nu_var[i])
            l1 = ','.join([str(lambda_1[i]), str(l1_se[i]), str(l1_p[i])])
            line = ',,'.join([y, l1, mu, nu])

            if i < k:
                l2 = ','.join([param_names[i], str(lambda_2[i]), str(l2_se[i]), str(l2_p[i]), str(param_var[i])])
                line = ',,'.join([line, l2])
            out.write(line + '\n')


def plot_variables(data: list, data_names: list, sd: float):
    if data_names:
        print(f'Regions with outliers: {data_names}')
        t = np.arange(1, len(data[0][0]) + 1)
        for i in range(len(data_names)):
            plt.figure()
            plt.suptitle(data_names[i])
            plt.plot(t, data[i][0], 'b')
            plt.plot(t, data[i][1] + sd * data[i][2], 'r')
            plt.plot(t, data[i][1] - sd * data[i][2], 'r')
    else:
        print('No outliers')
    plt.show()
