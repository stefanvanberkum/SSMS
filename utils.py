"""
This module provides utility methods.
"""
import os

import numpy as np
from matplotlib import pyplot as plt
from statsmodels.tsa.statespace.mlemodel import MLEResults

from state_space import SSMS


def print_results(results: MLEResults, save_path: str, name: str):
    """
    Pretty-prints the results for an SSMS model with one c/state intercept variable. Assumes n > k.

    :param results: results object for an SSMS model
    :param save_path: path to save location
    :param name: model name
    :return:
    """
    model = results.model
    if not isinstance(model, SSMS):
        print("Can't print parameters for a non-SSMS model.")
        return
    if np.size(model.exog, axis=1) != 1:
        print("Can't print a model that does not have exactly one c/state intercept variable.")
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

    y = ','.join(['region', 'var (y)', 'cov (y)'])
    l1 = ','.join(['lambda_1', 'se', 'p-value'])
    mu = ','.join(['var (mu)', 'cov (mu)'])
    nu = ','.join(['var (nu)', 'cov (nu)'])
    l2 = ','.join(['param', 'lambda_2', 'se', 'p-value', 'var'])
    header = ',,'.join([y, l1, mu, nu, l2])

    param_from = param_to
    if model.cov_rest == 'RC':
        param_to += n + 1
        y_var = params[param_from:param_from + n]
        y_cov = params[param_to - 1]
    else:
        param_to += n
        y_var = params[param_from:param_to]

    param_from = param_to
    if model.cov_rest == 'IDE':
        param_to += n
        mu_var = params[param_from:param_to]
    else:
        param_to += n + 1
        mu_var = params[param_from:param_from + n]
        mu_cov = params[param_to - 1]

    param_from = param_to
    if model.cov_rest == 'IDE':
        param_to += n
        nu_var = params[param_from:param_to]
    else:
        param_to += n + 1
        nu_var = params[param_from:param_from + n]
        nu_cov = params[param_to - 1]

    param_from = param_to
    param_to += k
    param_var = params[param_from:]

    with open(os.path.join(save_path, name + '_params.csv'), 'w') as out:
        out.write(header + '\n')

        for i in range(n):
            y = ','.join([regions[i], str(y_var[i])])
            l1 = ','.join([str(lambda_1[i]), str(l1_se[i]), str(l1_p[i])])

            if i == 0:
                if model.cov_rest == 'RC':
                    y = ','.join([y, str(y_cov)])
                else:
                    y = ','.join([y, '0'])
                if model.cov_rest == 'IDE':
                    mu = ','.join([str(mu_var[i]), '0'])
                    nu = ','.join([str(nu_var[i]), '0'])
                else:
                    mu = ','.join([str(mu_var[i]), str(mu_cov)])
                    nu = ','.join([str(nu_var[i]), str(nu_cov)])
            else:
                y = ','.join([y, ''])
                mu = ','.join([str(mu_var[i]), ''])
                nu = ','.join([str(nu_var[i]), ''])
            line = ',,'.join([y, l1, mu, nu])

            if i < k:
                l2 = ','.join([param_names[i], str(lambda_2[i]), str(l2_se[i]), str(l2_p[i]), str(param_var[i])])
                line = ',,'.join([line, l2])
            out.write(line + '\n')


def plot_variables(data: list, info: list, all_regions: False):
    """
    Plots variables.
    :param data: list of form [y, mu, threshold, obs_sd]
    :param info: list of from [index, name]
    :param all_regions: boolean to plot regions 1-by-1 (True) or all at the same time (False)
    :return:
    """
    if all_regions:
        if info:
            t = np.arange(1, len(data[0][0]) + 1)
            for i in range(len(info)):
                index = info[i][0]
                plt.figure()
                plt.suptitle(info[i][1])
                plt.plot(t, data[index][0], 'b')
                plt.plot(t, data[index][1] + data[index][2] * data[index][3], 'r')
                plt.plot(t, data[index][1] - data[index][2] * data[index][3], 'r')
                plt.show()
        else:
            print('No outliers')
    else:
        if info:
            t = np.arange(1, len(data[0][0]) + 1)
            for i in range(len(info)):
                index = info[i][0]
                plt.figure()
                plt.suptitle(info[i][1])
                plt.plot(t, data[index][0], 'b')
                plt.plot(t, data[index][1] + data[index][2] * data[index][3], 'r')
                plt.plot(t, data[index][1] - data[index][2] * data[index][3], 'r')
        else:
            print('No outliers')
        plt.show()

