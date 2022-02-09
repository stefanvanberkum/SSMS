"""
This module provides utility methods.
"""
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from statsmodels.tsa.statespace.mlemodel import MLEResults

from state_space import SSMS


def print_params(results: MLEResults, save_path: str):
    """
    Pretty-prints the parameters for an IDE-type SSMS model.

    :param results: results object for an IDE-type SSMS model
    :param save_path: path to save location
    :return:
    """
    model = results.model
    if not isinstance(model, SSMS):
        print("Can't print parameters for a non-SSMS model.")
        return
    if model.cov_rest != 'IDE':
        print("Can't print parameters for a non-IDE SSMS model.")

    # Retrieve fitted parameters, z-statistics, and p-values.
    regions = model.group_names
    params = results.params
    zvalues = results.zvalues
    pvalues = results.pvalues

    if model.cov_rest == 'RC':
        header = ','.join(['region', 'lambda_1', 'se', 'lambda_2'])
        with open(save_path + 'params.csv', 'w') as out:
            out.write()

    print(0)


def plot_sales(data: pd.DataFrame):
    grouped = data.groupby('Region', sort=False)
    group_names = [name for name, group in grouped]
    group_list = [group for name, group in grouped]

    # Collect grouped y and x values in a list.
    y_group = [group['SalesGoodsEUR'].to_numpy() for group in group_list]
    n = len(y_group[0])
    block = np.ceil(0.25 * n / 2).astype(int)

    t = np.arange(1, n + 1)
    for obs in range(len(y_group)):
        y = y_group[obs]

        mu = np.zeros(n)
        mu[0] = np.mean(y[1:block])
        mu[n - 1] = np.mean(y[n - 1 - block:n - 1])
        for i in range(1, n - 1):
            if i < block:
                mu[i] = np.mean(np.concatenate((y[:i], y[i + 1:i + 1 + block])))
            elif i + 1 + block > n - 1:
                mu[i] = np.mean(np.concatenate((y[i - block:i], y[i + 1:])))
            else:
                mu[i] = np.mean(np.concatenate((y[i - block:i], y[i + 1:i + 1 + block])))

        sd = np.std(y)
        plt.suptitle(group_names[obs])
        plt.plot(t, y, 'b')
        plt.plot(t, mu + 4 * sd, 'r')
        plt.plot(t, mu - 4 * sd, 'r')
        plt.show()
        plt.close('all')
