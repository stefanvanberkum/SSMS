"""
This module provides utility methods.
"""
import numpy as np
from matplotlib import pyplot as plt
from statsmodels.tsa.statespace.mlemodel import MLEResults

from state_space import SSMS


def print_params(results: MLEResults, save_path: str):
    """
    Pretty-prints the parameters for an SSMS model.

    :param results: results object for an SSMS model
    :param save_path: path to save location
    :return:
    """
    model = results.model
    if not isinstance(model, SSMS):
        print("Can't print parameters for a non-SSMS model.")
        return

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
