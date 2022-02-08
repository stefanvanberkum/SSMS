"""
This module provides utility methods.
"""
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
        with open(save_path + 'params.csv', 'w') as out:
            out.write()

    print(0)
