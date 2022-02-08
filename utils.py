"""
This module provides utility methods.
"""
from statsmodels.tsa.statespace.mlemodel import MLEResults

from state_space import SSMS


def print_params(results: MLEResults, save_path: str):
    """
    Pretty-prints the parameters for an RC-LLT (RSC) type model.

    :param results: results object for RC-LLT (RSC) type SSMS model
    :param save_path: path to save location
    :return:
    """
    model = results.model
    if not isinstance(model, SSMS):
        print("Can't print parameters for a non-SSMS model.")
        return
    if not model.llt or model.param_rest != 'RC' or model.cov_rest != 'RSC':
        print("Can't print parameters for this model type, just for RC-LLT (RSC).")
        return

    # Retrieve fitted parameters, z-statistics, and p-values.
    params = results.params
    zvalues = results.zvalues
    pvalues = results.pvalues

    print(0)
