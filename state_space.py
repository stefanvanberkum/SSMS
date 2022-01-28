"""
This module provides the state-space model functionality.
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm


# TODO: allow for different amounts of parameters:
# An unrestricted model, extends the basic Statsmodels state-space class.
class SSMS(sm.tsa.statespace.MLEModel):
    def __init__(self, log_s: np.array, x: np.array, n: int, t: int, k: int):
        """
        Construct an unrestricted state space model for sales.

        :param log_s:
        :param x:
        :param n:
        :param t:
        :param k:
        """

        # Intialize the state-space model.
        super(SSMS, self).__init__(endog='', k_states=2)


def construct_arrays(data: pd.DataFrame, group_name: str, y_name: str, x_names: list):
    # Filter data (drop all unnecessary columns).
    names = [group_name, y_name] + x_names
    data_select = data[names]

    # Group filtered data by user-specified group name (e.g., region) and collect in a list.
    grouped = data_select.groupby(group_name, sort=False)
    group_list = [group for name, group in grouped]

    # Collect grouped y and x values in a list.
    y_group = [group[y_name].to_numpy() for group in group_list]
    x_group = [group[x_names].to_numpy() for group in group_list]

    # Construct TxN array of y values (T periods, N observed time series).
    endog = np.c_[tuple(y_group)]

    # Construct Tx(N*K) array of x values (T periods, N*K regressors) -> [x_11, x_21, ..., x_K1, x_12, ..., x_KN].
    exog = np.c_[tuple(x_group)]

    return endog
