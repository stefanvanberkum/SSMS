"""
This module provides the state-space model functionality.
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm


# An unrestricted model, extends the basic Statsmodels state-space class.
class SSMS(sm.tsa.statespace.MLEModel):
    def __init__(self, data: pd.DataFrame, group_name: str, y_name: str, x_names: list):
        """
        Construct an unrestricted state space model for sales.

        :param data: a dataframe
        :param group_name: the column name of the grouping variable for each time series (e.g., region)
        :param y_name: the column name of the dependent variable
        :param x_names: a list of column names of the independent variables
        """

        # Construct arrays of endogenous (y) and exogenous (x) variables.
        endog, exog = construct_arrays(data, group_name, y_name, x_names)

        t = np.size(endog, 0)
        n = np.size(endog, 1)
        k = np.size(exog, 1)

        # Intialize the state-space model.
        super(SSMS, self).__init__(endog=endog, exog=exog, k_states=n * (k + 1), initialization='diffuse')

        self.ssm["design"] = np.zeros((self.k_endog, self.k_states, self.nobs))


def construct_arrays(data: pd.DataFrame, group_name: str, y_name: str, x_names: list):
    """
    Constructs arrays of endogenous (y) and exogenous (x) variables.

    :param data: a dataframe
    :param group_name: the column name of the grouping variable for each time series (e.g., region)
    :param y_name: the column name of the dependent variable
    :param x_names: a list of column names of the independent variables
    :return: a tuple (endog, exog), with endog TxN array of y values (T periods, N observed time series) and exog Tx(
    N*K) array of x values (T periods, N*K regressors) of the form [x_11, x_12, ..., x_1N, x_21, ..., x_KN]
    """

    # Filter data (drop all unnecessary columns).
    names = [group_name, y_name] + x_names
    data_select = data[names]

    # Group filtered data by user-specified group name (e.g., region) and collect in a list.
    grouped = data_select.groupby(group_name, sort=False)
    group_list = [group for name, group in grouped]

    # Collect grouped y and x values in a list.
    y_group = [group[y_name].to_numpy() for group in group_list]
    x_groups = [[group[x_name].to_numpy() for group in group_list] for x_name in x_names]

    # Construct TxN array of y values (T periods, N observed time series).
    endog = np.c_[tuple(y_group)]

    # Construct Tx(N*K) array of x values (T periods, N*K regressors) -> [x_11, x_12, ..., x_1N, x_21, ..., x_KN].
    exog = np.c_[tuple([np.c_[tuple(x_group)] for x_group in x_groups])]

    return endog, exog
