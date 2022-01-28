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


def construct_arrays(data: pd.DataFrame, group_name: str, y_name: str, z_names: list, c_names: list):
    """
    Constructs arrays of endogenous (y) and exogenous (x) variables.

    :param data: a dataframe
    :param group_name: the column name of the grouping variable for each time series (e.g., region)
    :param y_name: the column name of the dependent variable
    :param z_names: a list of column names of the independent variables to be placed in the Z (design) matrix
    :param c_names: a list of column names of the independent variables to be placed in the c (state intercept) matrix
    :return: a tuple (endog, exog, c), with endog TxN array of y values (T periods, N observed time series), exog Tx(
    N*K) array of x values (T periods, N*K regressors) of the form [x_11, x_12, ..., x_1N, x_21, ..., x_KN],
    and c TxC array of x values that are constant across observations (but vary over time).
    """

    # Filter data (drop all unnecessary columns).
    names = [group_name, y_name] + z_names + c_names
    data_select = data[names]

    # Group filtered data by user-specified group name (e.g., region) and collect in a list.
    grouped = data_select.groupby(group_name, sort=False)
    group_list = [group for name, group in grouped]

    # Collect grouped y and x values in a list.
    y_group = [group[y_name].to_numpy() for group in group_list]
    z_groups = [[group[z_name].to_numpy() for group in group_list] for z_name in z_names]

    # Construct TxN array of y values (T periods, N observed time series).
    endog = np.c_[tuple(y_group)]

    # Construct Tx(N*K) array of x values (T periods, N*K regressors) -> [x_11, x_12, ..., x_1N, x_21, ..., x_KN].
    exog = np.c_[tuple([np.c_[tuple(z_group)] for z_group in z_groups])]

    # Construct TxC array of x values that are constant across observations (but vary over time).
    c_groups = [[group[c_name].to_numpy() for group in group_list] for c_name in c_names]
    c = np.c_[tuple([group for group in group_list[0][c_names]])]
    print()

    return endog, exog, c
