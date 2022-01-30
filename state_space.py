"""
This module provides the state-space model functionality.
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm


# An unrestricted model, extends the basic Statsmodels state-space class.
class SSMS(sm.tsa.statespace.MLEModel):
    def __init__(self, data: pd.DataFrame, group_name: str, y_name: str, z_names: list, c_names: list, cov_type: str):
        """
        Constructs an unrestricted state space model for sales.

        :param data: a dataframe
        :param group_name: the column name of the grouping variable for each time series (e.g., region)
        :param y_name: the column name of the dependent variable
        :param z_names: a list of column names of the independent variables to be placed in the Z (design) matrix
        :param c_names: a list of column names of the independent variables to be placed in the c (state intercept)
            matrix
        :param cov_type: model type, one of {'F': full model, 'RSC': restricted state covariance (same correlation
            across time series), 'IDS': independently distributed states (not implemented), 'GSC': grouped state
            covariances ( not implemented)}
        """

        # Construct arrays of endogenous (y) and exogenous (x) variables.
        y, x_z, x_c = construct_arrays(data, group_name, y_name, z_names, c_names)

        n = np.size(y, 1)
        nk = np.size(x_z, 1)
        k = round(nk / n)

        # Intialize the state-space model.
        super(SSMS, self).__init__(endog=y, exog=x_c, k_states=n * (k + 1), initialization='diffuse')
        self.cov_type = cov_type

        # First part of Z matrix is NxN identity matrix for mu.
        z_mu = np.eye(n)

        # Split x_z matrix into k distinct parts for each time period.
        x_split = np.apply_along_axis(np.split, 1, x_z, indices_or_sections=k)

        # Transform each sequence of x variables into a diagonal matrix.
        x_diag = np.apply_along_axis(np.diag, 2, x_split)

        # Concatenate matrices to form [I_N, diag(x[t, 1]), diag(x[t, 2]), ..., diag(x[t, k])].
        x_concat = np.array([np.concatenate((z_mu, np.concatenate(tuple(arr), axis=1)), axis=1) for arr in x_diag])

        # Set Z_t = [I_N, diag(x[t, 1]), diag(x[t, 2]), ..., diag(x[t, k])].
        self.ssm["design"] = np.zeros((self.k_endog, self.k_states, self.nobs))
        for t in range(self.nobs):
            self.ssm["design", :, :, t] = x_concat[t]

        # Set R_t = I_(N * (K + 1)).
        self.ssm["selection"] = np.eye(self.k_states)

        # Placeholder for T_t.
        self.ssm["transition"] = np.eye(self.k_states)

    @property
    def start_params(self):
        """
        Set starting values (diffuse).

        :return: starting values
        """

        # TODO: different for different types

        n = np.size(self.endog, 1)
        k = round(self.k_states / n) - 1
        c_k = np.size(self.exog, axis=1)

        # Number of parameters: k_states + k_states * (c_k + 1).
        param_start = np.ones(self.k_states * (c_k + 2)) * 0.5

        # Number of covariances: n^2 + n^2 * (k + 1).
        cov_start = np.ones((n ** 2) * (k + 2))
        return np.concatenate((param_start, cov_start))

    def transform_params(self, unconstrained):
        """
        Restrict covariances to be non-negative.

        :param unconstrained: unconstrained parameters
        :return: constrained parameters
        """

        c_k = np.size(self.exog, axis=1)
        constrained = unconstrained.copy()

        constrained[self.k_states * (c_k + 2):] = constrained[self.k_states * (c_k + 2):] ** 2
        return constrained

    def untransform_params(self, constrained):
        """
        Reverses transform_params() transformation.

        :param constrained: constrained parameters
        :return: unconstrained parameters
        """

        c_k = np.size(self.exog, axis=1)
        unconstrained = constrained.copy()

        unconstrained[self.k_states * (c_k + 2):] = constrained[self.k_states * (c_k + 2):] ** 0.5
        return unconstrained

    def update(self, params, **kwargs):
        """
        Defines how parameters enter the model.

        :param params: array of parameters
        :param kwargs: additional arguments
        :return:
        """
        params = super(SSMS, self).update(params, **kwargs)

        n = np.size(self.endog, 1)
        k = round(self.k_states / n) - 1
        k_c = np.size(self.exog, axis=1)

        # Set T_t = T.
        index_from = 0
        index_to = self.k_states
        self["transition"] = np.diag(params[index_from:index_to])

        # Set c_t.
        self["state_intercept"] = np.zeros((self.k_states, self.nobs))
        for state in range(self.k_states):
            index_from = index_to
            index_to += k_c + 1
            col = params[index_from]

            for x in range(k_c):
                col += params[index_from + 1] * self.exog[:, x]
            self["state_intercept", state, :] = col

        # Set H_t = H and allow for off-diagonal elements (correlation between time series).
        self["obs_cov"] = np.zeros((n, n))
        for i in range(n):
            for j in range(i, n):
                index_from = index_to
                index_to += 1
                self["obs_cov", i, j] = params[index_from]

                if i != j:
                    self["obs_cov", j, i] = params[index_from]

        # Set Q_t = Q.
        if self.cov_type == 'F':
            # Allow for off-diagonal elements (correlation between time series).
            self["state_cov"] = np.zeros((self.k_states, self.k_states))
            for state in range(k + 1):
                cov_from = state * n
                cov_to = (state + 1) * n
                for i in range(cov_from, cov_to):
                    for j in range(i, cov_to):
                        index_from = index_to
                        index_to += 1
                        self["state_cov", i, j] = params[index_from]

                        if i != j:
                            self["state_cov", j, i] = params[index_from]
        elif self.cov_type == 'RSC':
            # Allow for off-diagonal elements, but restrict them to be the same across time-series.
            self["state_cov"] = np.zeros((self.k_states, self.k_states))
            for state in range(k + 1):
                index_from = index_to
                index_to += n + 1
                cov_from = state * n
                cov_to = (state + 1) * n
                variances = np.diag(params[index_from:index_to - 1])
                covariances = np.ones((n, n)) * params[index_to - 1] - np.diag(np.ones(n) * params[index_to - 1])
                self["state_cov", cov_from:cov_to, cov_from:cov_to] = variances + covariances


def construct_arrays(data: pd.DataFrame, group_name: str, y_name: str, z_names: list, c_names: list):
    """
    Constructs arrays of endogenous (y) and exogenous (x) variables.

    :param data: a dataframe
    :param group_name: the column name of the grouping variable for each time series (e.g., region)
    :param y_name: the column name of the dependent variable
    :param z_names: a list of column names of the independent variables to be placed in the Z (design) matrix
    :param c_names: a list of column names of the independent variables to be placed in the c (state intercept) matrix
    :return: a tuple (y, x_z, x_c), with y TxN array of y values (T periods, N observed time series), x_z Tx(N*K) array
        of x values (T periods, N*K regressors) of the form [x_11, x_12, ..., x_1N, x_21, ..., x_KN], and x_c TxC
        array of x values that are constant across observations (but vary over time).
    """

    # Filter data (drop all unnecessary columns).
    names = [group_name, y_name] + z_names + c_names
    data_select = data[names]

    # Group filtered data by user-specified group name (e.g., region) and collect in a list.
    grouped = data_select.groupby(group_name, sort=False)
    group_list = [group for name, group in grouped]

    # Collect grouped y and x values in a list.
    y_group = [group[y_name].to_numpy() for group in group_list]
    z_group = [[group[z_name].to_numpy() for group in group_list] for z_name in z_names]
    c_group = [[group[c_name].to_numpy() for group in group_list] for c_name in c_names]

    # Construct TxN array of y values (T periods, N observed time series).
    # We append all time series by column.
    y = np.c_[tuple(y_group)]

    # Construct Tx(N*K) array of x values (T periods, N*K regressors) -> [x_11, x_12, ..., x_1N, x_21, ..., x_KN].
    # First we append each time series by column, then we append the variables by column.
    x_z = np.c_[tuple([np.c_[tuple(x)] for x in z_group])]

    # Construct TxC array of x values that are constant across observations (but vary over time).
    # We append all variables for an arbitrarily chosen group by column.
    x_c = np.transpose(np.array([x[0] for x in c_group]))
    return y, x_z, x_c
