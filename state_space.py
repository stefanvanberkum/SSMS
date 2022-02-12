"""
This module provides the state-space model functionality.
"""
import math

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant


class SSMS(sm.tsa.statespace.MLEModel):
    def __init__(self, data: pd.DataFrame, group_name: str, y_name: str, z_names: list, c_names: list, cov_rest: str,
                 cov_group='', var_start=0.1, cov_start=0, fancy_start=True):
        """
        Constructs a state space model for sales, with variable of interest in the c/state intercept matrix for the
        mu (level) and beta (coefficients of independent variables) equation.

        :param data: a dataframe
        :param group_name: the column name of the grouping variable for each time series (e.g., region)
        :param y_name: the column name of the dependent variable
        :param z_names: a list of column names of the independent variables that have a direct effect on sales (to be
            placed in the Z/design matrix)
        :param c_names: a list of column names of the independent variables that have an indirect effect through the
            state equations (to be placed in the c/state intercept matrix)
        :param cov_rest: covariance restriction, one of {'GC': grouped covariances (allow for covariances within
            groups for the sales equation, states are still assumed to be independent), 'IDE': independently distributed
            errors (for both observations and states)}
        :param cov_group: grouping variable for covariances (only applicable if cov_rest='GC' or cov_rest='IDO'),
            default is no grouping variable ('')
        :param var_start: starting value for variances (only applicable if fancy_start=False), default is 0.1
        :param cov_start: starting value for covariances, default is zero
        :param fancy_start: use fancy starting values for the variances (computed with OLS), default is True
        """

        # Check whether required parameters are set.
        if cov_rest == 'GC' and cov_group == '':
            print("Please specify a grouping variable for the covariances or change the covariance restriction.")
            exit(-1)

        # Construct arrays of group names, and endogenous (y) and exogenous (x) variables.
        group_names, y, x_z, x_c, cov_groups, cov_counts = construct_arrays(data, group_name, y_name, z_names, c_names,
                                                                            cov_group)

        n = np.size(y, 1)
        nk = np.size(x_z, 1)
        self.k = round(nk / n)
        k = self.k
        self.k_c = len(c_names)

        n_cov = 0
        if cov_group != '':
            for group in cov_counts:
                n_cov += math.comb(group, 2)
        self.n_cov = n_cov

        self.group_names = group_names
        self.y_name = y_name
        self.z_names = z_names
        self.c_names = c_names
        self.cov_rest = cov_rest
        self.var_start = var_start
        self.cov_start = cov_start
        self.fancy_start = fancy_start
        self.cov_groups = cov_groups
        self.cov_counts = cov_counts

        # Intialize the state-space model.
        super(SSMS, self).__init__(endog=y, exog=x_c, k_states=2 * n + k, initialization='approximate_diffuse')

        # First part of Z matrix is NxN identity matrix for mu.
        z_mu = np.eye(n)

        # Second part is NxN matrix of zeros for nu.
        z_nu = np.zeros((n, n))

        # Split x_z matrix into k distinct parts for each time period.
        x_split = np.apply_along_axis(np.split, 1, x_z, indices_or_sections=k)

        # Save for starting value computation.
        self.x = x_split

        # Concatenate matrices to form [I_N, O_N, x[t, 1], x[t, 2], ..., x[t, k]].
        x_concat = np.array([np.hstack((z_mu, z_nu, np.transpose(np.vstack(arr)))) for arr in x_split])

        # Set Z_t.
        self.ssm["design"] = np.zeros((self.k_endog, self.k_states, self.nobs))
        for t in range(self.nobs):
            self.ssm["design", :, :, t] = x_concat[t]

        # Set R_t = I_(k_states).
        self.ssm["selection"] = np.eye(self.k_states)

        # Placeholder for T_t.
        self.ssm["transition"] = np.eye(self.k_states)

        # Placeholder for c_t.
        self.ssm["state_intercept"] = np.zeros((self.k_states, self.nobs))

        # Placeholder for H_t.
        self["obs_cov"] = np.zeros((self.k_endog, self.k_endog))

        # Placeholder for Q_t.
        self["state_cov"] = np.zeros((self.k_states, self.k_states))

    @property
    def start_params(self):
        """
        Set starting values using user-specified starting values from when the SSMS object was created.

        :return: starting values
        """

        n = self.k_endog
        k = self.k
        k_c = self.k_c

        # Number of lambda: (n + k) * c_k.
        lambda_start = np.zeros((n + k) * k_c)

        if self.fancy_start:
            # Run OLS regressions to get starting values for the variances.
            y_vars = np.zeros(n)
            mu_vars = np.zeros(n)
            x_vars = np.zeros((n, k))
            for i in range(n):
                y = self.endog[:, i]
                ols = OLS(y, add_constant(self.x[:, :, i]))
                ols_result = ols.fit()
                y_vars[i] = np.var(ols_result.resid)
                mu_vars[i] = np.square(ols_result.HC0_se[0])
                x_vars[i, :] = np.square(ols_result.HC0_se[1:])
            var_start = np.concatenate((y_vars, mu_vars, np.mean(x_vars, axis=0)))

        if self.cov_rest == 'GC':
            # For y, n variances, and within-group covariances.
            # For mu and nu, n variances, no covariance.
            # Only variances for the other k states (not region-specific).
            # Number of covariances: (n + n_cov) + 2 * n + k, where n_cov = sum(m_i choose 2) and m_i is the number of
            # elements in group i.
            n_cov = self.n_cov
            cov_start = np.ones((n + n_cov) + 2 * n + k) * self.cov_start

            # Change variances for y, mu, and nu.
            var_to = 0
            if self.fancy_start:
                # Change variances for y.
                var_from = var_to
                var_to += n + n_cov
                cov_start[var_from:var_from + n] = var_start[:n]

                # Change variances for mu.
                var_from = var_to
                var_to += n
                cov_start[var_from:var_to] = var_start[n:2 * n]

                # Change variances for nu.
                var_from = var_to
                var_to += n
                cov_start[var_from:var_to] = var_start[n:2 * n] / 10

                # Change variances for all states corresponding to independent variables.
                var_from = var_to
                var_to += k
                cov_start[var_from:var_to] = var_start[2 * n:]
            else:
                for i in range(3):
                    var_from = var_to
                    var_to += n + n_cov
                    cov_start[var_from:var_from + n] = self.var_start

                # Change variances for all states corresponding to independent variables.
                var_from = var_to
                var_to += k
                cov_start[var_from:var_to] = self.var_start
        else:
            # Only variances, no covariance.
            # Number of covariances: 3 * n + k.
            if self.fancy_start:
                cov_start = np.zeros(3 * n + k)

                # Change variances for y.
                cov_start[:n] = var_start[:n]

                # Change variances for mu.
                cov_start[n:2 * n] = var_start[n:2 * n]

                # Change variances for nu.
                cov_start[2 * n:3 * n] = var_start[n:2 * n] / 10

                # Change variances for all states corresponding to independent variables.
                cov_start[3 * n:] = var_start[2 * n:]
            else:
                cov_start = np.ones(3 * n + k) * self.var_start
        return np.concatenate((lambda_start, cov_start))

    def transform_params(self, unconstrained):
        """
        Restrict covariances to be non-negative.

        :param unconstrained: unconstrained parameters
        :return: constrained parameters
        """

        n = self.k_endog
        k = self.k
        k_c = self.k_c
        constrained = unconstrained.copy()

        # Force covariances to be positive.
        n_params = (n + k) * k_c
        constrained[n_params:] = unconstrained[n_params:] ** 2
        return constrained

    def untransform_params(self, constrained):
        """
        Reverses transform_params() transformation.

        :param constrained: constrained parameters
        :return: unconstrained parameters
        """

        n = self.k_endog
        k = self.k
        k_c = self.k_c
        unconstrained = constrained.copy()

        # Force covariances to be positive.
        n_params = (n + k) * k_c
        unconstrained[n_params:] = constrained[n_params:] ** 0.5
        return unconstrained

    def update(self, params, **kwargs):
        """
        Defines how parameters enter the model.

        :param params: array of parameters
        :param kwargs: additional arguments
        :return:
        """
        params = super(SSMS, self).update(params, **kwargs)

        n = self.k_endog
        k = self.k
        k_c = self.k_c

        # Set T_t = T.
        mat = np.diag(np.ones(self.k_states))
        mat[:n, n:2 * n] = np.eye(n)
        self["transition"] = mat

        # Set c_t.
        # Set intercept for mu.
        index_to = 0
        for obs in range(n):
            index_from = index_to
            index_to += k_c
            col = 0

            for x in range(k_c):
                col += params[index_from + x] * self.exog[:, x]
            self["state_intercept", obs, :] = col

        # Skip over the nu.
        start = 2 * n

        # Set intercept for beta (lambda).
        for state in range(start, self.k_states):
            index_from = index_to
            index_to += k_c
            col = 0

            for x in range(k_c):
                col += params[index_from + x] * self.exog[:, x]
            self["state_intercept", state, :] = col

        # Set H_t = H.
        if self.cov_rest == 'GC':
            # Allow for covariances within groups, but restrict them to be zero across groups.
            index_from = index_to
            index_to += n
            variances = np.diag(params[index_from:index_to])

            # Assign covariances to correct groups.
            covariances = np.zeros((n, n))
            for i in range(n):
                for j in range(i, n):
                    if i != j:
                        group_i = self.cov_groups[i]
                        group_j = self.cov_groups[j]

                        if group_i == group_j:
                            # Allow for covariance.
                            index_from = index_to
                            index_to += 1
                            covariances[i, j] = params[index_from]
                            covariances[j, i] = params[index_from]
            self["obs_cov"] = variances + covariances
        else:
            # Do not allow for off-diagonal elements.
            index_from = index_to
            index_to += n
            self["obs_cov"] = np.diag(params[index_from:index_to])

        # Set Q_t = Q.
        # Do not allow for mu and nu covariances.
        index_from = index_to
        index_to += self.k_states
        self["state_cov"] = np.diag(params[index_from:])


class SSMS_alt(sm.tsa.statespace.MLEModel):
    def __init__(self, data: pd.DataFrame, group_name: str, y_name: str, z_names: list, d_names: list, c_names: list,
                 cov_rest: str, cov_group='', var_start=0.1, cov_start=0, fancy_start=True):
        """
        Constructs a state space model for sales, with variable of interest in the sales and beta (coefficients of
        independent variables) equations.

        :param data: a dataframe
        :param group_name: the column name of the grouping variable for each time series (e.g., region)
        :param y_name: the column name of the dependent variable
        :param z_names: a list of column names of the independent variables that have a direct effect on sales (to be
            placed in the Z/design matrix)
        :param d_names: a list of column names of the independent variables that have a direct effect on sales
            through the d/obs intercept
        :param c_names: a list of column names of the independent variables that have an indirect effect through the
            state equations (to be placed in the c/state intercept matrix)
        :param cov_rest: covariance restriction, one of {'GC': grouped covariances (allow for covariances within
            groups for the sales equation, states are still assumed to be independent), 'IDE': independently distributed
            errors (for both observations and states)}
        :param cov_group: grouping variable for covariances (only applicable if cov_rest='GC' or cov_rest='IDO'),
            default is no grouping variable ('')
        :param var_start: starting value for variances (only applicable if fancy_start=False), default is 0.1
        :param cov_start: starting value for covariances, default is zero
        :param fancy_start: use fancy starting values for the variances (computed with OLS), default is True
        """

        # Check whether required parameters are set.
        if cov_rest == 'GC' and cov_group == '':
            print("Please specify a grouping variable for the covariances or change the covariance restriction.")
            exit(-1)

        # Construct arrays of group names, and endogenous (y) and exogenous (x) variables.
        group_names, y, x_z, x_dc, cov_groups, cov_counts = construct_arrays(data, group_name, y_name, z_names,
                                                                             d_names + c_names, cov_group)

        n = np.size(y, 1)
        nk = np.size(x_z, 1)
        self.k = round(nk / n)
        k = self.k
        self.k_d = len(d_names)
        self.k_c = len(c_names)

        n_cov = 0
        if cov_group != '':
            for group in cov_counts:
                n_cov += math.comb(group, 2)
        self.n_cov = n_cov

        self.group_names = group_names
        self.y_name = y_name
        self.z_names = z_names
        self.c_names = c_names
        self.cov_rest = cov_rest
        self.var_start = var_start
        self.cov_start = cov_start
        self.fancy_start = fancy_start
        self.cov_groups = cov_groups
        self.cov_counts = cov_counts

        # Intialize the state-space model.
        super(SSMS_alt, self).__init__(endog=y, exog=x_dc, k_states=2 * n + k, initialization='approximate_diffuse')

        # First part of Z matrix is NxN identity matrix for mu.
        z_mu = np.eye(n)

        # Second part is NxN matrix of zeros for nu.
        z_nu = np.zeros((n, n))

        # Split x_z matrix into k distinct parts for each time period.
        x_split = np.apply_along_axis(np.split, 1, x_z, indices_or_sections=k)

        # Save for starting value computation.
        self.x = x_split

        # Concatenate matrices to form [I_N, O_N, x[t, 1], x[t, 2], ..., x[t, k]].
        x_concat = np.array([np.hstack((z_mu, z_nu, np.transpose(np.vstack(arr)))) for arr in x_split])

        # Set Z_t.
        self.ssm["design"] = np.zeros((self.k_endog, self.k_states, self.nobs))
        for t in range(self.nobs):
            self.ssm["design", :, :, t] = x_concat[t]

        # Placeholder for d_t.
        self.ssm["obs_intercept"] = np.zeros((self.k_endog, self.nobs))

        # Set R_t = I_(k_states).
        self.ssm["selection"] = np.eye(self.k_states)

        # Placeholder for T_t.
        self.ssm["transition"] = np.eye(self.k_states)

        # Placeholder for c_t.
        self.ssm["state_intercept"] = np.zeros((self.k_states, self.nobs))

        # Placeholder for H_t.
        self["obs_cov"] = np.zeros((self.k_endog, self.k_endog))

        # Placeholder for Q_t.
        self["state_cov"] = np.zeros((self.k_states, self.k_states))

    @property
    def start_params(self):
        """
        Set starting values using user-specified starting values from when the SSMS object was created.

        :return: starting values
        """

        n = self.k_endog
        k = self.k
        k_d = self.k_d
        k_c = self.k_c

        # Number of lambda: n * k_d + k * k_c
        lambda_start = np.zeros(n * k_d + k * k_c)

        if self.fancy_start:
            # Run OLS regressions to get starting values for the variances.
            y_vars = np.zeros(n)
            mu_vars = np.zeros(n)
            x_vars = np.zeros((n, k))
            for i in range(n):
                y = self.endog[:, i]
                ols = OLS(y, add_constant(self.x[:, :, i]))
                ols_result = ols.fit()
                y_vars[i] = np.var(ols_result.resid)
                mu_vars[i] = np.square(ols_result.HC0_se[0])
                x_vars[i, :] = np.square(ols_result.HC0_se[1:])
            var_start = np.concatenate((y_vars, mu_vars, np.mean(x_vars, axis=0)))

        if self.cov_rest == 'GC':
            # For y, n variances, and within-group covariances.
            # For mu and nu, n variances, no covariance.
            # Only variances for the other k states (not region-specific).
            # Number of covariances: (n + n_cov) + 2 * n + k, where n_cov = sum(m_i choose 2) and m_i is the number of
            # elements in group i.
            n_cov = self.n_cov
            cov_start = np.ones((n + n_cov) + 2 * n + k) * self.cov_start

            # Change variances for y, mu, and nu.
            var_to = 0
            if self.fancy_start:
                # Change variances for y.
                var_from = var_to
                var_to += n + n_cov
                cov_start[var_from:var_from + n] = var_start[:n]

                # Change variances for mu.
                var_from = var_to
                var_to += n
                cov_start[var_from:var_to] = var_start[n:2 * n]

                # Change variances for nu.
                var_from = var_to
                var_to += n
                cov_start[var_from:var_to] = var_start[n:2 * n] / 10

                # Change variances for all states corresponding to independent variables.
                var_from = var_to
                var_to += k
                cov_start[var_from:var_to] = var_start[2 * n:]
            else:
                for i in range(3):
                    var_from = var_to
                    var_to += n + n_cov
                    cov_start[var_from:var_from + n] = self.var_start

                # Change variances for all states corresponding to independent variables.
                var_from = var_to
                var_to += k
                cov_start[var_from:var_to] = self.var_start
        else:
            # Only variances, no covariance.
            # Number of covariances: 3 * n + k.
            if self.fancy_start:
                cov_start = np.zeros(3 * n + k)

                # Change variances for y.
                cov_start[:n] = var_start[:n]

                # Change variances for mu.
                cov_start[n:2 * n] = var_start[n:2 * n]

                # Change variances for nu.
                cov_start[2 * n:3 * n] = var_start[n:2 * n] / 10

                # Change variances for all states corresponding to independent variables.
                cov_start[3 * n:] = var_start[2 * n:]
            else:
                cov_start = np.ones(3 * n + k) * self.var_start
        return np.concatenate((lambda_start, cov_start))

    def transform_params(self, unconstrained):
        """
        Restrict covariances to be non-negative.

        :param unconstrained: unconstrained parameters
        :return: constrained parameters
        """

        n = self.k_endog
        k = self.k
        k_d = self.k_d
        k_c = self.k_c
        constrained = unconstrained.copy()

        # Force covariances to be positive.
        n_params = n * k_d + k * k_c
        constrained[n_params:] = unconstrained[n_params:] ** 2
        return constrained

    def untransform_params(self, constrained):
        """
        Reverses transform_params() transformation.

        :param constrained: constrained parameters
        :return: unconstrained parameters
        """

        n = self.k_endog
        k = self.k
        k_d = self.k_d
        k_c = self.k_c
        unconstrained = constrained.copy()

        # Force covariances to be positive.
        n_params = n * k_d + k * k_c
        unconstrained[n_params:] = constrained[n_params:] ** 0.5
        return unconstrained

    def update(self, params, **kwargs):
        """
        Defines how parameters enter the model.

        :param params: array of parameters
        :param kwargs: additional arguments
        :return:
        """
        params = super(SSMS_alt, self).update(params, **kwargs)

        n = self.k_endog
        k = self.k
        k_d = self.k_d
        k_c = self.k_c

        # Set d_t.
        index_to = 0
        for obs in range(n):
            index_from = index_to
            index_to += k_d
            col = 0

            for x in range(k_d):
                col += params[index_from + x] * self.exog[:, x]
            self["obs_intercept", obs, :] = col

        # Set T_t = T.
        mat = np.diag(np.ones(self.k_states))
        mat[:n, n:2 * n] = np.eye(n)
        self["transition"] = mat

        # Set c_t.
        # Set intercept for beta (lambda).
        for state in range(2 * n, self.k_states):
            index_from = index_to
            index_to += k_c
            col = 0

            for x in range(k_c):
                col += params[index_from + x] * self.exog[:, k_d + x]
            self["state_intercept", state, :] = col

        # Set H_t = H.
        if self.cov_rest == 'GC':
            # Allow for covariances within groups, but restrict them to be zero across groups.
            index_from = index_to
            index_to += n
            variances = np.diag(params[index_from:index_to])

            # Assign covariances to correct groups.
            covariances = np.zeros((n, n))
            for i in range(n):
                for j in range(i, n):
                    if i != j:
                        group_i = self.cov_groups[i]
                        group_j = self.cov_groups[j]

                        if group_i == group_j:
                            # Allow for covariance.
                            index_from = index_to
                            index_to += 1
                            covariances[i, j] = params[index_from]
                            covariances[j, i] = params[index_from]
            self["obs_cov"] = variances + covariances
        else:
            # Do not allow for off-diagonal elements.
            index_from = index_to
            index_to += n
            self["obs_cov"] = np.diag(params[index_from:index_to])

        # Set Q_t = Q.
        # Do not allow for mu and nu covariances.
        index_from = index_to
        index_to += self.k_states
        self["state_cov"] = np.diag(params[index_from:])


def construct_arrays(data: pd.DataFrame, group_name: str, y_name: str, z_names: list, c_names: list, cov_group: str):
    """
    Constructs arrays of endogenous (y) and exogenous (x) variables.

    :param data: a dataframe
    :param group_name: the column name of the grouping variable for each time series (e.g., region)
    :param y_name: the column name of the dependent variable
    :param z_names: a list of column names of the independent variables to be placed in the Z (design) matrix
    :param c_names: a list of column names of the independent variables to be placed in the c (state intercept) matrix
    :param cov_group: grouping variable for covariances
    :return: a tuple (group_names, y, x_z, x_c, cov_group), with group_names Nx1 array of group names, y TxN array of
        y values ( T periods, N observed time series), x_z Tx(N*K) array of x values (T periods, N*K regressors) of the
        form [x_11, x_12, ..., x_1N, x_21, ..., x_KN], x_c TxC array of x values that are constant across observations (
        but vary over time), cov_groups Nx1 categorical array mapping time series to groups according to the covariance
        grouping variable, and cov_counts array of number of group members (index corresponds to group number)
    """

    # Filter data (drop all unnecessary columns).
    if cov_group != '':
        names = [group_name, y_name, cov_group] + z_names + c_names
    else:
        names = [group_name, y_name] + z_names + c_names
    data_select = data[names]

    # Group filtered data by user-specified group name (e.g., region) and collect in a list.
    grouped = data_select.groupby(group_name, sort=False)
    group_names = [name for name, group in grouped]
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

    # Construct covariance group mapping.
    cov_groups = np.zeros(len(group_list), dtype=int)
    cov_counts = None
    if cov_group != '':
        cov_grouping = np.array([group.iloc[0][cov_group] for group in group_list])
        cov_names = np.unique(cov_grouping)
        cov_counts = np.zeros(len(cov_names), dtype=int)
        group_dict = {}

        # Encode groups by group number in [0, n - 1].
        for i in range(len(cov_names)):
            cov_counts[i] = np.sum(cov_grouping == cov_names[i])
            group_dict[cov_names[i]] = i

        # Create array of group numbers.
        for i in range(len(cov_grouping)):
            cov_groups[i] = group_dict[cov_grouping[i]]
    return group_names, y, x_z, x_c, cov_groups, cov_counts
