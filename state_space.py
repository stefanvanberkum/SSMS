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
                 var_start: float, cov_start: float, fancy_start: bool):
        """
        Constructs a state space model for sales.

        :param data: a dataframe
        :param group_name: the column name of the grouping variable for each time series (e.g., region)
        :param y_name: the column name of the dependent variable
        :param z_names: a list of column names of the independent variables that have a direct effect on sales (to be
            placed in the Z/design matrix)
        :param c_names: a list of column names of the independent variables that have an indirect effect through the
            state equations (to be placed in the c/state intercept matrix)
        :param cov_rest: covariance restriction, one of {'RC': restricted covariances (same correlation across time
            series), 'IDO': restricted state covariances and independently distributed observations,
            'IDE': independently distributed errors (for both observations and states)}
        :param var_start: starting value for variances
        :param cov_start: starting value for covariances
        :param fancy_start: use fancy starting values for the variances (computed with OLS)
        """

        # Construct arrays of group names, and endogenous (y) and exogenous (x) variables.
        group_names, y, x_z, x_c = construct_arrays(data, group_name, y_name, z_names, c_names)

        n = np.size(y, 1)
        nk = np.size(x_z, 1)
        self.k = round(nk / n)
        k = self.k

        self.group_names = group_names
        self.y_name = y_name
        self.z_names = z_names
        self.c_names = c_names
        self.cov_rest = cov_rest
        self.var_start = var_start
        self.cov_start = cov_start
        self.fancy_start = fancy_start

        # Intialize the state-space model.
        super(SSMS, self).__init__(endog=y, exog=x_c, k_states=2 * n + k, initialization='diffuse')

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
        c_k = np.size(self.exog, axis=1)

        # Number of lambda: (n + k) * c_k.
        lambda_start = np.zeros((n + k) * c_k)

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

        if self.cov_rest == 'RC':
            # For y, mu, and nu, n variances, one covariance.
            # Only variances for the other k states (not region-specific).
            # Number of covariances: 3 * (n + 1) + k
            cov_start = np.ones(3 * (n + 1) + k) * self.cov_start

            # Change variances for y, mu, and nu.
            var_to = 0
            if self.fancy_start:
                # Change variances for y.
                var_from = var_to
                var_to += n + 1
                cov_start[var_from:var_from + n] = var_start[:n]

                # Change variances for mu.
                var_from = var_to
                var_to += n + 1
                cov_start[var_from:var_from + n] = var_start[n:2 * n]

                # Change variances for nu.
                var_from = var_to
                var_to += n + 1
                cov_start[var_from:var_from + n] = var_start[n:2 * n] / 10

                # Change variances for all states corresponding to independent variables.
                var_from = var_to
                var_to += k
                cov_start[var_from:var_to] = var_start[2 * n:]
            else:
                for i in range(3):
                    var_from = var_to
                    var_to += n + 1
                    cov_start[var_from:var_from + n] = self.var_start

                # Change variances for all states corresponding to independent variables.
                var_from = var_to
                var_to += k
                cov_start[var_from:var_to] = self.var_start
        elif self.cov_rest == 'IDO':
            # For y, n variances, no covariance.
            # For mu and nu, n variances, one covariance.
            # Only variances for the other k states (not region-specific).
            # Number of covariances: n + 2 * (n + 1) + k
            cov_start = np.ones(n + 2 * (n + 1) + k) * self.cov_start

            var_from = 0
            var_to = n
            if self.fancy_start:
                # Change variances for y.
                var_from = var_to
                var_to += n
                cov_start[var_from:var_from + n] = var_start[:n]

                # Change variances for mu.
                var_from = var_to
                var_to += n + 1
                cov_start[var_from:var_from + n] = var_start[n:2 * n]

                # Change variances for nu.
                var_from = var_to
                var_to += n + 1
                cov_start[var_from:var_from + n] = var_start[n:2 * n] / 10

                # Change variances for all states corresponding to independent variables.
                var_from = var_to
                var_to += k
                cov_start[var_from:var_to] = var_start[2 * n:]
            else:
                # Change variances for y.
                cov_start[var_from:var_to] = self.var_start

                # Change variances for mu and nu.
                for i in range(2):
                    var_from = var_to
                    var_to += n + 1
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
        c_k = np.size(self.exog, axis=1)
        constrained = unconstrained.copy()

        # Force covariances to be positive.
        n_params = (n + k) * c_k
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
        c_k = np.size(self.exog, axis=1)
        unconstrained = constrained.copy()

        # Force covariances to be positive.
        n_params = (n + k) * c_k
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
        k_c = np.size(self.exog, axis=1)

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

        # Set intercept for beta (nu and lambda).
        for state in range(start, self.k_states):
            index_from = index_to
            index_to += k_c
            col = 0

            for x in range(k_c):
                col += params[index_from + x] * self.exog[:, x]
            self["state_intercept", state, :] = col

        # Set H_t = H.
        if self.cov_rest == 'RC':
            # Allow for off-diagonal elements, but restrict them to be the same across time-series.
            index_from = index_to
            index_to += n + 1
            variances = np.diag(params[index_from:index_to - 1])
            covariances = np.ones((n, n)) * params[index_to - 1] - np.diag(np.ones(n) * params[index_to - 1])
            self["obs_cov"] = variances + covariances
        else:
            # Do not allow for off-diagonal elements.
            index_from = index_to
            index_to += n
            self["obs_cov"] = np.diag(params[index_from:index_to])

        # Set Q_t = Q.
        if self.cov_rest == 'IDE':
            # Do not allow for mu and nu covariances.
            index_from = index_to
            index_to += self.k_states
            self["state_cov"] = np.diag(params[index_from:])
        else:
            # Allow mu and nu covariances, but restrict them to be the same across time-series.
            cov_to = 0
            for block in range(2):
                index_from = index_to
                index_to += n + 1
                cov_from = cov_to
                cov_to += n
                variances = np.diag(params[index_from:index_to - 1])
                covariances = np.ones((n, n)) * params[index_to - 1] - np.diag(np.ones(n) * params[index_to - 1])
                self["state_cov", cov_from:cov_to, cov_from:cov_to] = variances + covariances

            # Set variances for the other k states.
            index_from = index_to
            index_to += k
            cov_from = cov_to
            cov_to += k
            self["state_cov", cov_from:cov_to, cov_from:cov_to] = np.diag(params[index_from:index_to])


class SSMS_old(sm.tsa.statespace.MLEModel):
    def __init__(self, data: pd.DataFrame, group_name: str, y_name: str, z_names: list, c_names: list, llt: bool,
                 alt: bool, param_rest: str, cov_rest: str, tau_start: float, var_start: float, cov_start: float):
        """
        Constructs a state space model for sales.

        :param data: a dataframe
        :param group_name: the column name of the grouping variable for each time series (e.g., region)
        :param y_name: the column name of the dependent variable
        :param z_names: a list of column names of the independent variables that have a direct effect on sales (to be
            placed in the Z/design matrix)
        :param c_names: a list of column names of the independent variables that have an indirect effect through the
            state equations (to be placed in the c/state intercept matrix)
        :param llt: true if local linear trend should be included
        :param alt: true if a tau parameter is to be included
        :param param_rest: parameter restriction, one of {'F': full model, 'RSI': restricted state intercept (same
            effect across regions for independent variables in the state equations), 'RC': restricted coefficients (
            same effect across regions for independent variables in the sales equation), 'NSC': non-stochastic
            coefficients (no stochastic element for coefficient of independent variables in the sales equation,
            not implemented)}
        :param cov_rest: covariance restriction, one of {'F': full model, 'RSC': restricted state covariance (same
            correlation across time series), 'IDS': independently distributed states}
        :param tau_start: starting value for state autocorrelation parameters
        :param var_start: starting value for variances
        :param cov_start: starting value for covariances
        """

        # Construct arrays of endogenous (y) and exogenous (x) variables.
        y, x_z, x_c = construct_arrays(data, group_name, y_name, z_names, c_names)

        n = np.size(y, 1)
        nk = np.size(x_z, 1)
        self.k = round(nk / n)
        k = self.k

        self.llt = llt
        self.alt = alt
        self.param_rest = param_rest
        self.cov_rest = cov_rest
        self.tau_start = tau_start
        self.var_start = var_start
        self.cov_start = cov_start

        # Intialize the state-space model.
        if self.param_rest == 'RC':
            if self.llt:
                # RC-LLT.
                k_states = 2 * n + k
            else:
                # RC.
                k_states = n + k
        else:
            if self.llt:
                # F/RSI-LLT.
                k_states = n * (k + 2)
            else:
                # F/RSI.
                k_states = n * (k + 1)
        super(SSMS, self).__init__(endog=y, exog=x_c, k_states=k_states, initialization='diffuse')

        # First part of Z matrix is NxN identity matrix for mu.
        z_mu = np.eye(n)

        # Split x_z matrix into k distinct parts for each time period.
        x_split = np.apply_along_axis(np.split, 1, x_z, indices_or_sections=k)
        if self.param_rest == 'RC':
            if self.llt:
                # RC-LLT.
                z_nu = np.zeros((n, n))

                # Concatenate matrices to form [I_N, O_N, x[t, 1], x[t, 2], ..., x[t, k]].
                x_concat = np.array([np.hstack((z_mu, z_nu, np.transpose(np.vstack(arr)))) for arr in x_split])
            else:
                # RC.
                # Concatenate matrices to form [I_N, x[t, 1], x[t, 2], ..., x[t, k]].
                x_concat = np.array([np.hstack((z_mu, np.transpose(np.vstack(arr)))) for arr in x_split])
        else:
            # F/RSI(-LLT).
            # Transform each sequence of x variables into a diagonal matrix.
            x_diag = np.apply_along_axis(np.diag, 2, x_split)
            if self.llt:
                # F/RSI-LLT.
                z_nu = np.zeros((n, n))

                # Concatenate matrices to form [I_N, O_N, diag(x[t, 1]), diag(x[t, 2]), ..., diag(x[t, k])].
                x_concat = np.array([np.hstack((z_mu, z_nu, np.hstack(tuple(arr)))) for arr in x_diag])
            else:
                # F/RSI.
                # Concatenate matrices to form [I_N, diag(x[t, 1]), diag(x[t, 2]), ..., diag(x[t, k])].
                x_concat = np.array([np.hstack((z_mu, np.hstack(tuple(arr)))) for arr in x_diag])

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
        c_k = np.size(self.exog, axis=1)

        if self.llt and self.alt:
            # Number of tau: k + 2.
            tau_start = np.ones(k + 2) * self.tau_start
        else:
            # Number of tau: k + 1.
            tau_start = np.ones(k + 1) * self.tau_start

        if self.llt:
            if self.param_rest == 'RSI':
                # Number of nu and lambda: (k_states - n) + (n + k) * c_k - n.
                other_start = np.zeros((self.k_states - n) + (n + k) * c_k - n)
            else:
                # Number of nu and lambda: (k_states - n) * (c_k + 1) - n.
                other_start = np.zeros((self.k_states - n) * (c_k + 1) - n)
        else:
            if self.param_rest == 'RSI':
                # Number of nu and lambda: k_states + (n + k) * c_k.
                other_start = np.zeros(self.k_states + (n + k) * c_k)
            else:
                # Number of nu and lambda: k_states * (c_k + 1).
                other_start = np.zeros(self.k_states * (c_k + 1))

        if self.cov_rest == 'F':
            # Full covariance matrix for y and all states.
            if self.param_rest == 'RC':
                # Only variances for the k states (not region-specific).
                if self.llt:
                    # Number of covariances: (n + C(n, 2)) + 2 * (n + C(n, 2)) + k
                    cov_start = np.ones(3 * (n + math.comb(n, 2)) + k) * self.cov_start
                else:
                    # Number of covariances: (n + C(n, 2)) + (n + C(n, 2)) + k.
                    cov_start = np.ones(2 * (n + math.comb(n, 2)) + k) * self.cov_start

                if self.llt:
                    # Change variances for y, mu, and nu.
                    n_full = 3
                else:
                    # Change variances for y and mu.
                    n_full = 2
                var_to = 0
                for i in range(n_full):
                    for row in range(n):
                        var_from = var_to
                        var_to += n - row
                        cov_start[var_from] = self.var_start

                # Change variances for all states corresponding to independent variables.
                for i in range(k):
                    var_from = var_to
                    var_to += 1
                    cov_start[var_from] = self.var_start
            else:
                if self.llt:
                    # Number of covariances: (n + C(n, 2)) + (k + 2) * (n + C(n, 2)).
                    cov_start = np.ones((k + 3) * (n + math.comb(n, 2))) * self.cov_start
                else:
                    # Number of covariances: (n + C(n, 2)) + (k + 1) * (n + C(n, 2)).
                    cov_start = np.ones((k + 2) * (n + math.comb(n, 2))) * self.cov_start

                # Change variances.
                if self.llt:
                    other = 3
                else:
                    other = 2
                var_to = 0
                for i in range(k + other):
                    for row in range(n):
                        var_from = var_to
                        var_to += n - row
                        cov_start[var_from] = self.var_start
        elif self.cov_rest == 'RSC':
            # Full covariance matrix for y.
            if self.param_rest == 'RC':
                # For mu, n variances, one covariance.
                # Only variances for the k states (not region-specific).
                if self.llt:
                    # For nu, n variances, one covariance.
                    # Number of covariances: (n + C(n, 2)) + 2 * (n + 1) + k
                    cov_start = np.ones((n + math.comb(n, 2)) + 2 * (n + 1) + k) * self.cov_start
                else:
                    # Number of covariances: (n + C(n, 2)) + (n + 1) + k
                    cov_start = np.ones((n + math.comb(n, 2)) + (n + 1) + k) * self.cov_start

                # Change variances for y.
                var_to = 0
                for row in range(n):
                    var_from = var_to
                    var_to += n - row
                    cov_start[var_from] = self.var_start

                # Change variances for mu.
                var_from = var_to
                var_to += n + 1
                cov_start[var_from:var_from + n] = self.var_start

                if self.llt:
                    # Change variances for nu.
                    var_from = var_to
                    var_to += n + 1
                    cov_start[var_from:var_from + n] = self.var_start

                # Change variances for all states corresponding to independent variables.
                for i in range(k):
                    var_from = var_to
                    var_to += 1
                    cov_start[var_from] = self.var_start
            else:
                # For each state covariance, n variances, one covariance.
                if self.llt:
                    # Number of covariances: (n + C(n, 2)) + (k + 2) * (n + 1).
                    cov_start = np.ones((n + math.comb(n, 2)) + (k + 2) * (n + 1)) * self.cov_start
                else:
                    # Number of covariances: (n + C(n, 2)) + (k + 1) * (n + 1).
                    cov_start = np.ones((n + math.comb(n, 2)) + (k + 1) * (n + 1)) * self.cov_start

                # Change variances for y.
                var_to = 0
                for row in range(n):
                    var_from = var_to
                    var_to += n - row
                    cov_start[var_from] = self.var_start

                # Change variances for all states.
                if self.llt:
                    other = 2
                else:
                    other = 1
                for i in range(k + other):
                    var_from = var_to
                    var_to += n + 1
                    cov_start[var_from:var_from + n] = self.var_start
        else:
            # Full covariance matrix for y.
            if self.param_rest == 'RC':
                # For mu, n variances, no covariance.
                # Only variances for the k states (not region-specific).
                if self.llt:
                    # For nu, n variances, no covariance.
                    # Number of covariances: (n + C(n, 2)) + 2 * n + k
                    cov_start = np.ones((n + math.comb(n, 2)) + 2 * n + k) * self.cov_start
                else:
                    # Number of covariances: (n + C(n, 2)) + n + k
                    cov_start = np.ones((n + math.comb(n, 2)) + n + k) * self.cov_start

                # Change variances for y.
                var_to = 0
                for row in range(n):
                    var_from = var_to
                    var_to += n - row
                    cov_start[var_from] = self.var_start

                # Change variances for mu.
                var_from = var_to
                var_to += n
                cov_start[var_from:var_to] = self.var_start

                if self.llt:
                    # Change variances for nu.
                    var_from = var_to
                    var_to += n
                    cov_start[var_from:var_to] = self.var_start

                # Change variances for all states corresponding to independent variables.
                for i in range(k):
                    var_from = var_to
                    var_to += 1
                    cov_start[var_from] = self.var_start
            else:
                # For each state covariance, n variances, no covariance.
                if self.llt:
                    # Number of covariances: (n + C(n, 2)) + (k + 2) * n.
                    cov_start = np.ones((n + math.comb(n, 2)) + (k + 2) * n) * self.cov_start
                else:
                    # Number of covariances: (n + C(n, 2)) + (k + 1) * n.
                    cov_start = np.ones((n + math.comb(n, 2)) + (k + 1) * n) * self.cov_start

                # Change variances for y.
                var_to = 0
                for row in range(n):
                    var_from = var_to
                    var_to += n - row
                    cov_start[var_from] = self.var_start

                # Change variances for all states.
                if self.llt:
                    other = 2
                else:
                    other = 1
                for i in range(k + other):
                    var_from = var_to
                    var_to += n
                    cov_start[var_from:var_to] = self.var_start
        return np.concatenate((tau_start, other_start, cov_start))

    def transform_params(self, unconstrained):
        """
        Restrict covariances to be non-negative.

        :param unconstrained: unconstrained parameters
        :return: constrained parameters
        """

        n = self.k_endog
        k = self.k
        c_k = np.size(self.exog, axis=1)
        constrained = unconstrained.copy()

        # Force covariances to be positive.
        if self.llt:
            if self.alt:
                n_tau = k + 2
            else:
                n_tau = k + 1
            if self.param_rest == 'RSI':
                # Number of nu and lambda: (k_states - n) + (n + k) * c_k - n.
                n_params = n_tau + (self.k_states - n) + (n + k) * c_k - n
            else:
                # Number of nu and lambda: (k_states - n) * (c_k + 1) - n.
                n_params = n_tau + (self.k_states - n) * (c_k + 1) - n
        else:
            if self.param_rest == 'RSI':
                # Number of nu and lambda: k_states + (n + k) * c_k.
                n_params = (k + 1) + self.k_states + (n + k) * c_k
            else:
                # Number of nu and lambda: k_states * (c_k + 1).
                n_params = (k + 1) + self.k_states * (c_k + 1)
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
        c_k = np.size(self.exog, axis=1)
        unconstrained = constrained.copy()

        # Force covariances to be positive.
        if self.llt:
            if self.alt:
                n_tau = k + 2
            else:
                n_tau = k + 1
            if self.param_rest == 'RSI':
                # Number of nu and lambda: (k_states - n) + (n + k) * c_k - n.
                n_params = n_tau + (self.k_states - n) + (n + k) * c_k - n
            else:
                # Number of nu and lambda: (k_states - n) * (c_k + 1) - n.
                n_params = n_tau + (self.k_states - n) * (c_k + 1) - n
        else:
            if self.param_rest == 'RSI':
                # Number of nu and lambda: k_states + (n + k) * c_k.
                n_params = (k + 1) + self.k_states + (n + k) * c_k
            else:
                # Number of nu and lambda: k_states * (c_k + 1).
                n_params = (k + 1) + self.k_states * (c_k + 1)
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
        k_c = np.size(self.exog, axis=1)

        # Set T_t = T.
        index_from = 0

        if self.alt:
            index_to = k + 2
        else:
            index_to = k + 1

        if self.param_rest == 'RC':
            if self.llt:
                if self.alt:
                    # RC-LLT-alt.
                    tau1_vec = np.ones(n) * params[index_from]
                    tau2_vec = np.ones(n) * params[index_from + 1]
                    mat = np.diag(np.concatenate((tau1_vec, tau2_vec, params[index_from + 2:index_to])))
                    mat[:n, n:2 * n] = np.eye(n)
                else:
                    # RC-LLT.
                    tau1_vec = np.ones(n) * params[index_from]
                    mat = np.diag(np.concatenate((tau1_vec, np.ones(n), params[index_from + 1:index_to])))
                    mat[:n, n:2 * n] = np.eye(n)
            else:
                # RC.
                tau1_vec = np.ones(n) * params[index_from]
                mat = np.diag(np.concatenate((tau1_vec, params[index_from + 1:index_to])))
        else:
            if self.llt:
                if self.alt:
                    # F/RSI-LLT-alt.
                    tau1_vec = np.ones(n) * params[index_from]
                    tau2_vec = np.ones(n) * params[index_from + 1]
                    tau3_vec = np.concatenate([np.ones(n) * param for param in params[index_from + 2:index_to]])
                    mat = np.diag(np.concatenate((tau1_vec, tau2_vec, tau3_vec)))
                    mat[:n, n:2 * n] = np.eye(n)
                else:
                    # F/RSI-LLT.
                    tau1_vec = np.ones(n) * params[index_from]
                    tau2_vec = np.concatenate([np.ones(n) * param for param in params[index_from + 1:index_to]])
                    mat = np.diag(np.concatenate((tau1_vec, np.ones(n), tau2_vec)))
                    mat[:n, n:2 * n] = np.eye(n)
            else:
                # F/RSI.
                param_vec = np.concatenate([np.ones(n) * param for param in params[index_from:index_to]])
                mat = np.diag(param_vec)
        self["transition"] = mat

        # Set c_t.
        if self.llt:
            # Set intercept for mu (just the lambda).
            for obs in range(n):
                index_from = index_to
                index_to += k_c
                col = 0

                for x in range(k_c):
                    col += params[index_from + x] * self.exog[:, x]
                self["state_intercept", obs, :] = col

            # Skip over the nu.
            start = 2 * n
        else:
            # Set intercept for mu.
            for obs in range(n):
                index_from = index_to
                index_to += k_c + 1
                col = params[index_from]

                for x in range(k_c):
                    col += params[index_from + x + 1] * self.exog[:, x]
                self["state_intercept", obs, :] = col
            start = n

        if self.param_rest == 'RSI':
            # Set intercept for beta (nu and region-invariant lambda).
            for var in range(k):
                # Set (common) lambda.
                index_from = index_to
                index_to += k_c
                common = 0

                for x in range(k_c):
                    common += params[index_from + x] * self.exog[:, x]

                # Set nu.
                for obs in range(n):
                    index_from = index_to
                    index_to += 1
                    col = params[index_from] + common
                    self["state_intercept", start + var * n + obs, :] = col
        else:
            # Set intercept for beta (nu and lambda).
            for state in range(start, self.k_states):
                index_from = index_to
                index_to += k_c + 1
                col = params[index_from]

                for x in range(k_c):
                    col += params[index_from + x + 1] * self.exog[:, x]
                self["state_intercept", state, :] = col

        # Set H_t = H and allow for off-diagonal elements (correlation between time series).
        for i in range(n):
            index_from = index_to
            index_to += n - i
            self["obs_cov", i, i:n] = params[index_from:index_to]
            self["obs_cov", i:n, i] = params[index_from:index_to]

        # Set Q_t = Q.
        if self.param_rest == 'RC':
            if self.llt:
                # Covariance matrices: mu and nu.
                blocks = 2
                solos = k
            else:
                # Covariance matrix: mu.
                blocks = 1
                solos = k
        else:
            if self.llt:
                # Covariance matrices: mu, nu, and betas.
                blocks = k + 2
                solos = 0
            else:
                # Covariance matrices: mu and betas.
                blocks = k + 1
                solos = 0

        # Set covariance matrices.
        cov_to = 0
        if self.cov_rest == 'F':
            # Allow for off-diagonal elements (correlation between time series).
            for block in range(blocks):
                cov_from = cov_to
                cov_to += n
                for i in range(cov_from, cov_to):
                    index_from = index_to
                    index_to += cov_to - i
                    self["state_cov", i, i:cov_to] = params[index_from:index_to]
                    self["state_cov", i:cov_to, i] = params[index_from:index_to]
        elif self.cov_rest == 'RSC':
            # Allow for off-diagonal elements, but restrict them to be the same across time-series.
            for block in range(blocks):
                index_from = index_to
                index_to += n + 1
                cov_from = cov_to
                cov_to += n
                variances = np.diag(params[index_from:index_to - 1])
                covariances = np.ones((n, n)) * params[index_to - 1] - np.diag(np.ones(n) * params[index_to - 1])
                self["state_cov", cov_from:cov_to, cov_from:cov_to] = variances + covariances
        else:
            # Do not allow for off-diagonal elements (set to zero).
            for block in range(blocks):
                index_from = index_to
                index_to += n
                cov_from = cov_to
                cov_to += n
                variances = np.diag(params[index_from:index_to])
                self["state_cov", cov_from:cov_to, cov_from:cov_to] = variances

        # Set solo variances.
        index_from = index_to
        index_to += solos
        cov_from = cov_to
        cov_to += solos
        self["state_cov", cov_from:cov_to, cov_from:cov_to] = np.diag(params[index_from:index_to])


def construct_arrays(data: pd.DataFrame, group_name: str, y_name: str, z_names: list, c_names: list):
    """
    Constructs arrays of endogenous (y) and exogenous (x) variables.

    :param data: a dataframe
    :param group_name: the column name of the grouping variable for each time series (e.g., region)
    :param y_name: the column name of the dependent variable
    :param z_names: a list of column names of the independent variables to be placed in the Z (design) matrix
    :param c_names: a list of column names of the independent variables to be placed in the c (state intercept) matrix
    :return: a tuple (group_names, y, x_z, x_c), with group_names Nx1 array of group names, y TxN array of y values (
        T periods, N observed time series), x_z Tx(N*K) array of x values (T periods, N*K regressors) of the form [x_11,
        x_12, ..., x_1N, x_21, ..., x_KN], and x_c TxC array of x values that are constant across observations (but vary
        over time).
    """

    # Filter data (drop all unnecessary columns).
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
    return group_names, y, x_z, x_c
