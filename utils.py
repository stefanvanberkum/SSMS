"""
This module provides utility methods.
"""
import os
import datetime

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from statsmodels.tsa.statespace.mlemodel import MLEResults
from plotnine import *

from state_space import SSMS, SSMS_alt, SSMS_alt_4


def plot_states(results: MLEResults, regions: list, z_names: list, save_path: str):
    """
    Plots states (all variables specified in z_names) and saves it in save_path.
    The dataframe states contains all the states (mu, nu, z_names) over time.

    :param results: train results from a SSMS_alt_4 class
    (this function might not work for with results from other classes)
    :param regions: list of region names
    :param z_names: a list of column names of the independent variables to be placed in the Z (design) matrix
    :param save_path: save path for plots
    :return:
    """

    n_regions = len(regions)
    n_betas = len(z_names)
    # Create confidence intervals for states (first n_regions*3 parameters are for the variances of y, mu and nu),
    cis = np.zeros((results.states.filtered.shape[0], n_betas*2))
    bound = 1.96 * np.sqrt(results.params[n_regions*3:])
    for state in range(n_betas):
        # CI for first observation (incorrectly) uses last observation,
        # but the first 6 observations (including CI's) are removed later
        cis[:, state] = np.roll(results.states.filtered[:, n_regions*2 + state], 1, axis=0) - bound[state]
        cis[:, state+n_betas] = np.roll(results.states.filtered[:, n_regions*2 + state], 1, axis=0) + bound[state]
    # Create list cols with columns names for states Dataframe
    cols = []
    for i in range(results.states.filtered.shape[1] + n_betas*2):
        if i < n_regions:
            cols.append('nu_'+regions[i])
        elif n_regions <= i < n_regions*2:
            cols.append('mu_'+regions[i-n_regions])
        elif n_regions*2 <= i < n_regions*2 + n_betas:
            cols.append(z_names[i-n_regions*2])
        elif n_regions*2 + n_betas <= i < n_regions*2 + n_betas*2:
            cols.append(z_names[i-(n_regions*2 + n_betas)] + '_lb')
        else:
            cols.append(z_names[i-(n_regions*2 + n_betas*2)] + '_ub')
    states = pd.DataFrame(np.concatenate((results.states.filtered, cis), axis=1), columns=cols)
    states['Date'] = pd.date_range(start='1/1/2018', periods=len(states), freq='W')
    # The first 6 observations are removed for nice graphs
    states = states.iloc[6:, :]
    # Important events are the first intelligent lockdown and relaxation of rules
    events = [datetime.datetime.strptime('12-03-2020', '%d-%m-%Y'), datetime.datetime.strptime('24-06-2020', '%d-%m-%Y')]
    for i in range(n_betas):
        if i == z_names.index('StringencyIndex'):
            # Remove 0-values when plotting StringencyIndex
            states = states[108:]
        p = ggplot(states, aes(x='Date')) \
            + scale_x_datetime(breaks=get_ticks(states, 8)[0], labels=get_ticks(states, 8)[1]) \
            + geom_ribbon(
            aes(ymin=states.iloc[:, n_regions*2 + n_betas + i], ymax=states.iloc[:, n_regions*2 + n_betas*2 + i],
                color='"95% CI"'), alpha=0.1) \
            + geom_line(aes(y=states.columns[n_regions*2+i], color='"State"')) \
            + geom_vline(xintercept=events, linetype="dotted") \
            + scale_color_manual(values=['#dedede', '#4472c4']) \
            + labs(x='Date', y='State', color='Legend')
        # print(p)
        ggsave(plot=p, filename='coefficient_for_'+z_names[i], path=save_path, verbose=False, dpi=600)


def forecast_error(results: MLEResults, model, regions: list, save_path: str, first=int, last=int, ci=bool, tp=str):
    """
    Computes forecast error with one-step ahead forecasts for each region and saves it in save_path.
    Plots SalesGoodsEUR with tp for regions indices specified in plot_regions and saves it in save_path.
    For now only the plots for the regions with the largest/smallest error are saved.

    :param results: (extended) results (from prepare_forecast())
    :param model: (extended) model (from prepare_forecast())
    :param regions: list of region names,
    the order of the names should be exactly the same as the order of the regions in the model
    :param save_path: save path for plots
    :param first: the time index from where your plots should start
    :param last: this time index should exactly be equal to the time index-1 where the sample of the model ends
    :param ci: whether to plot a confidence interval (True) or not (False),
    if the CI's become too big set ci=False otherwise the sales will be plotted as straight lines
    :param tp: specify the type of data (e.g. in_sample_prediction or one_step_ahead_forecast) you want to plot,
    use _ instead of spaces in for tp, since the name of the plots/excel files will also have this name
    :return:
    """

    n_regions = len(regions)
    data = results.get_prediction(start=first, end=last)

    # Calculate MASE using one-step ahead forecasts
    # mses = np.zeros(len(regions))
    mases = np.zeros(len(regions))
    for region in range(len(regions)):
        # mses[region] = np.mean(np.square(model.endog[first:, region] - data.predicted_mean[:, region]))
        mae = np.mean(np.abs(model.endog[first:, region] - data.predicted_mean[:, region]))
        mae_naive = np.mean(np.abs([x - model.endog[0:153:, region][i - 1] for i, x in enumerate(model.endog[0:153:, region])][1:]))
        mases[region] = mae/mae_naive
    # mse = np.mean(mses)
    mase = np.mean(mases)

    best, worst = np.argmin(mases), np.argmax(mases)
    pd.DataFrame(np.transpose(mases.reshape(1, n_regions)), index=regions, columns=['MASE']).to_excel(
        os.path.join(save_path, 'mases_' + tp + '.xlsx'))
    print(f'{tp}s starting from t={first}')
    print(f'Average MASE of regions : {mase}')
    print(f'Region with smallest MASE: {regions[best]}, {mases[best]}')
    print(f'Region with largest MASE: {regions[worst]}, {mases[worst]}')
    print()

    # Plot data for regions
    df = pd.DataFrame(np.concatenate((model.endog[first:, :], data.predicted_mean, data.conf_int()), axis=1))
    start_date = datetime.datetime(2018, 1, 1) + datetime.timedelta(weeks=first)
    df['Date'] = pd.date_range(start=start_date, periods=len(df), freq='W')
    # Specify the indices of the regions you want to plot in plot_regions
    plot_regions = [best, worst]
    # Important events are the second lockdown and relaxation of (almost all) rules
    events = [datetime.datetime.strptime('14-12-2020', '%d-%m-%Y'), datetime.datetime.strptime('26-06-2021', '%d-%m-%Y')]
    for i in range(len(plot_regions)):
        if ci:
            p = ggplot(df, aes(x='Date')) \
                + scale_x_datetime(breaks=get_ticks(df, 8)[0], labels=get_ticks(df, 8)[1]) \
                + geom_ribbon(
                aes(ymin=df.iloc[:, n_regions * 2 + plot_regions[i]], ymax=df.iloc[:, n_regions * 3 + plot_regions[i]],
                    color='"95% CI"'), alpha=0.1) \
                + geom_line(aes(y=df.iloc[:, plot_regions[i]], color='"Actual"')) \
                + geom_line(aes(y=df.iloc[:, n_regions + plot_regions[i]], color='"Forecast"')) \
                + geom_vline(xintercept=events, linetype="dotted") \
                + scale_color_manual(values=['#dedede', '#4472c4', '#ed7d31']) \
                + labs(x='Date', y='Sales', color='Legend')
        else:
            p = ggplot(df, aes(x='Date')) \
                + scale_x_datetime(breaks=get_ticks(df, 8)[0], labels=get_ticks(df, 8)[1]) \
                + geom_line(aes(y=df.iloc[:, plot_regions[i]], color='"Actual"')) \
                + geom_line(aes(y=df.iloc[:, n_regions + plot_regions[i]], color='"Forecast"')) \
                + geom_vline(xintercept=events, linetype="dotted") \
                + scale_color_manual(values=['#4472c4', '#ed7d31']) \
                + labs(x='Date', y='Sales', color='Legend')
        # print(p)
        ggsave(plot=p, filename=regions[plot_regions[i]] + '_' + tp, path=save_path, verbose=False, dpi=600)


def get_ticks(data: pd.DataFrame, n_ticks: int):
    """
    Returns x_axis ticks as dates

    :param data: dataframe where the last column should contain pandas.Timestamp objects
    :param n_ticks: number of ticks
    :return: ticks (breaks) and their labels
    """
    x_breaks = []
    x_labels = []
    n_ticks = n_ticks - 1
    interval = data.shape[0] / n_ticks
    for i in range(n_ticks + 1):
        x_breaks.append(data.iloc[0, -1:][0] + datetime.timedelta(weeks=interval * i))
        x_labels.append((data.iloc[0, -1:][0] + datetime.timedelta(weeks=interval * i)).strftime('%G-%V'))
    return x_breaks, x_labels


def print_results(results: MLEResults, save_path: str, name: str):
    """
    Pretty-prints the results for an SSMS model with n + k variables of interest (n in either sales or mu,
    k in beta equations). Assumes n > k.

    :param results: results object for an SSMS model
    :param save_path: path to save location
    :param name: model name
    :return:
    """
    model = results.model
    if not (isinstance(model, SSMS) or isinstance(model, SSMS_alt)):
        print("Can't print parameters for a non-SSMS model.")
        return
    if isinstance(model, SSMS) and np.size(model.exog, axis=1) != 1:
        print("Can't print a model that does not have exactly one c/state intercept variable.")
        return
    if isinstance(model, SSMS_alt) and np.size(model.exog, axis=1) != 2:
        print("Can't print a model that does not have exactly one d/obs intercept and one c/state intercept variable.")
        return

    # Print AIC, BIC, MSE, and MAE.
    with open(os.path.join(save_path, name + '_stats.csv'), 'w') as out:
        header = ','.join(['AIC', 'BIC', 'MSE', 'MAE'])
        stats = ','.join([str(results.aic), str(results.bic), str(results.mse), str(results.mae)])
        out.write('\n'.join([header, stats]))

    # Print fitted parameters, standard errors, and p-values.
    regions = model.group_names
    params = results.params
    ses = results.bse
    pvalues = results.pvalues

    n = len(regions)
    k = model.k
    n_cov = model.n_cov

    param_from = 0
    param_to = n
    lambda_1 = params[param_from:param_to]
    l1_se = ses[param_from:param_to]
    l1_p = pvalues[param_from:param_to]

    param_names = model.z_names
    param_from = param_to
    param_to += k
    lambda_2 = params[param_from:param_to]
    l2_se = ses[param_from:param_to]
    l2_p = pvalues[param_from:param_to]

    y = ','.join(['region', 'var (y)'])
    l1 = ','.join(['lambda_1', 'se', 'p-value'])
    mu = 'var (mu)'
    nu = 'var (nu)'
    l2 = ','.join(['param', 'lambda_2', 'se', 'p-value', 'var'])
    header = ',,'.join([y, l1, mu, nu, l2])

    param_from = param_to
    if model.cov_rest == 'GC':
        param_to += n + n_cov
        y_var = params[param_from:param_from + n]
    else:
        param_to += n
        y_var = params[param_from:param_to]

    param_from = param_to
    param_to += n
    mu_var = params[param_from:param_to]

    param_from = param_to
    param_to += n
    nu_var = params[param_from:param_to]

    param_from = param_to
    param_to += k
    param_var = params[param_from:]

    with open(os.path.join(save_path, name + '_params.csv'), 'w') as out:
        out.write(header + '\n')

        for i in range(n):
            y = ','.join([regions[i], str(y_var[i])])
            mu = str(mu_var[i])
            nu = str(nu_var[i])
            l1 = ','.join([str(lambda_1[i]), str(l1_se[i]), str(l1_p[i])])
            line = ',,'.join([y, l1, mu, nu])

            if i < k:
                l2 = ','.join([param_names[i], str(lambda_2[i]), str(l2_se[i]), str(l2_p[i]), str(param_var[i])])
                line = ',,'.join([line, l2])
            out.write(line + '\n')


def print_results_alt(results: MLEResults, save_path: str, name: str):
    """
    Pretty-prints the results for an SSMS model with k variables of interest (in beta equations). Assumes n > k.

    :param results: results object for an SSMS model
    :param save_path: path to save location
    :param name: model name
    :return:
    """
    model = results.model
    if not isinstance(model, SSMS_alt_4):
        print("Can't print parameters for a non-SSMS model.")
        return

    # Print AIC, BIC, MSE, and MAE.
    with open(os.path.join(save_path, name + '_stats.csv'), 'w') as out:
        header = ','.join(['AIC', 'BIC', 'MSE', 'MAE'])
        stats = ','.join([str(results.aic), str(results.bic), str(results.mse), str(results.mae)])
        out.write('\n'.join([header, stats]))

    # Print fitted parameters, standard errors, and p-values.
    regions = model.group_names
    params = results.params

    n = len(regions)
    k = model.k
    n_cov = model.n_cov

    param_names = model.z_names

    y = ','.join(['region', 'var (y)'])
    mu = 'var (mu)'
    nu = 'var (nu)'
    lm = ','.join(['param', 'var'])
    header = ',,'.join([y, mu, nu, lm])

    param_from = 0
    if model.cov_rest == 'GC':
        param_to = n + n_cov
        y_var = params[param_from:param_from + n]
    else:
        param_to = n
        y_var = params[param_from:param_to]

    param_from = param_to
    param_to += n
    mu_var = params[param_from:param_to]

    param_from = param_to
    param_to += n
    nu_var = params[param_from:param_to]

    param_from = param_to
    param_to += k
    param_var = params[param_from:]

    with open(os.path.join(save_path, name + '_params.csv'), 'w') as out:
        out.write(header + '\n')

        for i in range(n):
            y = ','.join([regions[i], str(y_var[i])])
            mu = str(mu_var[i])
            nu = str(nu_var[i])
            line = ',,'.join([y, mu, nu])

            if i < k:
                lm = ','.join([param_names[i], str(param_var[i])])
                line = ',,'.join([line, lm])
            out.write(line + '\n')


def plot_variables(data: list, info: list, all_regions: False):
    """
    Plots variables.
    :param data: list of form [y, mu, threshold, obs_sd]
    :param info: list of from [index, name]
    :param all_regions: boolean to plot regions 1-by-1 (True) or all at the same time (False)
    :return:
    """
    if all_regions:
        if info:
            t = np.arange(1, len(data[0][0]) + 1)
            for i in range(len(info)):
                index = info[i][0]
                plt.figure()
                plt.suptitle(info[i][1])
                plt.plot(t, data[index][0], 'b')
                plt.plot(t, data[index][1] + data[index][2] * data[index][3], 'r')
                plt.plot(t, data[index][1] - data[index][2] * data[index][3], 'r')
                plt.show()
        else:
            print('No outliers')
    else:
        if info:
            t = np.arange(1, len(data[0][0]) + 1)
            for i in range(len(info)):
                index = info[i][0]
                plt.figure()
                plt.suptitle(info[i][1])
                plt.plot(t, data[index][0], 'b')
                plt.plot(t, data[index][1] + data[index][2] * data[index][3], 'r')
                plt.plot(t, data[index][1] - data[index][2] * data[index][3], 'r')
        else:
            print('No outliers')
        plt.show()


def prepare_forecast(results: MLEResults, data: pd.DataFrame):
    """
    Prepares a new MLEResults object, such that regular methods can be used to compute forecasts. For out-of-sample
    forecasts, we can simply use 'in-sample' forecasts of a model with fixed parameters, obtained from the initial fit.

    :param results: the MLEResults object of the training fit
    :param data: the extended data (train + test)
    :return: a new NLEResults object, fitted with fixed parameters obtained from the initial training fit
    """

    model = results.model
    if not isinstance(model, SSMS_alt_4):
        print("Can't prepare forecasts for a non-SSMS model.")
        return

    new_model = SSMS_alt_4(data, group_name=model.group_name, y_name=model.y_name, z_names=model.z_names,
                           cov_rest=model.cov_rest)
    fitted_params = results.params
    new_result = new_model.filter(fitted_params)
    return new_model, new_result
