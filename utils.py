"""
This module provides utility methods.
"""
import datetime
import os

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from plotnine import *
from statsmodels.tsa.statespace.kalman_smoother import SmootherResults
from statsmodels.tsa.statespace.mlemodel import MLEResults, MLEResultsWrapper

from state_space import SSMS


def plot_states(filtered_results: MLEResultsWrapper, smoothed_results: SmootherResults, regions: list, z_names: list,
                save_path: str):
    """
    Plots states (all variables specified in z_names) and saves it in save_path.
    The dataframe states contains all the states (mu, nu, z_names) over time.

    :param filtered_results: filtered results from a SSMS class
    :param smoothed_results: smoothed results from a SSMS class, smoothed results should be an MLEResultsWrapper
    if you don't wanted smoothed states
    :param regions: list of region names
    :param z_names: a list of column names of the independent variables to be placed in the Z (design) matrix
    :param save_path: save path for plots
    :return:
    """

    n_regions = len(regions)
    n_betas = len(z_names)
    # Create confidence intervals for states (first n_regions*3 parameters are for the variances of y, mu and nu)
    if isinstance(smoothed_results, MLEResultsWrapper):
        states = np.transpose(filtered_results.filtered_state)
        cis = np.zeros((states.shape[0], n_betas * 3))
        # We use the state_cov (covariance matrix of state equation Q) to calculate the ci's
        bound = 1.96 * np.sqrt(filtered_results.params[n_regions * 3:])
    else:
        states = np.transpose(smoothed_results.smoothed_state)
        cis = np.zeros((states.shape[0], n_betas * 3))
        # We use the state_cov (covariance matrix of state equation Q) to calculate the ci's
        bound = 1.96 * np.sqrt(filtered_results.params[n_regions * 3:])
    for i in range(n_betas):
        cis[:, i] = states[:, n_regions * 2 + i] - bound[i]
        cis[:, i + n_betas] = states[:, n_regions * 2 + i] + bound[i]
        cis[:, i + n_betas * 2] = np.multiply(cis[:, i], cis[:, i + n_betas])
        cis[:, i + n_betas * 2][cis[:, i + n_betas * 2] < 0] = 0
        cis[:, i + n_betas * 2][cis[:, i + n_betas * 2] > 0] = 1
    # Create list cols with columns names for states Dataframe
    cols = []
    for i in range(states.shape[1] + n_betas * 3):
        if i < n_regions:
            cols.append('nu_' + regions[i])
        elif n_regions <= i < n_regions * 2:
            cols.append('mu_' + regions[i - n_regions])
        elif n_regions * 2 <= i < n_regions * 2 + n_betas:
            cols.append(z_names[i - n_regions * 2])
        elif n_regions * 2 + n_betas <= i < n_regions * 2 + n_betas * 2:
            cols.append(z_names[i - (n_regions * 2 + n_betas)] + '_lb')
        elif n_regions * 2 + n_betas * 2 <= i < n_regions * 2 + n_betas * 3:
            cols.append(z_names[i - (n_regions * 2 + n_betas * 2)] + '_ub')
        else:
            cols.append(z_names[i - (n_regions * 2 + n_betas * 3)] + '_significant')
    states_df = pd.DataFrame(np.concatenate((states, cis), axis=1), columns=cols)
    states_df['Date'] = pd.date_range(start='1/1/2018', periods=len(states_df), freq='W')
    states_df_01 = states_df.iloc[:, -(n_betas * 4 + 1):]
    states_df_01['Date'] = states_df_01['Date'].dt.strftime('%G%V')
    if isinstance(smoothed_results, MLEResultsWrapper):
        states_df_01.to_excel(os.path.join(save_path, 'states_filtered.xlsx'))
    else:
        states_df_01.to_excel(os.path.join(save_path, 'states_smoothed.xlsx'))
    # The first 5 observations are removed for nice graphs
    states_df = states_df.iloc[5:, :]
    # Important events are the first intelligent lockdown and relaxation of rules
    events = [datetime.datetime.strptime('2020-11-1', '%G-%V-%u'), datetime.datetime.strptime('2020-27-1', '%G-%V-%u')]
    events_full = [*events, *[datetime.datetime.strptime('2020-51-1', '%G-%V-%u'),
                              datetime.datetime.strptime('2021-25-1', '%G-%V-%u')]]
    for i in range(n_betas):
        if i == z_names.index('StringencyIndex'):
            # Remove 0-values when plotting StringencyIndex
            states_df_02 = states_df[108:]
        else:
            states_df_02 = states_df
        p = ggplot(states_df_02, aes(x='Date')) + scale_x_datetime(breaks=get_ticks(states_df_02, 8)[0],
                                                                   labels=get_ticks(states_df_02, 8)[1]) + geom_ribbon(
            aes(ymin=states_df_02.iloc[:, n_regions * 2 + n_betas + i],
                ymax=states_df_02.iloc[:, n_regions * 2 + n_betas * 2 + i], color='"95% CI"'), alpha=0.1) + geom_line(
            aes(y=states_df_02.columns[n_regions * 2 + i], color='"State"')) + geom_vline(xintercept=events_full,
                                                                                          linetype="dotted") + \
            geom_vline(
            xintercept=[datetime.datetime.strptime('2020-50-1', '%G-%V-%u')], linetype="solid") + scale_color_manual(
            values=['#dedede', '#4472c4']) + labs(x='Date', y='State', color='Legend')
        if isinstance(smoothed_results, MLEResultsWrapper):
            ggsave(plot=p, filename='coefficient_for_filtered_' + z_names[i], path=save_path, verbose=False, dpi=600)
        else:
            ggsave(plot=p, filename='coefficient_for_smoothed_' + z_names[i], path=save_path, verbose=False,
                   dpi=600)  # print(p)


def forecast_error(results: MLEResults, regions: list, save_path: str, first=int, last=int, ci=bool, tp=str, n_plots=4):
    """
    Computes forecast error with one-step ahead forecasts for each region and saves it in save_path.
    Plots forecasts, actual sales and errors of the n_plots best/worst MASE/MdASE regions

    :param results: (extended) results (from prepare_forecast())
    :param regions: list of region names,
    the order of the names should be exactly the same as the order of the regions in the model
    :param save_path: save path for plots
    :param first: the time index from where your plots should start
    :param last: this time index should exactly be equal to the time index-1 where the sample of the model ends
    :param ci: whether to plot a confidence interval (True) or not (False),
    if the CI's become too big set ci=False otherwise the sales will be plotted as straight lines
    :param tp: specify the type of data (e.g. one_step_ahead_forecast) you want to plot,
    use _ instead of spaces for tp, since the name of the plots/excel files will also have this name
    :param n_plots: the number of regions to plot, 4 (default) implies plotting the forecasts, actual sales
    and errors of the 4 best/worst MASE/MdASE regions (= 3 * 4 * 2 * 2 = 48 plots)
    :return:
    """

    n_regions = len(regions)
    model = results.model
    data = results.get_prediction(start=first, end=last)

    # Calculate MASE using one-step ahead forecasts
    mases = np.zeros(len(regions))
    maes = np.zeros((38, len(regions)))
    maes_naive = np.zeros((152, len(regions)))
    mdases = np.zeros(len(regions))
    for region in range(len(regions)):
        maes[:, region] = np.abs(model.endog[first:, region] - data.predicted_mean[:, region])
        maes_naive[:, region] = np.abs(
            [x - model.endog[0:153, region][i - 1] for i, x in enumerate(model.endog[0:153, region])][1:])
        mases[region] = np.mean(maes[:, region]) / np.mean(maes_naive[:, region])
        mdases[region] = np.median(maes[:, region]) / np.median(maes_naive[:, region])
    mean_mase = np.mean(mases)
    med_mase = np.median(mases)
    mean_mdase = np.mean(mdases)
    med_mdase = np.median(mdases)
    l1_mase = sum(x < 1 for x in mases) / mases.shape[0]
    l1_mdase = sum(x < 1 for x in mdases) / mdases.shape[0]

    best_mase, worst_mase = np.argmin(mases), np.argmax(mases)
    best_mdase, worst_mdase = np.argmin(mdases), np.argmax(mdases)
    mase_df = pd.DataFrame(np.transpose(mases.reshape(1, n_regions)), index=regions, columns=['MASE'])
    mdase_df = pd.DataFrame(np.transpose(mdases.reshape(1, n_regions)), index=regions, columns=['MdASE'])
    error_df = mase_df.merge(mdase_df, left_index=True, right_index=True, how='left')
    error_df[''] = ''
    error_df['Best MASE'] = [regions[best_mase], mases[best_mase], '', 'Best MdASE', regions[best_mdase],
                             mdases[best_mdase]] + [''] * (len(error_df) - 6)
    error_df['Worst MASE'] = [regions[worst_mase], mases[worst_mase], '', 'Worst MdASE', regions[worst_mdase],
                              mdases[worst_mdase]] + [''] * (len(error_df) - 6)
    error_df['Mean MASE'] = [mean_mase] + [''] * 2 + ['Mean MdASE', mean_mdase] + [''] * (len(error_df) - 5)
    error_df['Median MASE'] = [med_mase] + [''] * 2 + ['Median MdASE', med_mdase] + [''] * (len(error_df) - 5)
    error_df['Proportion of regions MASE<1'] = [l1_mase] + [''] * 2 + ['Proportion of regions MdASE<1', l1_mdase] + [
        ''] * (len(error_df) - 5)
    error_df.to_excel(os.path.join(save_path, 'errors_' + tp + '.xlsx'))

    # Plot forecasts (df_pred), actual sales (df_full) and MAE/MAE_naive (df_mae)
    df_pred = pd.DataFrame(np.concatenate((model.endog[first:, :], data.predicted_mean, data.conf_int()), axis=1))
    start_date = datetime.datetime(2018, 1, 1) + datetime.timedelta(weeks=first)
    df_pred['Date'] = pd.date_range(start=start_date, periods=len(df_pred), freq='W')

    df_full = pd.DataFrame(model.endog)
    df_full['Date'] = pd.date_range(start=datetime.datetime(2018, 1, 1), periods=len(df_full), freq='W')

    df_mae = pd.DataFrame(np.concatenate((maes_naive, maes), axis=0))
    df_mae['Date'] = pd.date_range(start=datetime.datetime(2018, 1, 8), periods=len(df_mae), freq='W')

    plot_regions = np.concatenate((mases.argsort()[:n_plots], mases.argsort()[-n_plots:][::-1],
                                   mdases.argsort()[:n_plots], mdases.argsort()[-n_plots:][::-1]), axis=0)
    # Important events are the second lockdown and relaxation of (almost all) rules
    events_test = [datetime.datetime.strptime('2020-51-1', '%G-%V-%u'),
                   datetime.datetime.strptime('2021-25-1', '%G-%V-%u')]
    events_full = [
        *[datetime.datetime.strptime('2020-11-1', '%G-%V-%u'), datetime.datetime.strptime('2020-27-1', '%G-%V-%u')],
        *events_test]
    for i in range(plot_regions.shape[0]):
        if ci:
            p = ggplot(df_pred, aes(x='Date')) + scale_x_datetime(breaks=get_ticks(df_pred, 8)[0],
                                                                  labels=get_ticks(df_pred, 8)[1]) + geom_ribbon(
                aes(ymin=df_pred.iloc[:, n_regions * 2 + plot_regions[i]],
                    ymax=df_pred.iloc[:, n_regions * 3 + plot_regions[i]], color='"95% CI"'), alpha=0.1) + geom_line(
                aes(y=df_pred.iloc[:, plot_regions[i]], color='"Actual"')) + geom_line(
                aes(y=df_pred.iloc[:, n_regions + plot_regions[i]], color='"Forecast"')) + geom_vline(
                xintercept=events_test, linetype="dotted") + scale_color_manual(
                values=['#dedede', '#4472c4', '#ed7d31']) + labs(x='Date', y='Sales', color='Legend')
            q = ggplot(df_full, aes(x='Date')) + scale_x_datetime(breaks=get_ticks(df_full, 8)[0],
                                                                  labels=get_ticks(df_full, 8)[1]) + geom_line(
                aes(y=df_full.iloc[:, plot_regions[i]], color='"Actual"')) + geom_vline(xintercept=events_full,
                                                                                        linetype="dotted") + geom_vline(
                xintercept=[datetime.datetime.strptime('2020-50-1', '%G-%V-%u')],
                linetype="solid") + scale_color_manual(values=['#4472c4']) + labs(x='Date', y='Sales', color='Legend')
            m = ggplot(df_mae, aes(x='Date')) + scale_x_datetime(breaks=get_ticks(df_mae, 8)[0],
                                                                 labels=get_ticks(df_mae, 8)[1]) + geom_line(
                aes(y=df_mae.iloc[0:152, plot_regions[i]], color='"AE_naive"'),
                data=df_mae['Date'][0:152].to_frame()) + geom_line(
                aes(y=df_mae.iloc[151:190, plot_regions[i]], color='"AE"'),
                data=df_mae['Date'][151:190].to_frame()) + geom_vline(xintercept=events_full,
                                                                      linetype="dotted") + geom_vline(
                xintercept=[datetime.datetime.strptime('2020-50-1', '%G-%V-%u')],
                linetype="solid") + scale_color_manual(values=['#4472c4', '#ed7d31']) + labs(x='Date', y='Error',
                                                                                             color='Legend')
        else:
            p = ggplot(df_pred, aes(x='Date')) + scale_x_datetime(breaks=get_ticks(df_pred, 8)[0],
                                                                  labels=get_ticks(df_pred, 8)[1]) + geom_line(
                aes(y=df_pred.iloc[:, plot_regions[i]], color='"Actual"')) + geom_line(
                aes(y=df_pred.iloc[:, n_regions + plot_regions[i]], color='"Forecast"')) + geom_vline(
                xintercept=events_test, linetype="dotted") + labs(x='Date', y='Sales')
        # print(m)
        if i < n_plots:
            ggsave(plot=p, filename=tp + '_mase_best_' + str(i + 1) + '_' + regions[plot_regions[i]], path=save_path,
                   verbose=False, dpi=600)
            ggsave(plot=q, filename='actual_sales_mase_best_' + str(i + 1) + '_' + regions[plot_regions[i]],
                   path=save_path, verbose=False, dpi=600)
            ggsave(plot=m, filename='mase_best_' + str(i + 1) + '_' + regions[plot_regions[i]], path=save_path,
                   verbose=False, dpi=600)
        elif i < n_plots * 2:
            ggsave(plot=p, filename=tp + '_mase_worst_' + str(i - n_plots + 1) + '_' + regions[plot_regions[i]],
                   path=save_path, verbose=False, dpi=600)
            ggsave(plot=q, filename='actual_sales_mase_worst_' + str(i - n_plots + 1) + '_' + regions[plot_regions[i]],
                   path=save_path, verbose=False, dpi=600)
            ggsave(plot=m, filename='mase_worst_' + str(i - n_plots + 1) + '_' + regions[plot_regions[i]],
                   path=save_path, verbose=False, dpi=600)
        elif i < n_plots * 3:
            ggsave(plot=p, filename=tp + '_mdase_best_' + str(i - n_plots * 2 + 1) + '_' + regions[plot_regions[i]],
                   path=save_path, verbose=False, dpi=600)
            ggsave(plot=q,
                   filename='actual_sales_mdase_best_' + str(i - n_plots * 2 + 1) + '_' + regions[plot_regions[i]],
                   path=save_path, verbose=False, dpi=600)
            ggsave(plot=m, filename='mdase_best_' + str(i - n_plots * 2 + 1) + '_' + regions[plot_regions[i]],
                   path=save_path, verbose=False, dpi=600)
        else:
            ggsave(plot=p, filename=tp + '_mdase_worst_' + str(i - n_plots * 3 + 1) + '_' + regions[plot_regions[i]],
                   path=save_path, verbose=False, dpi=600)
            ggsave(plot=q,
                   filename='actual_sales_mdase_worst_' + str(i - n_plots * 3 + 1) + '_' + regions[plot_regions[i]],
                   path=save_path, verbose=False, dpi=600)
            ggsave(plot=m, filename='mdase_worst_' + str(i - n_plots * 3 + 1) + '_' + regions[plot_regions[i]],
                   path=save_path, verbose=False, dpi=600)


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
    Pretty-prints the results for an SSMS model with k variables of interest (in beta equations). Assumes n > k.

    :param results: results object for an SSMS model
    :param save_path: path to save location
    :param name: model name
    :return:
    """
    model = results.model
    if not isinstance(model, SSMS):
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
    # if not isinstance(model):
    #     print("Can't prepare forecasts for a non-SSMS model.")
    #     return

    new_model = SSMS(data, group_name=model.group_name, y_name=model.y_name, z_names=model.z_names,
                     cov_rest=model.cov_rest)
    fitted_params = results.params
    new_result = new_model.filter(fitted_params)
    return new_model, new_result
