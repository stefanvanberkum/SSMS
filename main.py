"""
This is the main environment for the state-space model.
"""

import os
import time

import numpy as np
from matplotlib import pyplot as plt

from data_loader import load_data
from state_space import SSMS

"""
TODO:
- Parameter printing (standard errors + p-values?)
- State printing
- Forecasting (train/test)
- SE
    - Full sample
    - Bootstrap
    - Sample + bootstrap
"""


def main():
    data_path = os.path.join(os.path.expanduser('~'), 'Documents', 'SSMS', 'data')
    save_path = os.path.join(os.path.expanduser('~'), 'Documents', 'SSMS', 'results')

    start_time = time.time()

    data = load_data(data_path)
    test_data = data[data['Region'].isin(['NL310_503', 'NL33C_340', 'NL33C_506', 'NL212_507'])]

    # z_names = ['WVO', 'SchoolHolidayMiddle', 'SchoolHolidayNorth', 'SchoolHolidaySouth',
    # '0-25_nbrpromos_index_201801',
    #           '25-50_nbrpromos_index_201801', '50-75_nbrpromos_index_201801']
    z_names = ['WVO', 'SchoolHolidayMiddle', '0-25_nbrpromos_index_201801', '25-50_nbrpromos_index_201801',
               '50-75_nbrpromos_index_201801']
    c_names = ['StringencyIndex']
    llt = True
    alt = False
    param_rest = 'RC'
    cov_rest = 'RSC'
    tau_start = 0.5
    var_start = 1
    cov_start = 0

    model = SSMS(data, group_name='Region', y_name='SalesGoodsEUR', z_names=z_names, c_names=c_names, llt=llt, alt=alt,
                 param_rest=param_rest, cov_rest=cov_rest, tau_start=tau_start, var_start=var_start,
                 cov_start=cov_start)
    # initial = model.fit(maxiter=1000, maxfun=1000000)
    # result = model.fit(initial.params, method='nm', maxiter=20000)
    # initial = model.fit(method='nm', maxiter=20000)
    # result = model.fit(initial.params, maxiter=1000, maxfun=100000)
    result = model.fit(maxiter=1000, maxfun=1000000)
    print(result.summary())
    # print_params(result, save_path)

    split_time = time.time()
    print("Split:", (split_time - start_time), sep=' ')

    y_pred = result.get_prediction(start=10, end=190)
    y1_pred = y_pred.predicted_mean[:, 0]
    mse_1 = np.mean(np.square(model.endog[10:, 0] - y1_pred))
    y2_pred = y_pred.predicted_mean[:, 1]
    mse_2 = np.mean(np.square(model.endog[10:, 1] - y2_pred))
    y3_pred = y_pred.predicted_mean[:, 2]
    mse_3 = np.mean(np.square(model.endog[10:, 2] - y3_pred))
    y4_pred = y_pred.predicted_mean[:, 3]
    mse_4 = np.mean(np.square(model.endog[10:, 3] - y4_pred))
    mse = (mse_1 + mse_2 + mse_3 + mse_4) / 4
    t = np.arange(11, 192)
    fig, axes = plt.subplots(2, 2)
    axes[0, 0].set_title('mse: {0}'.format(format(mse_1, '.4f')))
    axes[0, 0].set_xticks([])
    axes[0, 0].plot(t, model.endog[10:, 0], 'b')
    axes[0, 0].plot(t, y1_pred, 'r')
    axes[0, 1].set_title('mse: {0}'.format(format(mse_2, '.4f')))
    axes[0, 1].set_xticks([])
    axes[0, 1].plot(t, model.endog[10:, 1], 'b')
    axes[0, 1].plot(t, y2_pred, 'r')
    axes[1, 0].set_title('mse: {0}'.format(format(mse_3, '.4f')))
    axes[1, 0].plot(t, model.endog[10:, 2], 'b')
    axes[1, 0].plot(t, y3_pred, 'r')
    axes[1, 1].set_title('mse: {0}'.format(format(mse_4, '.4f')))
    axes[1, 1].plot(t, model.endog[10:, 3], 'b')
    axes[1, 1].plot(t, y4_pred, 'r')
    name = 'full_test'
    fig.suptitle('{0} (mse: {1})'.format(name, format(mse, '.4f')))
    plt.savefig(os.path.join(save_path, name), dpi=300, format='png')
    np.savetxt(os.path.join(save_path, name + '.csv'), result.params, delimiter=',')
    plt.close('all')

    end_time = time.time()
    print("Runtime:", (end_time - start_time), sep=' ')

    result.save(os.path.join(save_path, 'result.pickle'))
    np.savetxt(os.path.join(save_path, 'result.csv'), result.params, delimiter=',')


def run_exploratory(data, save_path):
    """
    Runs different model types for exploratory analysis.

    :param data: the dataset
    :param save_path: save path for plots and parameters
    :return:
    """
    test_data = data[data['Region'].isin(['NL310_503', 'NL33C_340', 'NL33C_506', 'NL212_507'])]

    z_names = ['WVO', 'SchoolHolidayMiddle', 'SchoolHolidayNorth', 'SchoolHolidaySouth', '0-25_nbrpromos_index_201801',
               '25-50_nbrpromos_index_201801', '50-75_nbrpromos_index_201801']
    c_names = ['StringencyIndex']

    var_start = 1
    cov_start = 0

    def explore(params: list):
        """
        Explores a possible model specification.

        :param params: a list of parameters to be passed to SSMS, of the form [llt, alt, param_rest, cov_rest,
            tau_start] (see SSMS for detailed description of parameters).
        :return:
        """
        model = SSMS(test_data, group_name='Region', y_name='SalesGoodsEUR', z_names=z_names, c_names=c_names,
                     llt=params[0], alt=params[1], param_rest=params[2], cov_rest=params[3], tau_start=params[4],
                     var_start=var_start, cov_start=cov_start)
        result = model.fit(maxiter=1000, maxfun=1000000, disp=-1)

        # Compute predictions and MSE.
        y_pred = result.get_prediction(start=10, end=190)
        y1_pred = y_pred.predicted_mean[:, 0]
        mse_1 = np.mean(np.square(model.endog[10:, 0] - y1_pred))
        y2_pred = y_pred.predicted_mean[:, 1]
        mse_2 = np.mean(np.square(model.endog[10:, 1] - y2_pred))
        y3_pred = y_pred.predicted_mean[:, 2]
        mse_3 = np.mean(np.square(model.endog[10:, 2] - y3_pred))
        y4_pred = y_pred.predicted_mean[:, 3]
        mse_4 = np.mean(np.square(model.endog[10:, 3] - y4_pred))
        mse = (mse_1 + mse_2 + mse_3 + mse_4) / 4

        # Plot result.
        t = np.arange(11, 192)
        fig, axes = plt.subplots(2, 2)
        axes[0, 0].set_title('mse: {0}'.format(format(mse_1, '.4f')))
        axes[0, 0].set_xticks([])
        axes[0, 0].plot(t, model.endog[10:, 0], 'b')
        axes[0, 0].plot(t, y1_pred, 'r')
        axes[0, 1].set_title('mse: {0}'.format(format(mse_2, '.4f')))
        axes[0, 1].set_xticks([])
        axes[0, 1].plot(t, model.endog[10:, 1], 'b')
        axes[0, 1].plot(t, y2_pred, 'r')
        axes[1, 0].set_title('mse: {0}'.format(format(mse_3, '.4f')))
        axes[1, 0].plot(t, model.endog[10:, 2], 'b')
        axes[1, 0].plot(t, y3_pred, 'r')
        axes[1, 1].set_title('mse: {0}'.format(format(mse_4, '.4f')))
        axes[1, 1].plot(t, model.endog[10:, 3], 'b')
        axes[1, 1].plot(t, y4_pred, 'r')
        name = '_'.join([str(llt), param_rest, cov_rest, str(tau_start)])
        fig.suptitle('{0} (mse: {1})'.format(name, format(mse, '.4f')))
        plt.savefig(os.path.join(save_path, format(mse, '.4f') + '_' + name), dpi=300, format='png')
        plt.close('all')

        # Save parameters to CSV.
        np.savetxt(os.path.join(save_path, format(mse, '.4f') + '_' + name + '.csv'), result.params, delimiter=',')

    option_list = {'llt': [True, False], 'param_rest': ['F', 'RSI', 'RC'], 'cov_rest': ['F', 'RSC'],
                   'tau_start': [0.3, 0.5, 0.7, 0.9]}

    # First a big analysis. This yields a clear winner: RC-LLT (RSC).
    for llt in option_list['llt']:
        for param_rest in option_list['param_rest']:
            for cov_rest in option_list['cov_rest']:
                for tau_start in option_list['tau_start']:
                    explore([llt, False, param_rest, cov_rest, tau_start])

    # Try alternative specification (with a stationary trend nu). This does not seem to improve forecasts compared to
    # RC-LLT (RSC).
    for tau_start in option_list['tau_start']:
        explore([True, True, 'RC', 'RSC', tau_start])

    # Try more restrictive covariance specification. This again does not seem to improve forecasts compared to RC-LLT
    # (RSC).
    for tau_start in option_list['tau_start']:
        explore([True, False, 'RC', 'IDS', tau_start])


if __name__ == '__main__':
    main()
