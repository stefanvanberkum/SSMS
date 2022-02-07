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
- Constrain stationary?
- Parameter printing (standard errors + p-values?)
- State printing
- IDS + init
- GSC + init
- Forecasting (train/test)
"""


def main():
    data_path = os.path.join(os.path.expanduser('~'), 'Documents', 'SSMS', 'data')
    save_path = os.path.join(os.path.expanduser('~'), 'Documents', 'SSMS', 'results')

    start_time = time.time()

    data = load_data(data_path)
    test_data = data[data['Region'].isin(['NL310_503', 'NL33C_340', 'NL33C_506', 'NL212_507'])]

    z_names = ['WVO', 'SchoolHolidayMiddle', 'SchoolHolidayNorth', 'SchoolHolidaySouth', '0-25_nbrpromos_index_201801',
               '25-50_nbrpromos_index_201801', '50-75_nbrpromos_index_201801']
    c_names = ['StringencyIndex']

    option_list = {'llt': [True, False], 'param_rest': ['F', 'RSI', 'RC'], 'cov_rest': ['F', 'RSC'],
                   'tau_start': [0.3, 0.5, 0.7, 0.9]}
    var_start = 1
    cov_start = 0

    # Run with tau for llt (change name).
    for tau_start in option_list['tau_start']:
        model = SSMS(test_data, group_name='Region', y_name='SalesGoodsEUR', z_names=z_names, c_names=c_names, llt=True,
                     alt=True, param_rest='RC', cov_rest='RSC', tau_start=tau_start, var_start=var_start,
                     cov_start=cov_start)
        result = model.fit(maxiter=1000, maxfun=1000000, disp=-1)
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
        name = '_'.join(['True', 'RC', 'RSC', str(tau_start), 'alt'])
        fig.suptitle('{0} (mse: {1})'.format(name, format(mse, '.4f')))
        plt.savefig(os.path.join(save_path, format(mse, '.4f') + '_' + name), dpi=300, format='png')
        np.savetxt(os.path.join(save_path, format(mse, '.4f') + '_' + name + '.csv'), result.params, delimiter=',')
        plt.close('all')
    exit(0)

    for llt in option_list['llt']:
        for param_rest in option_list['param_rest']:
            for cov_rest in option_list['cov_rest']:
                for tau_start in option_list['tau_start']:
                    model = SSMS(test_data, group_name='Region', y_name='SalesGoodsEUR', z_names=z_names,
                                 c_names=c_names, llt=llt, alt=False, param_rest=param_rest, cov_rest=cov_rest,
                                 tau_start=tau_start, var_start=var_start, cov_start=cov_start)
                    # initial = model.fit(maxiter=500, maxfun=1000000)
                    # result = model.fit(initial.params, method='nm', maxiter=20000)
                    # initial = model.fit(method='nm', maxiter=2000)
                    # result = model.fit(initial.params, maxiter=500, maxfun=100000)
                    result = model.fit(maxiter=1000, maxfun=1000000, disp=-1)
                    # print(result.summary())
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
                    name = '_'.join([str(llt), param_rest, cov_rest, str(tau_start)])
                    fig.suptitle('{0} (mse: {1})'.format(name, format(mse, '.4f')))
                    plt.savefig(os.path.join(save_path, format(mse, '.4f') + '_' + name), dpi=300, format='png')
                    np.savetxt(os.path.join(save_path, format(mse, '.4f') + '_' + name + '.csv'), result.params,
                               delimiter=',')
                    plt.close('all')
    exit(0)

    result.save(os.path.join(save_path, 'result.pickle'))
    np.savetxt(os.path.join(save_path, 'result.csv'), result.params, delimiter=',')

    end_time = time.time()
    print("Runtime:", (end_time - start_time), sep=' ')


if __name__ == '__main__':
    main()
