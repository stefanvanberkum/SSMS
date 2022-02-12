"""
This is the main environment for the state-space model.
"""

import os
import time

import numpy as np
from matplotlib import pyplot as plt

from data_loader import load_data
from state_space import SSMS, SSMS_alt
from utils import print_results

"""
TODO:
- State printing
- Forecasting (train/test)
"""


def main():
    data_path = os.path.join(os.path.expanduser('~'), 'Documents', 'SSMS', 'data')
    save_path = os.path.join(os.path.expanduser('~'), 'Documents', 'SSMS', 'results')

    start_time = time.time()

    data = load_data(data_path)
    train_data = data[:63 * 153]
    test_data = data[63 * 153:]
    # test_data = data[data['Region'].isin(['NL310_503', 'NL33C_340', 'NL33C_506', 'NL212_507'])]
    # test_data = data[data['Region'].isin(
    #    ['NL310_503', 'NL33C_340', 'NL33C_506', 'NL212_507', 'NL414_340', 'NL328_501', 'NL333_505', 'NL230_508',
    #     'NL414_511', 'NL332_505'])]

    z_names = ['WVO', 'TG', 'SchoolHoliday', '0-25_nbrpromos_index_201801', '25-50_nbrpromos_index_201801',
               '50-75_nbrpromos_index_201801']
    d_names = ['StringencyIndex']
    c_names = ['StringencyIndexDiff']
    var_start = 1
    cov_start = 0
    cov_rests = ['IDE']
    cov_group = 'nuts3_code'
    cov_type = 'oim'
    alts = [False]

    # model = SSMS(train_data, group_name='Region', y_name='SalesGoodsEUR', z_names=z_names, c_names=c_names,
    #             cov_rest='IDE')
    # initial = model.fit(maxiter=1000, maxfun=1000000)
    # result = model.fit(initial.params, method='nm', maxiter=200000)
    # initial = model.fit(method='nm', maxiter=20000)
    # result = model.fit(initial.params, maxiter=1000, maxfun=100000)
    # result = model.fit(maxiter=1000, maxfun=1000000, cov_type='oim')
    # print(result.summary())
    # print_results(result, save_path, 'test')
    # exit(0)

    # run_exploratory(train_data, save_path)
    select_variables(train_data)

    end_time = time.time()
    print("Total runtime:", (end_time - start_time), sep=' ')


def run_exploratory(data, save_path):
    """
    Runs different model types for exploratory analysis.

    :param data: the dataset
    :param save_path: save path for plots and parameters
    :return:
    """

    z_names = ['WVO', 'TG', 'SchoolHoliday', '0-25_nbrpromos_index_201801', '25-50_nbrpromos_index_201801',
               '50-75_nbrpromos_index_201801']
    d_names = ['StringencyIndex']
    c_names = ['StringencyIndexDiff']
    var_start = 1
    cov_rests = ['GC', 'IDE']
    cov_group = 'nuts3_code'
    cov_type = 'oim'
    alts = [True, False]

    def explore(params: list):
        """
        Explores a possible model specification.

        :param params: a list of parameters to be tested, of the form [cov_rest, alt, cov_start]
        :return:
        """
        cov_rest = params[0]
        alt = params[1]
        cov_start = params[2]
        name = '_'.join([cov_rest, str(alt), str(cov_start)])
        print("Running " + name + "...")
        split_time = time.time()
        if alt:
            model = SSMS_alt(data, group_name='Region', y_name='SalesGoodsEUR', z_names=z_names, d_names=d_names,
                             c_names=c_names, cov_rest=cov_rest, cov_group=cov_group, var_start=var_start,
                             cov_start=cov_start, fancy_start=True)
            result = model.fit(maxiter=10000, maxfun=10000000, cov_type=cov_type)
        else:
            model = SSMS(data, group_name='Region', y_name='SalesGoodsEUR', z_names=z_names, c_names=c_names,
                         cov_rest=cov_rest, cov_group=cov_group, var_start=var_start, cov_start=cov_start,
                         fancy_start=True)
            result = model.fit(maxiter=10000, maxfun=10000000, cov_type=cov_type)

        # Make sure to plot the same regions each time.
        grouped = data.groupby('Region', sort=False)
        group_names = [name for name, group in grouped]
        region_1 = group_names.index('NL310_503')
        region_2 = group_names.index('NL33C_340')
        region_3 = group_names.index('NL33C_506')
        region_4 = group_names.index('NL212_507')

        # Compute predictions and MSE.
        y_pred = result.get_prediction(start=10, end=190)
        y1_pred = y_pred.predicted_mean[:, region_1]
        y2_pred = y_pred.predicted_mean[:, region_2]
        y3_pred = y_pred.predicted_mean[:, region_3]
        y4_pred = y_pred.predicted_mean[:, region_4]
        mse_1 = np.mean(np.square(model.endog[10:, region_1] - y1_pred))
        mse_2 = np.mean(np.square(model.endog[10:, region_2] - y2_pred))
        mse_3 = np.mean(np.square(model.endog[10:, region_3] - y3_pred))
        mse_4 = np.mean(np.square(model.endog[10:, region_4] - y4_pred))

        mses = np.zeros(len(group_names))
        for region in range(len(group_names)):
            mses[region] = np.mean(np.square(model.endog[10:, region] - y_pred.predicted_mean[:, region]))
        mse = np.mean(mses)

        # Plot preliminary result.
        t = np.arange(11, 192)
        fig, axes = plt.subplots(2, 2)
        axes[0, 0].set_title('mse: {0}'.format(format(mse_1, '.4f')))
        axes[0, 0].set_xticks([])
        axes[0, 0].plot(t, model.endog[10:, region_1], 'b')
        axes[0, 0].plot(t, y1_pred, 'r')
        axes[0, 1].set_title('mse: {0}'.format(format(mse_2, '.4f')))
        axes[0, 1].set_xticks([])
        axes[0, 1].plot(t, model.endog[10:, region_2], 'b')
        axes[0, 1].plot(t, y2_pred, 'r')
        axes[1, 0].set_title('mse: {0}'.format(format(mse_3, '.4f')))
        axes[1, 0].plot(t, model.endog[10:, region_3], 'b')
        axes[1, 0].plot(t, y3_pred, 'r')
        axes[1, 1].set_title('mse: {0}'.format(format(mse_4, '.4f')))
        axes[1, 1].plot(t, model.endog[10:, region_4], 'b')
        axes[1, 1].plot(t, y4_pred, 'r')
        fig.suptitle('{0} (mse: {1})'.format(name, format(mse, '.4f')))
        plt.savefig(os.path.join(save_path, name), dpi=300, format='png')
        plt.close('all')

        # Save result.
        print_results(result, save_path, name)
        result.save(os.path.join(save_path, name + '.pickle'))

        print("Done!")
        end_time = time.time()
        print("Runtime:", (end_time - split_time), sep=' ')

    # Run exploratory analysis.
    for cov_rest in cov_rests:
        for alt in alts:
            if cov_rest == 'GC':
                cov_starts = [0, 0.001, 0.01]
            else:
                cov_starts = [0]
            for cov_start in cov_starts:
                explore([cov_rest, alt, cov_start])


def select_variables(data):
    """
    Runs the IDE-type model selection.

    :param data: the dataset
    :return:
    """

    z_names = ['WVO', 'TG', 'SchoolHoliday', '0-25_nbrpromos_index_201801', '25-50_nbrpromos_index_201801',
               '50-75_nbrpromos_index_201801']
    c_names = ['StringencyIndexDiff']
    cov_rest = 'IDE'
    cov_type = 'oim'

    def select(params: list):
        """
        Runs a possible model specification.

        :param params: a list of the parameters to be used
        :return: the AIC
        """
        model_name = ', '.join(params)
        print("Running with: " + model_name + "...")
        split_time = time.time()
        model = SSMS(data, group_name='Region', y_name='SalesGoodsEUR', z_names=params, c_names=c_names,
                     cov_rest=cov_rest)
        result = model.fit(maxiter=10000, maxfun=10000000, cov_type=cov_type, disp=-1)

        print("Done!")
        end_time = time.time()
        print("Runtime:", (end_time - split_time), sep=' ')
        return result.aic

    # Run exploratory analysis.
    best_aic = select(z_names)
    while len(z_names) > 1:
        best_rest = ''
        for z_name in z_names:
            z_copy = z_names.copy()
            z_copy.remove(z_name)
            aic = select(z_copy)

            if aic < best_aic:
                best_aic = aic
                best_rest = z_name
        if best_rest != '':
            print("Best restriction: " + best_rest)
            z_names.remove(best_rest)
        else:
            break
    name = ', '.join(z_names)
    print("Best parameter set: " + name)


if __name__ == '__main__':
    main()
