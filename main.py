"""
This is the main environment for the state-space model.
"""

import os
import time

import numpy as np
from matplotlib import pyplot as plt

from data_loader import load_data
from state_space import SSMS
from utils import print_results

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

    # test_data = data[data['Region'].isin(['NL310_503', 'NL33C_340', 'NL33C_506', 'NL212_507'])]

    z_names = ['WVO', 'TG', 'SchoolHoliday', '0-25_nbrpromos_index_201801', '25-50_nbrpromos_index_201801',
               '50-75_nbrpromos_index_201801']
    c_names = ['StringencyIndexDiff']
    var_start = 1
    cov_start = 0
    cov_rests = ['RC', 'IDO', 'IDE']
    cov_type = 'oim'

    # model = SSMS(test_data, group_name='Region', y_name='SalesGoodsEUR', z_names=z_names, c_names=c_names,
    #             cov_rest='IDE', var_start=1, cov_start=0, fancy_start=True)
    # initial = model.fit(maxiter=1000, maxfun=1000000)
    # result = model.fit(initial.params, method='nm', maxiter=200000)
    # initial = model.fit(method='nm', maxiter=20000)
    # result = model.fit(initial.params, maxiter=1000, maxfun=100000)
    # result = model.fit(maxiter=1000, maxfun=1000000)
    # print(result.summary())
    # print_results(result, save_path, 'test')
    # exit(0)

    # Contaminated data.
    contaminated_data = load_data(data_path)

    # Clean data.
    clean_data = load_data(data_path, True, 6)

    # Extra clean data.
    extra_clean_data = load_data(data_path, True, 4)

    data_options = {'contaminated': contaminated_data, 'clean': clean_data, 'extra clean': extra_clean_data}

    for data_type in data_options:
        data = data_options[data_type]
        for cov_rest in cov_rests:
            print("Running " + cov_rest + " with " + data_type + " data...")
            split_time = time.time()

            model = SSMS(data, group_name='Region', y_name='SalesGoodsEUR', z_names=z_names, c_names=c_names,
                         cov_rest=cov_rest, var_start=var_start, cov_start=cov_start, fancy_start=True)
            result = model.fit(maxiter=10000, maxfun=10000000, cov_type=cov_type, disp=-1)

            # Make sure to plot the same regions each time.
            grouped = data.groupby('Region', sort=False)
            group_names = [name for name, group in grouped]
            region_1 = group_names.index('NL310_503')
            region_2 = group_names.index('NL33C_340')
            region_3 = group_names.index('NL33C_506')
            region_4 = group_names.index('NL212_507')

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
            name = '_'.join([cov_rest, data_type])
            fig.suptitle('{0} (mse: {1})'.format(name, format(mse, '.4f')))
            plt.savefig(os.path.join(save_path, name), dpi=300, format='png')
            plt.close('all')

            print_results(result, save_path, name)
            result.save(os.path.join(save_path, name + '.pickle'))

            print("Done!")
            end_time = time.time()
            print("Runtime:", (end_time - split_time), sep=' ')
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
    c_names = ['StringencyIndexDiff']

    def explore(params: list):
        """
        Explores a possible model specification.

        :param params: a list of parameters to be tested, of the form [cov_rest, var_start, cov_start]
        :return:
        """
        model = SSMS(data, group_name='Region', y_name='SalesGoodsEUR', z_names=z_names, c_names=c_names,
                     cov_rest=params[0], var_start=1, cov_start=params[1], fancy_start=True)
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
        name = '_'.join([params[0], str(params[1])])
        fig.suptitle('{0} (mse: {1})'.format(name, format(mse, '.4f')))
        plt.savefig(os.path.join(save_path, format(mse, '.4f') + '_' + name), dpi=300, format='png')
        plt.close('all')

        # Save parameters to CSV.
        np.savetxt(os.path.join(save_path, format(mse, '.4f') + '_' + name + '.csv'), result.params, delimiter=',')

    # Run exploratory analysis.
    explore(['RC', 0])
    explore(['IDO', 0])
    explore(['IDE', 0])


if __name__ == '__main__':
    main()
