"""
This is the main environment for the state-space model.
"""

import os
import time

import numpy as np
from statsmodels.iolib.smpickle import load_pickle

from data_loader import load_data
from state_space import SSMS
from utils import forecast_error, plot_states, prepare_forecast


def main():
    model_select = False
    use_pickle = False  # Make sure that the settings below match those of your pickle instance.

    data_path = os.path.join(os.path.expanduser('~'), 'Documents', 'SSMS', 'data')
    save_path = os.path.join(os.path.expanduser('~'), 'Documents', 'SSMS', 'results')

    start_time = time.time()
    data = load_data(data_path, save_path)
    train_data = data[:63 * 153]
    test_data = data[63 * 153:]
    # regions = ['NL310_503', 'NL33C_340', 'NL33C_506', 'NL212_507', 'NL414_340', 'NL328_501', 'NL333_505', 'NL230_508',
    #            'NL414_511', 'NL332_505']
    # regions = ['NL310_503', 'NL33C_340', 'NL33C_506', 'NL212_507', 'NL414_340', 'NL328_501', 'NL333_505', 'NL230_508']
    # data = data[data['Region'].isin(regions)]
    # train_data = train_data[train_data['Region'].isin(regions)]
    # test_data = test_data[test_data['Region'].isin(regions)]

    # z_names = ['WVO', 'TG', 'SchoolHoliday', '0-25_nbrpromos_index_201801', '25-50_nbrpromos_index_201801',
    #           '50-75_nbrpromos_index_201801', 'StringencyIndex']
    z_names = ['WVO', 'StringencyIndex']
    d_names = ['StringencyIndex']
    c_names = ['StringencyIndexDiff']
    var_start = 1
    cov_start = 0
    cov_rests = ['IDE']
    cov_group = 'nuts3_code'
    cov_type = 'oim'
    alts = [False]

    if model_select:
        model_selection(train_data)
    elif use_pickle:
        result = load_pickle(os.path.join(save_path, 'result.pickle'))
        result.model.group_name = 'Region'
        model = result.model
    else:
        model = SSMS(train_data, group_name='Region', y_name='SalesGoodsEUR', z_names=z_names, cov_rest='IDE')
        result = model.fit(maxiter=100000, maxfun=100000000, cov_type='oim')
        result.save(os.path.join(save_path, 'result.pickle'))

    if not model_select:
        extended_filtered_result = prepare_forecast(result, data)
        smoothed_result = model.smooth(result.params, return_ssm=1)

        group_names = result.model.group_names
        plot_states(result, result, group_names, z_names, save_path)
        plot_states(result, smoothed_result, group_names, z_names, save_path)
        forecast_error(extended_filtered_result, group_names, save_path, 153, 190, 1,
                       'one_step_ahead_forecast')  # forecast_error(result, group_names, save_path, 10, 152, 0,
        # 'in_sample_prediction', n_plots=8)

    end_time = time.time()
    print("Total runtime:", (end_time - start_time), sep=' ')


def model_selection(data):
    """
    Runs model selection.

    :param data: the dataset
    :return:
    """

    z_names = ['WVO', 'TG', 'SchoolHoliday', '0-25_nbrpromos_index_201801', '25-50_nbrpromos_index_201801',
               '50-75_nbrpromos_index_201801', 'StringencyIndex']
    cov_rests = ['GC', 'IDE']
    cov_group = 'nuts3_code'
    cov_type = 'oim'

    def model_select(params: list):
        """
        Runs a possible model specification.

        :param params: a list of form [cov_rest, cov_start]
        :return: the AIC
        """
        model_name = '_'.join([params[0], str(params[1])])
        print("Running " + model_name + "...")
        split_time = time.time()
        model = SSMS(data, group_name='Region', y_name='SalesGoodsEUR', z_names=z_names, cov_rest=params[0],
                     cov_start=params[1], cov_group=cov_group)
        result = model.fit(maxiter=10000, maxfun=10000000, cov_type=cov_type, disp=-1)

        print("Done!")
        end_time = time.time()
        print("Runtime:", (end_time - split_time), sep=' ')
        return result.aic

    def param_select(params: list, model_type: list):
        """
        Runs a possible model specification.

        :param params: a list of the parameters to be used
        :param model_type: a list of form [cov_rest, cov_start]
        :return: the AIC
        """
        model_name = ', '.join(params)
        print("Running with: " + model_name + "...")
        split_time = time.time()
        model = SSMS(data, group_name='Region', y_name='SalesGoodsEUR', z_names=params, cov_rest=model_type[0],
                     cov_start=model_type[1], cov_group=cov_group)
        result = model.fit(maxiter=10000, maxfun=10000000, cov_type=cov_type, disp=-1)

        print("Done!")
        end_time = time.time()
        print("Runtime:", (end_time - split_time), sep=' ')
        return result.aic

    best_aic = np.inf
    best_type = [None, None]
    for cov_rest in cov_rests:
        if cov_rest == 'GC':
            cov_starts = [0, 0.001, 0.01]
        else:
            cov_starts = [0]
        for cov_start in cov_starts:
            aic = model_select([cov_rest, cov_start])
            if aic < best_aic:
                best_aic = aic
                best_type[0] = cov_rest
                best_type[1] = cov_start
    name = '_'.join([best_type[0], str(best_type[1])])
    print("Best type: " + name)

    # Run exploratory analysis.
    while len(z_names) > 1:
        best_rest = ''
        for z_name in z_names:
            z_copy = z_names.copy()
            z_copy.remove(z_name)
            aic = param_select(z_copy, best_type)

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
