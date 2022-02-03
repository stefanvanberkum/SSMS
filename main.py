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
- Parameter initialization fix
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
    test_data = data[data['Region'].isin(['NL310_503', 'NL33C_340', 'NL33C_506', 'NL125_507'])]

    z_names = ['WVO', 'SchoolHolidayMiddle', 'SchoolHolidayNorth', 'SchoolHolidaySouth', '0-25_nbrpromos_index_201801',
               '25-50_nbrpromos_index_201801', '50-75_nbrpromos_index_201801']
    c_names = ['StringencyIndex']
    model = SSMS(test_data, group_name='Region', y_name='SalesGoodsEUR', z_names=z_names, c_names=c_names,
                 cov_type='RSC')
    result = model.fit(maxiter=500, maxfun=1000000)
    # initial = model.fit(method='nm', maxiter=2000)
    # result = model.fit(initial.params, maxiter=500, maxfun=100000)
    print(result.summary())
    y_pred = result.get_prediction(start=10, end=190)
    y1_pred = y_pred.predicted_mean[:, 0]
    y2_pred = y_pred.predicted_mean[:, 1]
    y3_pred = y_pred.predicted_mean[:, 2]
    y4_pred = y_pred.predicted_mean[:, 3]
    t = np.arange(11, 192)
    plt.plot(t, model.endog[10:, 0], 'b')
    plt.plot(t, y1_pred, 'r')
    plt.show()
    plt.clf()
    plt.plot(t, model.endog[10:, 1], 'b')
    plt.plot(t, y2_pred, 'r')
    plt.show()
    plt.clf()
    plt.plot(t, model.endog[10:, 2], 'b')
    plt.plot(t, y3_pred, 'r')
    plt.show()
    plt.clf()
    plt.plot(t, model.endog[10:, 3], 'b')
    plt.plot(t, y4_pred, 'r')
    plt.show()
    plt.clf()
    exit(0)

    result.save(os.path.join(save_path, 'result.pickle'))
    np.savetxt(os.path.join(save_path, 'result.csv'), result.params, delimiter=',')

    end_time = time.time()
    print("Runtime:", (end_time - start_time), sep=' ')


if __name__ == '__main__':
    main()
