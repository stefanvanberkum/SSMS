"""
This is the main environment for the state-space model.
"""

import os
import time

import numpy as np

from data_loader import load_data
from state_space import SSMS


def main():
    data_path = os.path.join(os.path.expanduser('~'), 'Documents', 'SSMS', 'data')
    save_path = os.path.join(os.path.expanduser('~'), 'Documents', 'SSMS', 'results')

    start_time = time.time()

    data = load_data(data_path)
    test_data = data[data['Region'].isin(['NL310_503', 'NL329_502'])]

    z_names = ['WVO', 'TempAvg']
    c_names = ['StringencyIndex']
    model = SSMS(test_data, group_name='Region', y_name='SalesGoodsEUR', z_names=z_names, c_names=c_names,
                 cov_type='RSC')
    result = model.fit()

    result.save(os.path.join(save_path, 'result.pickle'))
    np.savetxt(os.path.join(save_path, 'result.csv'), result.params, delimiter=',')

    end_time = time.time()
    print("Runtime:", (end_time - start_time), sep=' ')


if __name__ == '__main__':
    main()
