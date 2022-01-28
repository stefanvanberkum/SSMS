"""
This is the main environment for the state-space model.
"""

import os

from data_loader import load_data
from state_space import construct_arrays


def main():
    data_path = os.path.join(os.path.expanduser('~'), 'Documents', 'SSMS', 'data')

    data = load_data(data_path)
    z_names = ['WVO', 'TempAvg']
    c_names = ['StringencyIndex']
    endog, exog, c = construct_arrays(data, group_name='Region', y_name='SalesGoodsEUR', z_names=z_names,
                                      c_names=c_names)
    print()


if __name__ == '__main__':
    main()
