"""
This is the main environment for the state-space model.
"""

import os

from data_loader import load_data


def main():
    data_path = os.path.join(os.path.expanduser('~'), 'Documents', 'SSMS', 'data')

    data = load_data(data_path)
    print()


if __name__ == '__main__':
    main()
