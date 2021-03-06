# SSMS

State-space model for sales (SSMS).

## Installation

- Create a virtual environment using Python 3.9.
- Install the requirements by running the following command in your virtual environment: ```conda install --file requirements.txt``` (or ```pip install -r requirements.txt```).
- You can open and edit the code in any editor, we used the PyCharm IDE: https://www.jetbrains.com/pycharm/.

## What's included?

- data_loader.py: this module provides functions for loading and formatting the data.
- main.py: this is the main environment for the state-space model.
- state_space.py: this module provides the state-space model functionality.
- utils.py: this module provides utility methods.

## How to use?

- The main execution environment for this code is main.py, so this is where you'll want to be if you simply want to run
  the code.
- To run model selection, set ```model_select = True``` and run main.py.
- To fit a new model or use a saved pickle instance, set ```model_select = False```.
    - To fit a new model, set ```use_pickle = False```.
    - To use a saved pickle instance, set ```use_pickle = True```
