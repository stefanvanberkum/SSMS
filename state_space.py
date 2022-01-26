"""
This module provides the state-space model functionality.
"""

import statsmodels.api as sm


# A fully extended model, extends the basic Statsmodels state-space class.
class SSMS(sm.tsa.statespace.MLEModel):
    def __init__(self, endog):
        # Intialize the state-space model.
        super(SSMS, self).__init__(endog, k_states=2)
