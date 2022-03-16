import numpy as np

def getFitVals(params, conf, parameter_name):
    # After 'lorentzian_fit' get a value from the fit and its uncertainty.
    if parameter_name == 'Peak':
        idx = 1

    elif parameter_name == 'Width':
        idx = 2

    elif parameter_name == 'Amplitude':
        idx = 3

    elif parameter_name == 'Offset':
        vals = params[-1]
        unc = np.abs(conf[-1, 2] - conf[-1, 1])
        return vals, unc

    if len(conf) == 0:
        unc = []

    else:
        unc = 0.5 * np.abs(conf[idx: 3: len(conf) - 1, 2] - conf[idx:3:(len(conf) - 1), 1])

    vals = params[idx: 3: len(params) - 1]
    return vals, unc
