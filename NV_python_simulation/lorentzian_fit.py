import numpy as np
from scipy.signal import savgol_filter
from scipy.optimize import curve_fit
import scipy

def lorentzian_fit(x, y, dhyp, Nhyp, Npeak, *argv):
    # y = smooth(y, 0.003, 'loess')
    y = savgol_filter(y, 51, 3) # window size 51, polynomial order 3
    numvarargs = len(argv)
    optimset = ['TolFun', np.max(np.mean(y[:]) * 1e-6, 1e-15), 'TolX', np.max(np.mean(x[:])) * 1e-6, 1e-15]
    optargs = np.array([], [], '3c', optimset)
    for i in range(numvarargs):
        optargs[0 : numvarargs - 1] = argv

    options = np.array(['TolFun', 1e-11, 'TolX', 1e-4, 'MaxIter', 200])
    # opts = statset('nlinfit')
    # opts.DerivStep = 1e-15
    # opts.RobustWgtFun = '';
    # opts.MaxIter = 100000;
    # opts.TolFun = 1e-17;
    # opts.TolX = 1e-17;
    # opts.Display = 'final';

    # [params, residual, jacobian, covariance, MSE, ErrorModelInfo] = nlinfit(x, y, func, p0, options)
    # popt, pcov = curve_fit(func, x, y, p0=(1.0, 0.2))

