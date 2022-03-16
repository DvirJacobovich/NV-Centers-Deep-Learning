import numpy as np
import scipy
from scipy.signal import savgol_filter


def getFitGuess(t, data, contrast):
    # Guess the parameters of the peaks in data based on variable 't'
    smooth_data = savgol_filter(data, 51, 3)  # window size 51, polynomial order 3
    locs, non = scipy.signal.find_peaks(smooth_data)
    num_pks = len(locs)
    initial_guess = np.zeros(1, num_pks + 1)
    for i in range(num_pks):
        initial_guess[2 * i - (2 - i)] = t[locs[i]]  # peak location
        initial_guess[2 * i - (2 - i) + 1] = 7  # width
        initial_guess[2 * i - (2 - i) + 2] = -np.abs(smooth_data[locs[i]] - np.min(smooth_data))

    initial_guess[num_pks * 3 + 1] = np.min(smooth_data)  # offset
    return smooth_data, num_pks, initial_guess,locs


