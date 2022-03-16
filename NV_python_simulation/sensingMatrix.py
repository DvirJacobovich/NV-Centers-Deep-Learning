import numpy as np

def sensingMatrix(full_window_size, num_of_measures, freqs_to_try):
    sens_mtx = np.zeros((num_of_measures, full_window_size))
    for i in range(len(sens_mtx)):
        curr_freqs = np.random.choice(full_window_size, freqs_to_try, replace=False)
        sens_mtx[i][curr_freqs] = 1

    return sens_mtx
#
# full_window_size = 200
# num_of_measures = 50
# freqs_to_try = 3
#
# sensingMatrix(full_window_size, num_of_measures, freqs_to_try)