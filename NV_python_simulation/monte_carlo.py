import mock_diamond
import sensingMatrix
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler

import numpy as np

NUM_MEASURES = 50

freqs_to_try = 3
base_noise = 0.004
num_trials = 1

# Get diamond simulation
contrast = 0.002
data = mock_diamond.mock_diamond()
target_num_pks = len(data.peak_locs)
freqs = data.smp_freqs

window_start = min(data.peak_locs)-10
window_end = max(data.peak_locs)+10

x = 1

sensMtx = sensingMatrix.sensingMatrix(len(freqs), NUM_MEASURES, freqs_to_try=3)
projs = np.matmul(sensMtx, freqs)
projs = StandardScaler().fit_transform(projs)




