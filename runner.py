import os
import scipy.io as sio

path_parent = os.path.dirname(os.getcwd())
rootdir = os.path.join(path_parent, 'Data')


class MeasureStruct:
    def __init__(self, meas_num):
        # real experiment data
        samp_strc = sio.loadmat(os.path.join(rootdir, 'ARF1_3freq_' + str(meas_num) + '.mat'))['exp']
        self.samp_freqs = samp_strc['smp_freqs']
        self.ref_measurement = samp_strc['ref_measurements']
        self.sig_measurement = samp_strc['sig_measurements']


if __name__ == '__main__':
        m = MeasureStruct(1)
        x = 1

