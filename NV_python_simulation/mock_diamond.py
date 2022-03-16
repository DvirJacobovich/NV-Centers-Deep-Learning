import numpy as np
from getFitGuess import getFitGuess
from getFitVals import getFitVals
from lorentzian_fit import lorentzian_fit
from matplotlib import pyplot as plt


class mock_diamond:

    def __init__(self, *args):
        self.num_pts = 200
        # Magnetic Field Magnitude and Direction(in lab frame)

        self.B_mag = ((125 - 45) * np.random.rand(1) + 45)[0]  # Gauss
        # self.B_mag = 90

        self.B_theta = 30  # Degrees
        self.B_phi = 60  # Degrees

        # self.B_mag = 49.2  # Gauss
        # self.B_theta = 30  # Degrees
        # self.B_phi = 70  # Degrees

        # Noise Simulation
        self.add_noise = False
        self.sigma = 0.002  # expressed as a fraction of the mean

        # Derived Quantities
        self.peak_locs = np.array([])  # will contain peak locations based on Magnetic Field
        self.smp_freqs = np.array([])  # where we measure in our simulation
        self.target = np.array([])  # lineshape we want to reconstruct

        # model of what we actually measure
        self.sig = np.array([])
        self.ref = np.array([])

        # for combining 'sig' and 'ref' into a single array for compatibility with adaptive_reconstruction class
        self.signal = np.zeros((1, 2))

        # Orientations of Diamond (in lab frame) assumed fixed
        self.diamond_unit_vecs = (1 / np.sqrt(3)) * np.array([[1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]])
        # Target Peak Amplitude
        self.peak_amp = 0.01
        # Lorentzian Properties
        self.center_freq = 2875  # MHz
        self.width = 10  # MHz
        self.kernel = self.lorentzian_kernel('N')
        self.move_center = False
        self.center_offset = 0
        self.raster_flag = False
        self.chk_noise = np.array([])

        if len(args) > 0:
            self.smp_freqs, self.sig, self.ref = args
            self.num_pts = len(self.smp_freqs)

            if len(args) == 3:
                self.raster_flag = False
                self.target = (self.ref - self.sig) / np.mean(self.ref)

            elif len(args) == 4:
                self.raster_flag = True
                self.sigma = args[3]
                self.target = self.ref - self.sig

            non, curr_num_pks, curr_guess = getFitGuess(self.smp_freqs, self.target, 0.5 * np.max(self.target))
            if curr_num_pks != 8:  # too many peaks
                exit(-1)

            non2, curr_params, non1, non, curr_conf = lorentzian_fit(self.smp_freqs.T, self.target.T,
                                                                     2, 2, curr_num_pks, curr_guess)

            self.peak_locs, non = getFitVals(curr_params, curr_conf, 'Peak')
            self.peak_locs = self.peak_locs.T

        else:
            # mock_experiment with no arguments assumes default values and creates a mock lineshape for running
            # a Compressed Sensing simulation
            B_vec = self.pol2xyz()
            B_projs = np.sum(self.diamond_unit_vecs * B_vec, 1)
            detunings = 2.8 * B_projs  # distance from center in MHz, assume 2.8 MHz/G
            full_peak_window = np.max(self.center_freq + detunings) - np.min(self.center_freq - detunings)
            if self.move_center:
                max_to_move = 20
                self.center_offset = max_to_move * (2 * np.random.rand() - 1)
                self.shifts = 0.5 * (2 * np.random.rand(8, 1) - 1)

            else:
                self.shifts, self.center_offset = 0, 0

            self.peak_locs = (
                        np.sort(np.array([self.center_freq - detunings, self.center_freq + detunings]).reshape(1, 8)) \
                        + self.center_offset + self.shifts).T

            # Calculate Number of Points in Simulation
            # aim for a frequency spacing of ~2 MHz and that the peaks take
            # up about 60% of the full window
            fraction_peak_window = 0.6
            df = 2  # MHz
            full_peak_window = max(self.peak_locs) - min(self.peak_locs)

            # if full_peak_window < 200:
            #     full_peak_window = 200

            pad_width = ((1 - fraction_peak_window) / (2 * fraction_peak_window)) * full_peak_window
            MHz_full_window = (0.5 * full_peak_window + pad_width) + (0.5 * full_peak_window + pad_width)

            # self.num_pts = np.ceil((full_peak_window + 2 * pad_width) / df)
            self.num_pts = 200
            points_per_MHzs = self.num_pts / MHz_full_window
            self.smp_freqs = self.center_freq + np.linspace(-(0.5 * full_peak_window + pad_width),
                                                            (0.5 * full_peak_window + pad_width), int(self.num_pts))

            self.target = self.getLineShape(self.smp_freqs)
            x = 1

    def getLineShape(self, freqs):
        """
         Gets the lineshape for the diamond at frequencies 'freqs'
        :param freqs: freqs
        :return:
        """
        amp = self.peak_amp / (2 / (np.pi * self.width))
        L = lambda detuning: amp * self.kernel(freqs, self.width, detuning)
        lineshape = np.squeeze(np.zeros((len(freqs), 1))).T
        for i in range(len(np.squeeze(self.peak_locs))):
            lineshape = np.squeeze(lineshape) + L(np.asscalar(self.peak_locs[i])).T

        return lineshape.T

    def lorentzian_kernel(self, args):
        """
        Different coices of Lorentzian kernel
        :return: Output is a function whose form is a 1-D Lorentzian with
        properties determined by args:
        'ZMN' for Zero-Mean Normalized Lorentzian (= standard_cauchy distribution).
        'ZMU' for Zero-Mean Unormalized Lorentzian.
        'N' for Normalized.
        'U' for Unormalized.
        """
        if args == 'ZMN':
            return np.random.standard_cauchy

        if args == 'ZMU':
            return lambda x, w, x0=0: 1 / (
                    1 + (2 * (x - x0) / w) ** 2)  # - np.mean(1 / (1 + (2 * (x - x0) / w)))

        if args == 'N':
            # return lambda x, w, x0=0: np.divide((2 / np.pi * w), (1 + np.power((2 * (x - x0) / w), 2)))
            return lambda x, w, x0: (2 / (np.pi * w)) * (1 / (1 + (2 * ((x - x0) / w)) ** 2))

        if args == 'U':
            return lambda x, w, x0: 1 / (1 + (2 * (x - x0) / w)) ** 2

    def pol2xyz(self, *args):
        """
          pol2xyz Converts vector in spherical to rctangular coordinates Assumes angles are given in degrees.
        :return:
        """

        rad2deg = lambda angle: (np.sin(angle * np.pi / 180), np.cos(angle * np.pi / 180))

        if len(args) == 2:
            theta, phi = args
            return [rad2deg(theta)[0] * rad2deg(phi)[1], rad2deg(theta)[0] * rad2deg(phi)[0], rad2deg(phi)[1]]

        else:
            if len(args) == 3:
                mag, theta, phi = args

            else:  # if len(args) == 0
                mag = self.B_mag
                theta = self.B_theta
                phi = self.B_phi

            return mag * np.array(
                [rad2deg(theta)[0] * rad2deg(phi)[1], rad2deg(theta)[0] * rad2deg(phi)[0], rad2deg(theta)[1]])

    def getRaster(self, *args):
        """
        %Get a Simulated Raster Scan with or without noise
        """
        freqs = self.smp_freqs if len(args) == 0 else args[0]
        if self.raster_flag:
            return self.target

        else:
            npts = len(freqs)
            raster_lineshape = self.getLineShape(freqs)

            if self.add_noise:
                sim_ref = np.ones(npts, 1) + self.sigma * np.random.randn(npts, 1)
                sim_sig = sim_ref - raster_lineshape + self.sigma * np.random.randn(npts, 1)

            else:
                sim_ref = np.ones(npts, 1)
                sim_sig = sim_ref - self.target

            return (sim_ref - sim_sig) / np.mean(sim_ref)

    def getMeasurement(self, sample_freqs):
        """
        Get a Simulated Raster Scan with or without noise
        :param sample_freqs
        """
        indices = np.where(self.smp_freqs == sample_freqs)
        if self.add_noise:
            if self.raster_flag:
                ref_noise = self.sigma * np.random.randn() * np.mean(self.ref)
                self.signal[1, 2] = np.mean(self.ref[indices]) + ref_noise
                self.signal[1, 1] = ref_noise + np.sum(self.sig[indices]) + self.sigma * np.random.randn() \
                                    * np.mean(self.ref)

                np.append(self.chk_noise, ref_noise, axis=0)

            else:
                self.signal[1, 2] = 1 + self.sigma * np.random.randn()
                self.signal[1, 1] = self.signal[1, 2] + self.sigma * np.random.randn() - np.sum(self.target[indices])

        else:
            self.signal[1, 2] = 1
            self.signal[1, 2] = 1 - np.sum(self.target[indices])
