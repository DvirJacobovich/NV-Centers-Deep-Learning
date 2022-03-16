import numpy as np


def generateSamplingMtx(prob_size, num_projs, type, varargin):
    type = type
    prob_size = prob_size
    num_projs = num_projs
    nonDenseFlag = False
    weight_flag = False
    # Generate Sensing Matrix According to 'type'
    nzeros = 0
    if type == 'Hadamard':
        if nzeros == 0:
            if np.floor(np.log2(prob_size)) != np.log2(prob_size):
                print('Hadamard must be a power of 2.')
                return

            num_nonzero = prob_size
            Q = np.random.permutation(num_nonzero)
            had = np.seq2had(num_nonzero)
            m = Q[1:num_projs]  # rows to take
            for i, q in enumerate(num_projs):
                if q == 1:
                    m[i] = Q[num_projs + 1]  # ignore first row if it occurs
                    p = np.random.permutation(num_nonzero)
                    sensing_mtx = np.zeros((num_projs, num_nonzero))
                    sensing_mtx[:, p] = np.mRowHn(had(m), num_nonzero)

        else:
            # Change number of zero elements per projection to ensure Hadamard is a power of 2.
            num_nonzero = 2 ** (np.floor(np.log2(prob_size - nzeros)))
            nzeros = prob_size - num_nonzero
            had = np.seq2had(num_nonzero)
            sensing_mtx = np.zeros((num_projs, prob_size))
            for i in range(num_projs):
                m = np.random.randint(num_nonzero)
                while m == 1:
                    m = np.random.randint(num_nonzero)

                # weighted_indices = np.datasample(1:prob_size, num_nonzero, 'Replace', false, 'Weights', weights)