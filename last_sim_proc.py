import os
import time
import scipy.io as sio
import torch
import torch.nn as nn
import NV_real_data_model
import torch.optim as optim
import simulated_NV_DataSet
import NV_simulateed_data_model
import NV_normalize_data as nr
import mat73
from mat4py import loadmat
import scipy.io
import numpy as np
import h5py

from matplotlib import pyplot as plt
from torchvision import transforms
import torch.nn.functional as f

##
# Simlualated data: inputs: in CS staruct: smp_mtx (100x200), basis_mtx (200x200) (need to multiply these two to get the
# 100x200 Theta (maybe not need the basis_max), sig_measurements(100, 1), and at last inside CS struct we have 'sim' struct
# that conclude the physical parameters such that: B_mag, B_phi, B_theta (can be represented by 3 matrices). we need to get
# from galya fixed dimensions and physical parameters range so that in these dimensions and applying this ranges we
# have 8 lorenzian peaks.
#
# As for the lables we have 'target' (200, 1) in the 'sim' struct inside 'CS'.

sim_data_loc = r'C:\Users\owner\Desktop\Dvir\NV-centers Compressed learning\NV-centers CompressedLearning-20211121T154621Z-001\compressedLearning-main\matlab_code\COmpSENS'

NUN_REAL_SAMPS = 99
NUM_SIM_SAMPS = 5

TANH = nn.Tanh()

use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
torch.backends.cudnn.benchmark = True

path_parent = os.path.dirname(os.getcwd())
rootdir = os.path.join(path_parent, 'matlab_code\COmpSENS')

# inputs tensors:
# basis_mtx_tensors = torch.empty([NUM_SIM_SAMPS, 200, 200])

sig_measurements_tensors = torch.ones([NUM_SIM_SAMPS, 1, 100])
smp_mtx_tensors = torch.ones([NUM_SIM_SAMPS, 100, 200])
physical_params_tensors = torch.ones([NUM_SIM_SAMPS, 3, 100, 1])

# lables tensors:
target_tensors = torch.ones([NUM_SIM_SAMPS, 200, 1])

for smp in range(NUM_SIM_SAMPS):
    file_name = os.path.join(rootdir, 'CS_' + str(smp) + '_.mat')
    # inputs:
    CS_struct = sio.loadmat(os.path.join(rootdir, 'CS_' + str(smp) + '_.mat'))['cs']

    sig_measurements = torch.from_numpy(CS_struct['sig_measurements'][0][0].T)
    sig_measurements_tensors[smp, :, :] = sig_measurements

    smp_mtx = torch.from_numpy(CS_struct['smp_mtx'][0][0])
    smp_mtx_tensors[smp, :, :] = smp_mtx

    # basis_mtx = torch.from_numpy(CS_struct['basis_mtx'][0][0])

    # physical inputs:
    physical_data_struct = sio.loadmat(os.path.join(rootdir, 'CS_' + str(smp) + '_.mat'))['data_struct']

    B_theta_tensor = torch.ones([100, 1]) * float(physical_data_struct['B_theta'])
    B_phi_tensor = torch.ones([100, 1]) * float(physical_data_struct['B_phi'])
    B_mag_tensor = torch.ones([100, 1]) * float(physical_data_struct['B_mag'])

    ten = B_theta_tensor, B_phi_tensor, B_mag_tensor
    combine_physical = torch.stack(ten, dim=0)
    physical_params_tensors[smp, :, :, :] = combine_physical

    # Outputs:
    target = torch.from_numpy(physical_data_struct['target'][0][0])
    target_tensors[smp, :, :] = target

    x = 1

batch_size = 10
tr_params = {'batch_size': batch_size, 'shuffle': True}

sig_meas_set = simulated_NV_DataSet.simulated_NV_DataSet(sig_measurements_tensors, target_tensors)
smp_mtx_set = simulated_NV_DataSet.simulated_NV_DataSet(smp_mtx_tensors, target_tensors)
physical_set = simulated_NV_DataSet.simulated_NV_DataSet(physical_params_tensors, target_tensors)

training_generator1 = torch.utils.data.DataLoader(sig_meas_set, **tr_params)
training_generator2 = torch.utils.data.DataLoader(smp_mtx_set, **tr_params)
training_generator3 = torch.utils.data.DataLoader(physical_set, **tr_params)

max_epochs = 5

# Learning rate for optimizers
lr = 0.0001  # 0.0002
# Beta1 hyper-param for Adam optimizers
beta1 = 0.9

training_generator = zip(training_generator1, training_generator2, training_generator3)


def train(net: NV_simulateed_data_model, criterion=nn.MSELoss()):
    optimizer = optim.Adam(net.parameters(), lr=lr, betas=(beta1, 0.999))
    # losses, gen_lst, iters = [], [], 0
    # Loop over epochs
    for epoch in range(max_epochs):
        # Training
        training_generator = zip(training_generator1, training_generator2, training_generator3)
        for i, ((sig_meas_batch, target_batch), (smp_mtx_batch, _), (physics_batch, _)) in enumerate(
                training_generator):
            optimizer.zero_grad()

            # Transfer to GPU
            sig_meas_batch, smp_mtx_batch, physics_batch = sig_meas_batch.to(device), smp_mtx_batch.to(
                device), physics_batch.to(device)
            target_batch = target_batch.to(device)

            net_rec = net(sig_meas_batch, smp_mtx_batch, physics_batch)
            err = criterion(net_rec, target_batch)

            err.backward()  # perform back-propagation

            # Output training stats
            if i % 10 == 0:
                print('Epoch num: ', epoch, ' Iteration num: ', i)
                print('Epoch num %d, out of %d. Batch num %d, out of %d. \tLoss %.22f\t' % (
                    epoch, max_epochs, i, (NUN_REAL_SAMPS / batch_size), err.item()))


if __name__ == '__main__':
    AE = NV_simulateed_data_model.NV_simulateed_data_model().to(device)
    AE.apply(NV_simulateed_data_model.weights_init)

    # AE = network_class.network_class().to(device)
    # AE.apply(network_class.weights_init)

    # vec = torch.rand(1, 4096)
    # output = AE(vec)

    start_time = time.time()
    # train(AE, training_generator=training_generator)
    train(AE)
    print(time.time() - start_time)
    torch.save(AE.state_dict(), os.path.join(rootdir, 'net'))

# list1 = [("a", "a"), ("b", "b"), ("c", "c"), ("d", "d")]
# list2 = [1, 2, 3, 4]
# zip_object = zip(list1, list2)
#
# for i, (el1, el2) in enumerate(zip_object):
#     print(i, el1, el2)
#
# for i, ((batch1, lable), (batch2, lable)) in enumerate(training_generator):
#     print('batch1.shape: ', batch1.shape, 'batch2.shape: ', batch2.shape, 'lable: ', lable.shape)
