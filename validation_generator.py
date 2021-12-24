import os
import time
import scipy.io as sio
import torch
import torch.nn as nn
import torch.optim as optim
import simulated_NV_DataSet
import NV_simulateed_data_model


def validation_generator():
    NUM_SIM_VALID_SAMPS = 2

    path_parent = os.path.dirname(os.getcwd())
    rootdir = os.path.join(path_parent, 'matlab_code\COmpSENS\\validation_data')

    sig_measurements_tensors = torch.ones([NUM_SIM_VALID_SAMPS, 1, 100])
    smp_mtx_tensors = torch.ones([NUM_SIM_VALID_SAMPS, 1, 100, 200])
    physical_params_tensors = torch.ones([NUM_SIM_VALID_SAMPS, 100, 3])

    # lables tensors:
    target_tensors = torch.ones([NUM_SIM_VALID_SAMPS, 200, 1])

    for smp in range(NUM_SIM_VALID_SAMPS):
        file_name = os.path.join(rootdir, 'CS_' + str(smp) + '_.mat')
        # inputs:
        CS_struct = sio.loadmat(os.path.join(rootdir, 'CS_' + str(smp) + '_.mat'))['cs']

        sig_measurements = torch.from_numpy(CS_struct['sig_measurements'][0][0].T)
        sig_measurements_tensors[smp, :, :] = sig_measurements

        smp_mtx = torch.from_numpy(CS_struct['smp_mtx'][0][0])[None, :, :]
        smp_mtx_tensors[smp, :, :, :] = smp_mtx

        # basis_mtx = torch.from_numpy(CS_struct['basis_mtx'][0][0])

        # physical inputs:
        physical_data_struct = sio.loadmat(os.path.join(rootdir, 'CS_' + str(smp) + '_.mat'))['data_struct']

        B_theta_tensor = torch.ones([100]) * float(physical_data_struct['B_theta'])
        B_phi_tensor = torch.ones([100]) * float(physical_data_struct['B_phi'])
        B_mag_tensor = torch.ones([100]) * float(physical_data_struct['B_mag'])

        ten = B_theta_tensor, B_phi_tensor, B_mag_tensor
        combine_physical = torch.stack(ten, dim=1)
        physical_params_tensors[smp, :, :] = combine_physical

        # Outputs:
        target = torch.from_numpy(physical_data_struct['target'][0][0])
        target_tensors[smp, :, :] = target

        x = 1

    batch_size = 1
    tr_params = {'batch_size': batch_size, 'shuffle': False}

    sig_meas_set = simulated_NV_DataSet.simulated_NV_DataSet(sig_measurements_tensors, target_tensors)
    smp_mtx_set = simulated_NV_DataSet.simulated_NV_DataSet(smp_mtx_tensors, target_tensors)
    physical_set = simulated_NV_DataSet.simulated_NV_DataSet(physical_params_tensors, target_tensors)

    vald_gen1 = torch.utils.data.DataLoader(sig_meas_set, **tr_params)
    vald_gen2 = torch.utils.data.DataLoader(smp_mtx_set, **tr_params)
    vald_gen3 = torch.utils.data.DataLoader(physical_set, **tr_params)

    return vald_gen1, vald_gen2, vald_gen3
