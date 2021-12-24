import os
import time
import scipy.io as sio
import torch
import torch.nn as nn
import torch.optim as optim
import NV_plots
import simulated_NV_DataSet
import NV_simulateed_data_model
import validation_generator as vg
from matplotlib import pyplot as plt

##
# Simlualated data: inputs: in CS staruct: smp_mtx (100x200), basis_mtx (200x200) (need to multiply these two to get the
# 100x200 Theta (maybe not need the basis_max), sig_measurements(100, 1), and at last inside CS struct we have 'sim' struct
# that conclude the physical parameters such that: B_mag, B_phi, B_theta (can be represented by 3 matrices). we need to get
# from galya fixed dimensions and physical parameters range so that in these dimensions and applying this ranges we
# have 8 lorenzian peaks.
#
# As for the lables we have 'target' (200, 1) in the 'sim' struct inside 'CS'.

sim_data_loc = r'C:\Users\owner\Desktop\Dvir\NV-centers Compressed learning\NV-centers CompressedLearning-20211121T154621Z-001\compressedLearning-main\matlab_code\COmpSENS'

NUM_SIM_SAMPS = 100  # max 1501
TANH = nn.Tanh()

use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
torch.backends.cudnn.benchmark = True

path_parent = os.path.dirname(os.getcwd())
rootdir = os.path.join(path_parent, 'matlab_code\COmpSENS')

# inputs tensors:
# basis_mtx_tensors = torch.empty([NUM_SIM_SAMPS, 200, 200])

sig_measurements_tensors = torch.ones([NUM_SIM_SAMPS, 1, 100])
smp_mtx_tensors = torch.ones([NUM_SIM_SAMPS, 1, 100, 200])
physical_params_tensors = torch.ones([NUM_SIM_SAMPS, 100, 3])

# lables tensors:
target_tensors = torch.ones([NUM_SIM_SAMPS, 200, 1])

for smp in range(NUM_SIM_SAMPS):
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

batch_size = 32
tr_params = {'batch_size': batch_size, 'shuffle': False}

sig_meas_set = simulated_NV_DataSet.simulated_NV_DataSet(sig_measurements_tensors, target_tensors)
smp_mtx_set = simulated_NV_DataSet.simulated_NV_DataSet(smp_mtx_tensors, target_tensors)
physical_set = simulated_NV_DataSet.simulated_NV_DataSet(physical_params_tensors, target_tensors)

tr_gen1 = torch.utils.data.DataLoader(sig_meas_set, **tr_params)
tr_gen2 = torch.utils.data.DataLoader(smp_mtx_set, **tr_params)
tr_gen3 = torch.utils.data.DataLoader(physical_set, **tr_params)

valid_gen1, valid_gen2, valid_gen3 = vg.validation_generator()
# num epochs
max_epochs = 5
# Learning rate for optimizers
lr = 0.0001  # 0.0002
# Beta1 hyper-param for Adam optimizers
beta1 = 0.9

training_gen = zip(tr_gen1, tr_gen2, tr_gen3)
validation_gen = zip(valid_gen1, valid_gen2, valid_gen3)


def train(net: NV_simulateed_data_model, criterion=nn.MSELoss()):
    optimizer = optim.Adam(net.parameters(), lr=lr, betas=(beta1, 0.999))
    losses, gen_lst = [], []
    # Loop over epochs
    for epoch in range(max_epochs):
        # Training
        training_generator = zip(tr_gen1, tr_gen2, tr_gen3)
        for i, ((sig_meas_batch, target_batch), (smp_mtx_batch, _), (physics_batch, _)) in enumerate(
                training_generator):
            optimizer.zero_grad()

            # Transfer to GPU
            sig_meas_batch, smp_mtx_batch, physics_batch = sig_meas_batch.to(device), smp_mtx_batch.to(
                device), physics_batch.to(device)
            target_batch = target_batch.to(device)
            target_batch = torch.transpose(target_batch, dim0=2, dim1=1)
            # target_batch = TANH(target_batch)

            net_rec = net(sig_meas_batch, smp_mtx_batch, physics_batch)
            err = criterion(net_rec, target_batch)

            err.backward()  # perform back-propagation

            # Output training stats
            if i % 10 == 0:
                print('Epoch num: ', epoch, ' Iteration num: ', i)
                print('Epoch num %d, out of %d. Batch num %d, out of %d. \tLoss %.22f\t' % (
                    epoch, max_epochs, i, (NUM_SIM_SAMPS / batch_size), err.item()))

            losses.append(err.item())

            # loss_file = open("losses.txt", "w")  # write losses to text file
            # for element in losses:
            #     loss_file.write(str(element) + "\n")
            # loss_file.close()

            # Validation
            with torch.set_grad_enabled(False):
                validation_generator = zip(valid_gen1, valid_gen2, valid_gen3)
                for i, ((sig_meas_valid, target_valid), (smp_mtx_valid, _), (physics_valid, _)) in enumerate(
                        validation_generator):
                    sig_meas_valid, smp_mtx_valid, physics_valid = sig_meas_valid.to(device), smp_mtx_valid.to(
                        device), physics_valid.to(device)
                    target_valid = target_valid.to(device)
                    target_valid = torch.transpose(target_valid, dim0=2, dim1=1)
                    # target_valid = TANH(target_valid)

                    net_rec = net(sig_meas_valid, smp_mtx_valid, physics_valid)
                    NV_plots.NV_plots(net_rec, target_valid)

    plt.plot(losses)
    plt.show()


if __name__ == '__main__':
    AE = NV_simulateed_data_model.NV_simulateed_data_model().to(device)
    AE.apply(NV_simulateed_data_model.weights_init)

    start_time = time.time()
    train(AE)
    print(time.time() - start_time)
    torch.save(AE.state_dict(), os.path.join(rootdir, 'net'))
