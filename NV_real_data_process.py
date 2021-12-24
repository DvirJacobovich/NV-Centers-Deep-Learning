import os
import time
import scipy.io as sio
import torch
import torch.nn as nn
import NV_DataSet
import NV_real_data_model
import torch.optim as optim
from matplotlib import pyplot as plt
from torchvision import transforms
import torch.nn.functional as f

import NV_normalize_data as nr

NUN_REAL_SAMPS = 99
TANH = nn.Tanh()

use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
torch.backends.cudnn.benchmark = True

path_parent = os.path.dirname(os.getcwd())
rootdir = os.path.join(path_parent, 'Data')

measure_inp_tensor = torch.empty([NUN_REAL_SAMPS, 1, 100])
mtx_inp_tensor = torch.empty([NUN_REAL_SAMPS, 100, 200])
freq_lable_tensor = torch.empty([NUN_REAL_SAMPS, 1, 1, 200])

for smp_num in range(1, NUN_REAL_SAMPS):
    samp_strc = sio.loadmat(os.path.join(rootdir, 'ARF1_3freq_' + str(smp_num) + '.mat'))['exp']
    measure_inp_tensor[smp_num - 1, :, :] = nr.norm_measures(torch.tensor(samp_strc['sig_measurements'][0][0].T).float())[None, :]
    mtx_inp_tensor[smp_num - 1, :, :] = nr.norm_measures(torch.tensor(samp_strc['smp_mtx'][0][0]).float())
    # freq_lable_tensor[smp_num - 1, :, :] = nr.norm_measures(torch.tensor(samp_strc['fit'][0][0]).float())[:, None]
    x = 1
    # freq_lable_tensor[smp_num - 1, :, :, :] = TANH(torch.tensor(samp_strc['fit'][0][0]).float())[:, :, None]
    freq_lable_tensor[smp_num - 1, :, :, :] = torch.squeeze(TANH(torch.tensor(samp_strc['fit'][0][0]).float()))[None, None, :]
# torch.squeeze(TANH(torch.tensor(samp_strc['fit'][0][0]).float()))[None, None, :]


batch_size = 10
tr_params = {'batch_size': batch_size, 'shuffle': True}

inp1_set = NV_DataSet.NvDataSet(measure_inp_tensor, freq_lable_tensor)
inp2_set = NV_DataSet.NvDataSet(mtx_inp_tensor, freq_lable_tensor)

training_generator1 = torch.utils.data.DataLoader(inp1_set, **tr_params)
training_generator2 = torch.utils.data.DataLoader(inp2_set, **tr_params)

max_epochs = 5

# Learning rate for optimizers
lr = 0.0001  # 0.0002
# Beta1 hyper-param for Adam optimizers
beta1 = 0.9

training_generator = zip(training_generator1, training_generator2)


def train(net: NV_real_data_model, criterion=nn.MSELoss()):
    optimizer = optim.Adam(net.parameters(), lr=lr, betas=(beta1, 0.999))
    losses, gen_lst, iters = [], [], 0
    # Loop over epochs
    for epoch in range(max_epochs):
        # Training
        training_generator = zip(training_generator1, training_generator2)
        for i, ((batch1, lable), (batch2, lable)) in enumerate(training_generator):
            optimizer.zero_grad()
            # Transfer to GPU
            batch1, batch2, lable = batch1.to(device), batch2.to(device), lable.to(device)

            net_rec = net(batch1, batch2)
            err = criterion(net_rec, lable)

            err.backward()  # perform back-propagation

            # Output training stats
            if i % 10 == 0:
                print('Epoch num: ', epoch, ' Iteration num: ', i)
                print('Epoch num %d, out of %d. Batch num %d, out of %d. \tLoss %.22f\t' % (
                    epoch, max_epochs, i, (NUN_REAL_SAMPS / batch_size), err.item()))


if __name__ == '__main__':
    AE = NV_real_data_model.NV_real_data_model().to(device)
    AE.apply(NV_real_data_model.weights_init)

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
