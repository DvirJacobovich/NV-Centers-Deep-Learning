import torch


def norm_measures(measure_tensor):
    """
    No need to keep the mean and std since measure is the input.
    :param measure_tensor:
    """
    meas = torch.squeeze(measure_tensor)
    meas_scaled = meas - torch.mean(meas)
    x1_scaled_normed = meas_scaled / torch.std(meas)
    return x1_scaled_normed


def norm_lables(inp_lable):
    x = torch.squeeze(inp_lable)
    x = x / x.sum(0).expand_as(x)
    x[torch.isnan(x)] = 0  # if an entire column is zero, division by 0 will cause NaNs
    x = 2 * x - 1

    # lable = torch.squeeze(inp_lable)
    # min_lab = torch.min(lable)
    # max_lab = torch.max(lable)
    # x = lable - min_lab
    # x = x / max_lab
    # x = 2*x - 1
    return x


def norm_nv(lable_nv):
    """
    Returns the balance scaled and norm. To recover back after getting the
    network output y we have: y' = y + scaled_mean, and then y'' = y' * original_std to
    the real original values of the balance.
    :return:
    """
    nv = torch.squeeze(lable_nv)
    original_std = torch.std(nv)
    nv_scaled = torch.div(nv, original_std)
    scaled_mean = torch.mean(nv_scaled)
    nv_scaled_normed = nv_scaled - scaled_mean
    nv_scaled_normed = nv_scaled_normed[None, :, :]
    return nv_scaled_normed, scaled_mean, original_std


def recover_nv(net_output, scaled_mean, original_std):
    return original_std * (net_output + scaled_mean)


def norm_all_nv(train_nv):
    for i in range(len(train_nv)):
        # print('mean before: ', torch.mean(train_balance[i, :, :, :]), 'std before: ', torch.std(train_balance[i, :, :, :]))
        train_nv[i, :, :] = norm_nv(train_nv[i, :, :])[0]
        # print('mean adter: ', torch.mean(train_balance[i, :, :, :]), 'std after: ', torch.std(train_balance[i, :, :, :]))
    return train_nv


def norm_all_measure(measure_tensor):
    for i in range(len(measure_tensor)):
        print('mean before: ', torch.mean(measure_tensor[i, :]), 'std before: ', torch.std(measure_tensor[i, :]))
        measure_tensor[i, :] = norm_measures(measure_tensor[i, :])
        print('mean adter: ', torch.mean(measure_tensor[i, :]), 'std after: ', torch.std(measure_tensor[i, :]))

    return measure_tensor
