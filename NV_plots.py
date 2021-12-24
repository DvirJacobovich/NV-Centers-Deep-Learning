import torch
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable


def tensor_to_plt_im(im: torch.Tensor):
    return im.permute(1, 2, 0)


def NV_plots(net_rec, target):
    plt.plot(torch.squeeze(net_rec), label='Net rec', color='r')
    plt.plot(torch.squeeze(target), label='Target', color='b')
    # plt.title('Target Lorentians')
    plt.title('Net Recs and Target Lorentzians')

    plt.legend()
    plt.show()


def NV_plot(net_rec):
    fig = plt.figure()
    plt.imshow(net_rec)
    plt.colorbar(label="Reconstruction Network output", orientation="vertical")
    plt.title('Reconstruction Network output with 25%')

    plt.show()


def show_diff_nv(net_rec, original, check):
    fig = plt.figure(figsize=(10, 5), dpi=50)
    ax1 = fig.add_subplot(1, 2, 1)
    plt.title(check + ' check Net Rec (Normed) 25%')
    b1 = ax1.imshow(net_rec)

    divider1 = make_axes_locatable(ax1)
    cax1 = divider1.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(b1, cax=cax1, orientation='vertical')

    ax2 = fig.add_subplot(1, 2, 2)
    plt.title(check + ' check Orig (Normed)')
    b2 = ax2.imshow(original)

    divider2 = make_axes_locatable(ax2)
    cax2 = divider2.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(b2, cax=cax2, orientation='vertical')

    plt.show()
