import torch
from torch import nn

in_channels = 1
proj_channels = 50
CELU_alpha = 2.5


class NV_real_data_model(nn.Module):
    def __init__(self):
        super(NV_real_data_model, self).__init__()
        self.proj_path = nn.Sequential(
            #  frequency vector size is 200 we have 50% so the measurements vector is 100.
            nn.Linear(100, 100),
            nn.Sigmoid(),

            nn.Linear(100, 200),
            nn.Sigmoid(),
            nn.BatchNorm1d(1),

            nn.Linear(200, 400),
            nn.Sigmoid(),
            nn.BatchNorm1d(1),

            nn.Linear(400, 800),
            nn.Sigmoid(),
            nn.BatchNorm1d(1),

            nn.Linear(800, 1600),
            nn.Sigmoid(),
            nn.BatchNorm1d(1),

            nn.Linear(1600, 400),
            nn.Sigmoid(),
            nn.BatchNorm1d(1),

            nn.Conv1d(1, 50, kernel_size=3, stride=1, padding=1),
            torch.nn.GELU(),
            nn.BatchNorm1d(50),

            nn.Conv1d(50, 50, kernel_size=3, stride=1, padding=1),
            torch.nn.GELU(),
            nn.BatchNorm1d(proj_channels),

            nn.Conv1d(50, 50, kernel_size=3, stride=1, padding=1),
            torch.nn.GELU(),
            nn.BatchNorm1d(proj_channels),

            nn.Conv1d(50, 50, kernel_size=3, stride=1, padding=1),
            torch.nn.GELU(),
            nn.BatchNorm1d(proj_channels),

            # nn.CELU(alpha=CELU_alpha, inplace=False),
            nn.Linear(400, 400)

        )

        self.matx_path = nn.Sequential(
            # The matrix input is of size 100x200.
            nn.Linear(200, 200),
            # nn.ELU(alpha=1.0, inplace=False),
            # torch.nn.ReLU(inplace=False),
            nn.LeakyReLU(negative_slope=0.01, inplace=False),
            nn.BatchNorm2d(1),
            # size 50x200

            nn.Linear(200, 1000),
            # nn.ELU(alpha=1.0, inplace=False),
            # torch.nn.ReLU(inplace=False),
            nn.LeakyReLU(negative_slope=0.01, inplace=False),
            nn.BatchNorm2d(1),
            # size 50x400

            nn.Linear(1000, 1000),
            # nn.ELU(alpha=1.0, inplace=False),
            nn.LeakyReLU(negative_slope=0.01, inplace=False),
            # torch.nn.ReLU(inplace=False),
            nn.BatchNorm2d(1),

            nn.Linear(1000, 1000),
            # nn.ELU(alpha=1.0, inplace=False),
            # torch.nn.ReLU(inplace=False),
            nn.LeakyReLU(negative_slope=0.01, inplace=False),
            nn.BatchNorm2d(1),

            nn.Linear(1000, 1000),
            # torch.nn.ReLU(inplace=False),
            nn.LeakyReLU(negative_slope=0.01, inplace=False),
            # nn.ELU(alpha=1.0, inplace=False),
            nn.BatchNorm2d(1),

            nn.Linear(1000, 200),
            # nn.ELU(alpha=1.0, inplace=False),
            nn.LeakyReLU(negative_slope=0.01, inplace=False),
            # torch.nn.ReLU(inplace=False),
            nn.BatchNorm2d(1),

            nn.Conv2d(1, 100, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            # nn.ELU(alpha=1.0, inplace=False),
            nn.Sigmoid(),
            nn.BatchNorm2d(100),

            nn.Conv2d(100, 100, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            # nn.ELU(alpha=1.0, inplace=False),
            nn.Sigmoid(),
            nn.BatchNorm2d(100),

            nn.Conv2d(100, 100, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            # nn.ELU(alpha=1.0, inplace=False),
            nn.Sigmoid(),
            nn.BatchNorm2d(100),

            nn.Conv2d(100, 1, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            # nn.ELU(alpha=1.0, inplace=False),
            nn.Sigmoid(),
            nn.BatchNorm2d(1),

            nn.Linear(200, 200),
            # nn.ELU(alpha=1.0, inplace=False),
            nn.BatchNorm2d(1),


        )

        self.combined_paths = nn.Sequential(
            nn.Linear(40000, 1000),
            nn.Sigmoid(),
            nn.BatchNorm2d(1),

            nn.Linear(1000, 500),
            nn.Sigmoid(),
            nn.Linear(500, 200),
            nn.Sigmoid(),
            nn.BatchNorm2d(1),

            nn.Linear(200, 200),
            nn.Sigmoid(),

            # nn.Linear(100, 100),
            # nn.BatchNorm2d(1),

            nn.Tanh()
            # nn.CELU(inplace=False),

        )

    def magic_combine(self, x, dim_begin, dim_end):
        combined_shape = list(x.shape[:dim_begin]) + [-1] + list(x.shape[dim_end:])
        return x.view(combined_shape)

    # a = torch.zeros(1, 2, 3, 4, 5, 6)
    # b = magic_combine(a, 2, 5)  # combine dimension 2, 3, 4
    # print(b.size())

    def forward(self, proj_inpt, mat_inpt):
        proj_out = self.proj_path(proj_inpt)
        mat_out = self.matx_path(mat_inpt[:, None, :, :])
        x1 = proj_out.view(proj_out.size(0), -1)
        x2 = mat_out.view(mat_out.size(0), -1)
        combined = torch.cat((x1, x2), dim=1)
        x = 1
        # out = self.magic_combine(self.combined_paths(combined[None, None, :, :]), 2, 4)
        out = self.combined_paths(combined[:, None, None, :])
        return out


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

# Usage: torch.nn.Conv1d(in_channels, out_channels, kernel_size, stride=1, padding=0,
# dilation=1, groups=1, bias=True, padding_mode='zeros', device=None, dtype=None)
