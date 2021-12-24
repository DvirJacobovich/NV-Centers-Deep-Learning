import torch
from torch import nn

in_channels = 1
proj_channels = 50
CELU_alpha = 2.5
negative_slope = 1e-2
COMB_PATH_CHANNELS = 500


class NV_simulateed_data_model(nn.Module):
    def __init__(self):
        super(NV_simulateed_data_model, self).__init__()
        self.proj_path = nn.Sequential(
            #  Frequency vector size is 200 we have 50% so the measurements vector is 100.
            nn.BatchNorm1d(1),
            nn.Linear(100, 100),
            nn.Sigmoid(),
            nn.BatchNorm1d(1),
            # size: batch_size x (1 x 100)

            ###################### First part ######################

            nn.Linear(100, 400),
            nn.Sigmoid(),
            nn.BatchNorm1d(1),
            # size: batch_size x (1 x 400)

            nn.Conv1d(1, 1000, kernel_size=1, stride=1, padding=0),
            nn.ReLU(),
            nn.BatchNorm1d(1000),
            # size: batch_size x (1000 x 400)

            nn.Conv1d(1000, 1, kernel_size=1, stride=1, padding=0),
            nn.ReLU(),
            nn.BatchNorm1d(1),
            # size: batch_size x (1 x 400)

            ###################### Second part ######################

            nn.Linear(400, 1000),
            nn.Sigmoid(),
            nn.BatchNorm1d(1),
            # size: batch_size x (1 x 1000)

            nn.Conv1d(1, 100, kernel_size=1, stride=1, padding=0),
            nn.ReLU(),
            nn.BatchNorm1d(100),
            # size: batch_size x (100 x 1000)

            nn.Conv1d(100, 1, kernel_size=1, stride=1, padding=0),
            nn.ReLU(),
            nn.BatchNorm1d(1),
            # size: batch_size x (1 x 1000)

            ###################### Third part ######################

            nn.Linear(1000, 10000),
            nn.Sigmoid(),
            nn.BatchNorm1d(1),
            # size: batch_size x (1 x 10000)

            nn.Conv1d(1, 100, kernel_size=1, stride=1, padding=0),
            nn.ReLU(),
            nn.BatchNorm1d(100),
            # size: batch_size x (100 x 10000)

            nn.Conv1d(100, 1, kernel_size=1, stride=1, padding=0),
            nn.ReLU(),
            nn.BatchNorm1d(1),
            # size: batch_size x (1 x 10000)

        )

        self.matx_path = nn.Sequential(

            ################### First part ###################

            # The matrix input is of size 100x200.
            nn.Linear(200, 200),
            nn.Sigmoid(),
            nn.BatchNorm2d(1),
            # size: batch_size x (100 x 200)

            nn.Conv2d(1, 100, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            # nn.LeakyReLU(negative_slope=0.01, inplace=False),
            nn.Sigmoid(),
            nn.BatchNorm2d(100),
            # size: batch_size x 100 x (100 x 200)

            nn.Conv2d(100, 1, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            # nn.LeakyReLU(negative_slope=0.01, inplace=False),
            nn.Sigmoid(),
            nn.BatchNorm2d(1),
            # size: batch_size x 1 x (100 x 200)

            ################### Second part ###################

            nn.Linear(200, 1000),
            nn.Sigmoid(),
            # nn.LeakyReLU(negative_slope=0.01, inplace=False),
            nn.BatchNorm2d(1),
            # size: batch_size x 1 x (100 x 1000)

            nn.Conv2d(1, 100, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.Sigmoid(),
            # nn.LeakyReLU(negative_slope=0.01, inplace=False),
            nn.BatchNorm2d(100),
            # size: batch_size x 100 x (100 x 1000)

            nn.Conv2d(100, 1, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            # nn.LeakyReLU(negative_slope=0.01, inplace=False),
            nn.Sigmoid(),
            nn.BatchNorm2d(1),
            # size: batch_size x 1 x (100 x 1000)

            ################### Third part ###################

            nn.Linear(1000, 1000),
            # nn.LeakyReLU(negative_slope=0.01, inplace=False),
            nn.Sigmoid(),
            nn.BatchNorm2d(1),
            # size: batch_size x 1 x (100 x 1000)

            nn.Conv2d(1, 100, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            # nn.LeakyReLU(negative_slope=0.01, inplace=False),
            nn.Sigmoid(),
            nn.BatchNorm2d(100),
            # size: batch_size x 100 x (100 x 1000)

            nn.Conv2d(100, 1, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            # nn.LeakyReLU(negative_slope=0.01, inplace=False),
            nn.Sigmoid(),
            nn.BatchNorm2d(1),
            # size: batch_size x 1 x (100 x 1000)

            nn.Linear(1000, 100),
            nn.BatchNorm2d(1),
            # size: batch_size x 1 x (100 x 100)

        )

        self.combined_paths = nn.Sequential(
            # input here is of size: batch_size x 30000
            nn.Linear(30000, 200),
            nn.Sigmoid(),
            nn.BatchNorm1d(1),

            nn.Conv1d(1, COMB_PATH_CHANNELS, kernel_size=3, stride=1, padding=1),
            # nn.LeakyReLU(negative_slope=negative_slope),
            nn.ReLU(),
            nn.BatchNorm1d(COMB_PATH_CHANNELS),

            nn.Conv1d(COMB_PATH_CHANNELS, COMB_PATH_CHANNELS, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            # nn.LeakyReLU(negative_slope=negative_slope, inplace=False),
            nn.BatchNorm1d(COMB_PATH_CHANNELS),

            nn.Conv1d(COMB_PATH_CHANNELS, COMB_PATH_CHANNELS, kernel_size=3, stride=1, padding=1),
            # nn.LeakyReLU(negative_slope=negative_slope, inplace=False),
            nn.ReLU(),
            nn.BatchNorm1d(COMB_PATH_CHANNELS),

            nn.Conv1d(COMB_PATH_CHANNELS, COMB_PATH_CHANNELS, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            # nn.LeakyReLU(negative_slope=negative_slope, inplace=False),
            nn.BatchNorm1d(COMB_PATH_CHANNELS),

            nn.Conv1d(COMB_PATH_CHANNELS, COMB_PATH_CHANNELS, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            # nn.LeakyReLU(negative_slope=negative_slope, inplace=False),
            nn.BatchNorm1d(COMB_PATH_CHANNELS),

            nn.Conv1d(COMB_PATH_CHANNELS, COMB_PATH_CHANNELS, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            # nn.LeakyReLU(negative_slope=negative_slope, inplace=False),
            nn.BatchNorm1d(COMB_PATH_CHANNELS),

            nn.Conv1d(COMB_PATH_CHANNELS, COMB_PATH_CHANNELS, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            # nn.LeakyReLU(negative_slope=negative_slope, inplace=False),
            nn.BatchNorm1d(COMB_PATH_CHANNELS),

            nn.Conv1d(COMB_PATH_CHANNELS, 1, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            # nn.LeakyReLU(negative_slope=negative_slope, inplace=False),
            # nn.BatchNorm1d(1)
            nn.Linear(200, 200),
            # nn.Tanh(),
            nn.Sigmoid()
            # torch.nn.CELU(alpha=1.0, inplace=False)

        )

    def magic_combine(self, x, dim_begin, dim_end):
        combined_shape = list(x.shape[:dim_begin]) + [-1] + list(x.shape[dim_end:])
        return x.view(combined_shape)

    def forward(self, proj_input, mat_input, magnetic_params_input):
        """
        :param proj_input: size: batch_size x 1 x 100
        :param mat_input: batch_size x 1 x 100 x 200
        :param magnetic_params_input: batch_size x 100 x 3
        :return: out - size: target size.
        """
        proj_out = self.proj_path(proj_input)  # proj_out size: batch_size x (1 x 10000)
        mat_out = self.matx_path(mat_input)  # mat_out size: batch_size x 1 x (100 x 100)

        x1 = proj_out.view(proj_out.size(0), -1)  # size: batch_size x 10000
        x2 = mat_out.view(mat_out.size(0), -1)  # size: batch_size x 10000

        # The physical Magnetic field parameters sets the locations and the shapes of the
        # eight Lorentzians peaks (due to the projection of the field by the angles and mag).
        # That's why we set these parameter at this point when we are starting to reconstruct
        # the Fluorescency.

        l1 = nn.Linear(3, 100)  # turns Magnetic parameters from 100 x 3 to - 100 x 100
        physics_out = l1(magnetic_params_input)

        x3 = physics_out.view(physics_out.size(0), -1)  # size: batch_size x 10000

        combined = torch.cat((x1, x2, x3), dim=1)[:, None, :]
        x = 1
        # out = self.magic_combine(self.combined_paths(combined[None, None, :, :]), 2, 4)
        out = self.combined_paths(combined)
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
