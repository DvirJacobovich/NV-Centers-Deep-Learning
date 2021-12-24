import os
import pathlib
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import mat73
from os.path import dirname, join as pjoin
import scipy.io as sio

meas = torch.rand(1, 200)
l1 = nn.Linear(200, 400)
out1 = l1(meas)
print(out1.size())

sig1 = nn.Sigmoid()
l2 = nn.Linear(400, 800)
out2 = l2(sig1(out1))
print(out2.size())

l3 = nn.Linear(800, 1600)
at1 = nn.MultiheadAttention(embed_dim=1600, num_heads=4)

# nn.Linear(800, 1600),
# nn.Linear(1600, 1600),
# nn.Linear(1600, 1600),
