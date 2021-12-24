import torch
import torch.nn as nn
import validation_generator

vald_gen1, vald_gen2, vald_gen3 = validation_generator.validation_generator()

t = torch.rand([2, 2])
g = nn.Softmin()
print(g(t).size())
#
# batch_size = 10
#
# mat = torch.rand([batch_size, 1, 100, 100])
# proj = torch.rand([batch_size, 1, 10000])
#
# x1 = proj.view(proj.size(0), -1)
# x2 = mat.view(mat.size(0), -1)
#
# print(x1.size())
# print(x2.size())
#
# physics_inpt = torch.rand([batch_size, 100, 3])
#
# l1 = nn.Linear(3, 100)  # turns physical parameters from 100 x 3 to - 100 x 100
# out = l1(physics_inpt)[:, None, :, :]
# print(out.size())
#
# x3 = out.view(out.size(0), -1)  # size: batch_size x 10000
# print(x3.size())
#
# combined = torch.cat((x1, x2, x3), dim=1)
# print(combined.size())
#
# y = nn.Linear(30000, 200)
# o = y(combined)[:, :, None]
# print(o.size())
#
# # batch_size = 10
# # t = torch.rand([batch_size, 1, 100, 200])
# # print('input: ', t.size())
# #
# # l1 = nn.Linear(200, 200)
# # out1 = l1(t)
# # print('FC1 ', out1.size())
# #
# # c1 = nn.Conv2d(1, 100, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
# # out2 = c1(out1)
# # print('Conv1: ', out2.size())
# #
# # c2 = nn.Conv2d(100, 1, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
# # out3 = c2(out2)
# # print('Conv2: ', out3.size())
# #
# # l2 = nn.Linear(200, 1000)
# # out4 = l2(out3)
# # print('FC2: ', out4.size())
# #
# # c3 = nn.Conv2d(1, 100, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
# # out5 = c3(out4)
# # print('Conv3: ', out5.size())
# #
# # c4 = nn.Conv2d(100, 1, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
# # out6 = c4(out5)
# # print('Conv4: ', out6.size())
# #
# # l3 = nn.Linear(1000, 1000)
# # out7 = l3(out6)
# # print('FC3: ', out7.size())
# #
# # c5 = nn.Conv2d(1, 100, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
# # out8 = c5(out7)
# # print('Conv5: ', out8.size())
# #
# # c6 = nn.Conv2d(100, 1, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
# # out9 = c6(out8)
# # print('Conv6: ', out9.size())
# #
