import torch
import torch.autograd as autograd
import torch.optim as optim
import numpy as np
import matplotlib as mpl
from datetime import datetime

mpl.use('Agg')

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
from torch.autograd import Variable
# from tensorflow.examples.tutorials.mnist import input_data
import torch.nn as nn
import torch.nn.functional as F
import shutil,sys

#  for cgan series and fm_conv_overly_noise.py
class D_Net_conv(nn.Module):
    def __init__(self,ngpu,inchannel,main_gpu=0):
        super(D_Net_conv,self).__init__()
        self.ngpu = ngpu
        self.main_gpu = main_gpu
        self.pipeline = nn.Sequential(
            nn.Conv2d(inchannel,48,5), # 28-> 24
            nn.ELU(),
            nn.MaxPool2d(2),
            nn.Conv2d(48,128,5), # 12-> 8
            nn.ReLU(),
            nn.MaxPool2d(2), # 8->4
            nn.Conv2d(128,256,3,1,0),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.pipeline2 = nn.Sequential(
            nn.Linear(256,36),
            nn.ELU(),
            nn.Linear(36,1),
            nn.Sigmoid()
        )


    def forward(self,x):
        if isinstance(x.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.pipeline, x, range(self.main_gpu, self.main_gpu+self.ngpu))
            output = output.view(-1, 256)
            output = nn.parallel.data_parallel(self.pipeline2, output, range(self.main_gpu,self.main_gpu+self.ngpu))
        else:
            output = self.pipeline(x)
            output = output.view(-1, 256)
            output = self.pipeline2(output)

        return output

# For
class D_Net_conv_64(nn.Module):
    def __init__(self, ngpu,main_gpu,inchannel):
        super(D_Net_conv_64, self).__init__()
        self.ngpu = ngpu
        self.main_gpu = main_gpu
        self.ngf = 64
        self.pipeline = nn.Sequential(
            nn.Conv2d(inchannel, self.ngf, 4,2,1),  # 32
            nn.BatchNorm2d(self.ngf),
            nn.ELU(0.2),
            # nn.MaxPool2d(2),
            nn.Conv2d(self.ngf, self.ngf*2, 4,2,1),  # 16
            nn.BatchNorm2d(self.ngf*2),
            nn.ELU(0.2),
            # nn.MaxPool2d(2),  # 8->4
            nn.Conv2d(self.ngf*2, self.ngf*4, 4,2, 1),#8
            nn.BatchNorm2d(self.ngf*4),
            nn.ELU(0.2),
            nn.Conv2d(self.ngf*4, self.ngf*4, 4, 2, 1),  # 4
            nn.BatchNorm2d(self.ngf*4),
            nn.ELU(0.2),
            nn.Conv2d(self.ngf*4, 1, 4,1,0), # 1
            nn.ELU(0.2),
            nn.Sigmoid()
            # nn.MaxPool2d(2)
        )
        # self.pipeline2 = nn.Sequential(
        #     nn.Linear(256*16, 256),
        #     nn.ELU(),
        #     nn.Linear(256, 1),
        #     nn.Sigmoid()
        # )

    def forward(self, x):
        if isinstance(x.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.pipeline, x, range(self.main_gpu, self.main_gpu+self.ngpu))
            output = output.view(-1, 1)
            # output = nn.parallel.data_parallel(self.pipeline2, output, range(self.main_gpu,self.main_gpu+self.ngpu))
        else:
            output = self.pipeline(x)
            output = output.view(-1, 1)
            # output = self.pipeline2(output)
        return output


class E_Net_conv_64(nn.Module):
    def __init__(self, ngpu,main_gpu,inchannel,outchannel=10):
        super(E_Net_conv_64, self).__init__()
        self.ngpu = ngpu
        self.main_gpu = main_gpu
        self.ngf = 64
        self.outchannel = outchannel

        channel_1 = [inchannel, self.ngf,self.ngf*2,self.ngf*2, self.ngf*2]
        channel_2 = [self.ngf*2, self.ngf*4, self.ngf*5, self.ngf*5, self.ngf*6]
        channel_3 = [self.ngf*6, self.ngf*8, self.ngf*8, self.ngf*8]
        # channel_4 = [self.ngf*8, self.ngf*10]
        conv_layers = []
        conv_layers.extend(self.get_conv_groups(channels=channel_1,repeat_num=4))
        conv_layers.append(nn.MaxPool2d(2))
        conv_layers.extend(self.get_conv_groups(channels=channel_2,repeat_num=4))
        conv_layers.append(nn.MaxPool2d(2))
        conv_layers.extend(self.get_conv_groups(channels=channel_3,repeat_num=3))
        conv_layers.append(nn.MaxPool2d(2))
        conv_layers.append(nn.Conv2d(self.ngf*8, self.ngf*8,4)) # self.ngf * 10 * 2 *2
        conv_layers.append(nn.Conv2d(self.ngf*8, self.ngf*10,2,0))

        self.conv = torch.nn.Sequential(*conv_layers)
        self.pipeline2  = nn.Sequential(
            nn.Linear(self.ngf*10, self.ngf),
            nn.ELU(),
            nn.Linear(self.ngf, self.outchannel),
            nn.Sigmoid()
        )
        #
        # self.pipeline = nn.Sequential(
        #     nn.Conv2d(inchannel, self.ngf, 5,1),  # 60
        #     nn.BatchNorm2d(self.ngf),
        #     nn.ELU(),
        #     nn.MaxPool2d(2),#30
        #     nn.Conv2d(self.ngf, self.ngf*2, 5,1),  # 26
        #     nn.BatchNorm2d(self.ngf*2),
        #     nn.ELU(),
        #     nn.MaxPool2d(2),  # 13
        #     nn.Conv2d(self.ngf*2, self.ngf*4, 4,1),# 10
        #     nn.MaxPool2d(2), # 5
        #     nn.ELU(),
        #     nn.Conv2d(self.ngf*4, self.ngf*8, 5,1), #1
        #     nn.ELU()
        #     # nn.MaxPool2d(2)
        # )
    def get_conv_groups(self, channels, repeat_num):
        layers  = []
        assert np.size(channels)-1==repeat_num
        for i in range(repeat_num):
            layers.append(nn.Conv2d(channels[i], channels[i+1],4,1,1))
            layers.append(nn.BatchNorm2d(channels[i+1]))
            layers.append(nn.ELU(0.2))
        return layers

    def forward(self, x):
        if isinstance(x.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.conv, x, range(self.main_gpu, self.main_gpu+self.ngpu))
            output = output.view(-1, self.ngf*10)
            output = nn.parallel.data_parallel(self.pipeline2, output, range(self.main_gpu,self.main_gpu+self.ngpu))
        else:
            output = self.conv(x)
            output = output.view(-1, self.ngf*10)
            output = self.pipeline2(output)
        return output


# class G_Net_conv_64_old(nn.Module):
#     def __init__(self, ngpu,main_gpu,inchannel):
#         super(D_Net_conv_64_old, self).__init__()
#         self.ngpu = ngpu
#         self.main_gpu = main_gpu
#         self.pipeline = nn.Sequential(
#             nn.Conv2d(inchannel, 48, 4,2,1),  # 32
#             nn.BatchNorm2d(48),
#             nn.ELU(),
#             # nn.MaxPool2d(2),
#             nn.Conv2d(48, 128, 4,2,1),  # 16
#             nn.BatchNorm2d(128),
#             nn.ELU(),
#             # nn.MaxPool2d(2),  # 8->4
#             nn.Conv2d(128, 256, 4,2, 1),#8
#             nn.BatchNorm2d(256),
#             nn.ELU(),
#             nn.Conv2d(256, 256, 4, 2, 1),  # 4
#             nn.BatchNorm2d(256),
#             nn.ELU(),
#             # nn.MaxPool2d(2)
#         )
#         self.pipeline2 = nn.Sequential(
#             nn.Linear(256*16, 256),
#             nn.ELU(),
#             nn.Linear(256, 1),
#             nn.Sigmoid()
#         )
#
#     def forward(self, x):
#         if isinstance(x.data, torch.cuda.FloatTensor) and self.ngpu > 1:
#             output = nn.parallel.data_parallel(self.pipeline, x, range(self.main_gpu, self.main_gpu+self.ngpu))
#             output = output.view(-1, 256*16)
#             output = nn.parallel.data_parallel(self.pipeline2, output, range(self.main_gpu,self.main_gpu+self.ngpu))
#         else:
#             output = self.pipeline(x)
#             output = output.view(-1, 256*16)
#             output = self.pipeline2(output)
#
#         return output

class D_Net_Cifar10(nn.Module):
        def __init__(self, ngpu, input_features, output_features=1, main_gpu=0):
            super(D_Net_Cifar10, self).__init__()
            self.nz = 100
            self.ngf = 128
            self.ngpu = ngpu
            self.output_features = output_features
            self.main_gpu = main_gpu
            self.pipeline1 = nn.Sequential(
                nn.Conv2d(input_features, self.ngf, 5, 2, 0, bias=True),
                nn.BatchNorm2d(self.ngf),
                nn.ELU(True),
                nn.Conv2d(self.ngf, self.ngf * 2, 5, 2, 0, bias=True),
                nn.BatchNorm2d(self.ngf * 2),
                nn.ELU(True),
                nn.Conv2d(self.ngf * 2, self.ngf * 4, 5, 2, 0, bias=True),
                nn.BatchNorm2d(self.ngf * 4),
                nn.ELU(True),
                # nn.Conv2d(self.ngf * 4, self.ngf * 8, 5, 2, 0, bias=True),
                # nn.BatchNorm2d(self.ngf * 8),
                # state size. (self.ngf*2) x 15 x 15
                # nn.ELU(True),
            )
            self.pipeline2 = nn.Sequential(
                nn.Linear(self.ngf * 4, self.ngf),
                nn.ELU(True),
                nn.Linear(self.ngf, self.output_features),
                nn.Sigmoid()
            )

        def forward(self, x):
            if isinstance(x.data, torch.cuda.FloatTensor) and self.ngpu > 1:
                x2 = nn.parallel.data_parallel(self.pipeline1, x, range(self.main_gpu, self.main_gpu + self.ngpu))
                x2 = x2.view(-1, self.ngf * 4)
                output = nn.parallel.data_parallel(self.pipeline2, x2, range(self.main_gpu, self.main_gpu + self.ngpu))
            else:
                x2 = self.pipeline1(x)
                x2 = x2.view(-1, self.ngf * 4)
                output = self.pipeline2(x2)
            return output

class G_Net_conv_64(nn.Module):
    def __init__(self, ngpu, main_gpu=0, in_channel=118, out_channel=1):
        super(G_Net_conv_64, self).__init__()
        self.nz = 100
        self.ngf = 64
        self.ngpu = ngpu
        self.main_gpu = main_gpu
        self.in_dimension = in_channel
        self.out_channel = out_channel
        self.pipeline = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(self.in_dimension, self.ngf * 8, 4, 2, 0),
            nn.BatchNorm2d(self.ngf * 8),
            nn.ReLU(True),
            # state size. (self.ngf*8) x 4 x 4
            nn.ConvTranspose2d(self.ngf * 8, self.ngf * 4, 4, 2, 1),
            nn.BatchNorm2d(self.ngf * 4),
            nn.ReLU(True),
            # state size. (self.ngf*4) x 8 x 8
            nn.ConvTranspose2d(self.ngf * 4, self.ngf * 2, 4, 2, 1),
            nn.BatchNorm2d(self.ngf * 2),
            nn.ReLU(True),
            # state size. (self.ngf*2) x 15 x 15
            nn.ConvTranspose2d(self.ngf * 2, self.ngf, 4, 2, 1),
            nn.BatchNorm2d(self.ngf),
            nn.ReLU(True),
            # state size. (self.ngf) x 28 x 28
            nn.ConvTranspose2d(self.ngf, self.out_channel, 4, 2, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        if isinstance(x.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.pipeline, x,
                                               range(self.main_gpu, self.main_gpu + self.ngpu))
        else:
            output = self.pipeline(x)
        return output
#  for cgan series
class G_Net_conv(nn.Module):
    def __init__(self,ngpu,main_gpu=0, in_channel=110, out_channel = 3):
        super(G_Net_conv,self).__init__()
        self.nz = 100
        self.ngf = 64
        self.ngpu = ngpu
        self.main_gpu = main_gpu
        self.in_dimension = in_channel
        self.out_channel = out_channel
        self.pipeline= nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(self.in_dimension, self.ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(self.ngf * 8),
            nn.ReLU(True),
            # state size. (self.ngf*8) x 4 x 4
            nn.ConvTranspose2d(self.ngf * 8, self.ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.ngf * 4),
            nn.ReLU(True),
            # state size. (self.ngf*4) x 8 x 8
            nn.ConvTranspose2d(self.ngf * 4, self.ngf * 2, 3, 2, 1, bias=False),
            nn.BatchNorm2d(self.ngf * 2),
            nn.ReLU(True),
            # state size. (self.ngf*2) x 15 x 15
            nn.ConvTranspose2d(self.ngf * 2,out_channel, 2, 2, 1, bias=False),
            # nn.BatchNorm2d(self.ngf),
            # nn.ReLU(True),
            # # state size. (self.ngf) x 32 x 32
            # nn.ConvTranspose2d(    self.ngf,      1, 4, 2, 1, bias=False),
            nn.Sigmoid()
        )

    def forward(self,x):
        if isinstance(x.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.pipeline, x, range(self.main_gpu,self.main_gpu+self.ngpu))
        else:
            output = self.pipeline(x)

        return output




class G_Net_conv_32(nn.Module):
    def __init__(self,ngpu,main_gpu=0, in_channel=110, out_channel = 1):
        super(G_Net_conv_32,self).__init__()
        self.nz = 100
        self.ngf = 64
        self.ngpu = ngpu
        self.main_gpu = main_gpu
        self.in_dimension = in_channel
        self.out_channel = out_channel
        self.pipeline= nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(self.in_dimension, self.ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(self.ngf * 8),
            nn.ReLU(True),
            # state size. (self.ngf*8) x 4 x 4
            nn.ConvTranspose2d(self.ngf * 8, self.ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.ngf * 4),
            nn.ReLU(True),
            # state size. (self.ngf*4) x 8 x 8
            nn.ConvTranspose2d(self.ngf * 4, self.ngf * 2, 3, 2, 1, bias=False),
            nn.BatchNorm2d(self.ngf * 2),
            nn.ReLU(True),
            # state size. (self.ngf*2) x 15 x 15
            nn.ConvTranspose2d(self.ngf * 2,self.ngf, 2, 2, 1, bias=False),
            nn.BatchNorm2d(self.ngf),
            nn.ReLU(True),
            # state size. (self.ngf) x 28 x 28
            nn.ConvTranspose2d(    self.ngf, self.out_channel, 5, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self,x):
        if isinstance(x.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.pipeline, x, range(self.main_gpu,self.main_gpu+self.ngpu))
        else:
            output = self.pipeline(x)

        return output

class G_Net(nn.Module):
    def __init__(self):
        # self.c = c
        super(G_Net,self).__init__()
        self.pipeline1 = nn.Sequential(
            nn.Linear(110,256,bias=True),
            nn.ELU(),
            nn.Linear(256,1024, bias=True),
        )
        self.pipeline2 = nn.Sequential(
            nn.ELU(),
            nn.Linear(1024, 784, bias=True),
            nn.Sigmoid()
        )

    def forward(self,x):
        # output = self.main(x)
        # return output
        output1 = self.pipeline1(x)
        output = self.pipeline2(output1)
        return output1, output


class D_Net(nn.Module):
    def __init__(self):
        super(D_Net,self).__init__()
        self.main = nn.Sequential(
            nn.Linear(794,256,bias=True),
            nn.ELU(),
            nn.Linear(256, 100, bias=True),
            nn.ELU(),
            nn.Linear(100,1,bias=True),
            nn.Sigmoid()
        )

    def forward(self,x):
        output = self.main(x)
        return output

# 2017/4/28 G_net_feature_map, for fm_conv_overlay_noise.py
class G_Net_FM(nn.Module):
    def __init__(self,ngpu, input_features,main_gpu):
        super(G_Net_FM,self).__init__()
        self.main_gpu = main_gpu
        self.ngpu = ngpu
        self.main = nn.Sequential(
            nn.ConvTranspose2d(input_features,1,kernel_size=5),
            nn.Tanh()
        )
    def forward(self,x):
        if isinstance(x.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, x, range(self.main_gpu , self.main_gpu+self.ngpu))
        else:
            output = self.main(x)
        return output

class G_Net_FM_2(nn.Module):
    def __init__(self, ngpu,input_features,output_features=1,main_gpu = 0):
        super(G_Net_FM_2, self).__init__()
        self.nz = 100
        self.ngf = 128
        self.ngpu = ngpu
        self.main_gpu = main_gpu
        self.pipeline = nn.Sequential(
            # # state size. (self.ngf*4) x 8 x 8
            # nn.ConvTranspose2d(input_features, self.ngf * 2, 3, 2, 1, bias=True),
            # nn.BatchNorm2d(self.ngf * 2),
            # nn.ELU(True),
            # # state size. (self.ngf*2) x 15 x 15
            # nn.ConvTranspose2d(self.ngf * 2, output_features, 2, 2, 1, bias=True),
            # nn.Sigmoid()
            # state size. (self.ngf*4) x 8 x 8
            nn.ConvTranspose2d(input_features, self.ngf * 2, 5, 1, 0, bias=True),
            nn.BatchNorm2d(self.ngf * 2),
            nn.ELU(True),
            nn.ConvTranspose2d(self.ngf*2, self.ngf*4, 4,2,1,bias=True),
            nn.BatchNorm2d(self.ngf*4),
            # state size. (self.ngf*2) x 15 x 15
            nn.ELU(True),
            nn.ConvTranspose2d(self.ngf * 4, output_features, 5, 1, 0, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        if isinstance(x.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.pipeline, x, range(self.main_gpu, self.main_gpu+self.ngpu))
        else:
            output = self.pipeline(x)

        return output

class G_Net_FM_21(nn.Module):
    def __init__(self, ngpu, input_features,output_features=1,main_gpu = 0):
        super(G_Net_FM_21, self).__init__()
        self.nz = 100
        self.ngf = 128
        self.ngpu = ngpu
        self.main_gpu = main_gpu
        self.pipeline = nn.Sequential(
            # state size. (self.ngf*4) x 8 x 8
            nn.ConvTranspose2d(input_features, self.ngf , 5, 1, 0, bias=True),
            nn.BatchNorm2d(self.ngf ),
            nn.ELU(True),
            nn.ConvTranspose2d(self.ngf, 10, 4,2,1,bias=True),
            nn.BatchNorm2d(10)
        )

    def forward(self, x):
        if isinstance(x.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.pipeline, x, range(self.main_gpu, self.main_gpu+self.ngpu))
        else:
            output = self.pipeline(x)
        return output

class G_Net_FM_3(nn.Module):
    def __init__(self, ngpu, input_features, output_features=1, main_gpu=0):
        super(G_Net_FM_3, self).__init__()
        self.nz = 100
        self.ngf = 128
        self.ngpu = ngpu
        self.main_gpu = main_gpu
        self.pipeline = nn.Sequential(
            # # state size. (self.ngf*4) x 8 x 8
            # nn.ConvTranspose2d(input_features, self.ngf * 2, 3, 2, 1, bias=True),
            # nn.BatchNorm2d(self.ngf * 2),
            # nn.ELU(True),
            # # state size. (self.ngf*2) x 15 x 15
            # nn.ConvTranspose2d(self.ngf * 2, output_features, 2, 2, 1, bias=True),
            # nn.Sigmoid()
            # state size. (self.ngf*4) x 8 x 8
            nn.ConvTranspose2d(input_features, 128, 6, 1, 1, bias=True),
            nn.BatchNorm2d(128),
            nn.ELU(True),
            nn.ConvTranspose2d(128, 64, 4,2 ,1,bias=True),
            nn.BatchNorm2d(64),
            nn.ELU(True),
            nn.ConvTranspose2d(64, self.ngf * 2, 5, 1, 0, bias=True),
            nn.BatchNorm2d(self.ngf * 2),
            nn.ELU(True),
            nn.ConvTranspose2d(self.ngf * 2, self.ngf * 4, 4, 2, 1, bias=True),
            nn.BatchNorm2d(self.ngf * 4),
            # state size. (self.ngf*2) x 15 x 15
            nn.ELU(True),
            nn.ConvTranspose2d(self.ngf * 4, output_features, 5, 1, 0, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        if isinstance(x.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.pipeline, x, range(self.main_gpu, self.main_gpu + self.ngpu))
        else:
            output = self.pipeline(x)
        return output

class G_Net_FM_510(nn.Module):
    def __init__(self, ngpu, input_features, output_features=1, main_gpu=0):
        super(G_Net_FM_510, self).__init__()
        self.nz = 100
        self.ngf = 128
        self.ngpu = ngpu
        self.main_gpu = main_gpu
        self.pipeline = nn.Sequential(
            # # state size. (self.ngf*4) x 8 x 8
            # nn.ConvTranspose2d(input_features, self.ngf * 2, 3, 2, 1, bias=True),
            # nn.BatchNorm2d(self.ngf * 2),
            # nn.ELU(True),
            # # state size. (self.ngf*2) x 15 x 15
            # nn.ConvTranspose2d(self.ngf * 2, output_features, 2, 2, 1, bias=True),
            # nn.Sigmoid()
            # state size. (self.ngf*4) x 8 x 8
            nn.ConvTranspose2d(input_features, self.ngf * 4, 6, 1, 1, bias=True),
            nn.BatchNorm2d(self.ngf * 4),
            nn.ELU(True),
            nn.ConvTranspose2d(self.ngf * 4, self.ngf * 3, 4, 2, 1, bias=True),
            nn.BatchNorm2d(self.ngf * 3),
            nn.ELU(True),
            nn.ConvTranspose2d(self.ngf * 3, self.ngf * 2, 5, 1, 0, bias=True),
            nn.BatchNorm2d(self.ngf * 2),
            nn.ELU(True),
            nn.ConvTranspose2d(self.ngf * 2, self.ngf * 1, 4, 2, 1, bias=True),
            nn.BatchNorm2d(self.ngf * 1),
            # state size. (self.ngf*2) x 15 x 15
            nn.ELU(True),
            nn.ConvTranspose2d(self.ngf * 1, output_features, 5, 1, 0, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        if isinstance(x.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.pipeline, x, range(self.main_gpu, self.main_gpu + self.ngpu))
        else:
            output = self.pipeline(x)
        return output

class G_Net_FM_Cifar(nn.Module):
    def __init__(self, ngpu, input_features, output_features=3, main_gpu=0):
        super(G_Net_FM_Cifar, self).__init__()
        self.nz = 100
        self.ngf = 128
        self.ngpu = ngpu
        self.main_gpu = main_gpu
        self.pipeline = nn.Sequential(
            nn.ConvTranspose2d(input_features, self.ngf * 8, 4, 2, 0, bias=True),
            nn.BatchNorm2d(self.ngf * 8),
            nn.ELU(True),
            nn.ConvTranspose2d(self.ngf * 8, self.ngf * 4, 5, 2, 0, bias=True),
            nn.BatchNorm2d(self.ngf * 4),
            nn.ELU(True),
            nn.ConvTranspose2d(self.ngf * 4, self.ngf * 2, 5, 2, 0, bias=True),
            nn.BatchNorm2d(self.ngf * 2),
            nn.ELU(True),
            nn.ConvTranspose2d(self.ngf * 2, self.ngf * 1, 5, 1, 0, bias=True),
            nn.BatchNorm2d(self.ngf * 1),
            nn.ELU(True),
            nn.ConvTranspose2d(self.ngf * 1, output_features, 4, 1, 0, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        if isinstance(x.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.pipeline, x, range(self.main_gpu, self.main_gpu + self.ngpu))
        else:
            output = self.pipeline(x)
        return output
# Encoder Model
cfg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG11_2': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 256, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}
class E_Net_VGG_cifar10(nn.Module):
    def __init__(self, vgg_name='VGG11'):
        super(E_Net_VGG_cifar10, self).__init__()
        self.features = self._make_layers(cfg[vgg_name])
        self.classifier = nn.Linear(512, 10)

    def forward(self, x):
        out = self.features(x)
        out_tmp = out.view(out.size(0), -1)
        out = self.classifier(out_tmp)
        return out, out_tmp

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)


class E_Net_VGG_cifar10_64(nn.Module):
    def __init__(self, vgg_name='VGG11_2'):
        super(E_Net_VGG_cifar10_64, self).__init__()
        self.features = self._make_layers(cfg[vgg_name])
        self.classifier1 = nn.Linear(256, 64)
        self.sigmoid = nn.Sigmoid()
        self.classifier2 = nn.Linear(64,10)

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out2 = self.sigmoid(self.classifier1(out))
        out = self.classifier2(out2)
        return out,out2

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)


class D_Net_Cifar10(nn.Module):
    def __init__(self, ngpu, input_features, output_features=1, main_gpu=0):
        super(D_Net_Cifar10, self).__init__()
        self.nz = 100
        self.ngf = 128
        self.ngpu = ngpu
        self.output_features = output_features
        self.main_gpu = main_gpu
        self.pipeline1 = nn.Sequential(
            nn.Conv2d(input_features, self.ngf , 5, 2, 0, bias=True),
            nn.BatchNorm2d(self.ngf),
            nn.ELU(True),
            nn.Conv2d(self.ngf, self.ngf * 2, 5, 2, 0, bias=True),
            nn.BatchNorm2d(self.ngf * 2),
            nn.ELU(True),
            nn.Conv2d(self.ngf * 2, self.ngf * 4, 5, 2, 0, bias=True),
            nn.BatchNorm2d(self.ngf * 4),
            nn.ELU(True),
            # nn.Conv2d(self.ngf * 4, self.ngf * 8, 5, 2, 0, bias=True),
            # nn.BatchNorm2d(self.ngf * 8),
            # state size. (self.ngf*2) x 15 x 15
            # nn.ELU(True),
        )
        self.pipeline2 = nn.Sequential(
            nn.Linear(self.ngf*4, self.ngf),
            nn.ELU(True),
            nn.Linear(self.ngf,self.output_features),
            nn.Sigmoid()
        )

    def forward(self, x):
        if isinstance(x.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            x2 = nn.parallel.data_parallel(self.pipeline1, x, range(self.main_gpu, self.main_gpu + self.ngpu))
            x2 = x2.view(-1,self.ngf*4)
            output = nn.parallel.data_parallel(self.pipeline2, x2, range(self.main_gpu, self.main_gpu + self.ngpu))
        else:
            x2 = self.pipeline1(x)
            x2 = x2.view(-1,self.ngf*4)
            output = self.pipeline2(x2)
        return output

# 2017/4/28 Encoder Network for fm_conv_overlay_noise.py
class E_Net(nn.Module):
    def __init__(self):
        super(E_Net,self).__init__()
        self.conv1 = nn.Conv2d(1,10,kernel_size=5)
        self.conv2 = nn.Conv2d(10,50,kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(800,240)
        self.fc2 = nn.Linear(240,10)

    def forward(self,x):
        x1 = self.conv1(x) # batch_size * 10* 24 * 24
        x = F.relu(F.max_pool2d(x1,2))
        x = self.conv2_drop(self.conv2(x))
        x = F.relu(F.max_pool2d(x,2))
        x = x.view(-1,800) # 50*16
        x = self.fc2(F.dropout(F.relu(self.fc1(x)), training=True))
        return F.log_softmax(x),x1

class E_Net_small(nn.Module):
    def __init__(self):
        super(E_Net_small, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv3 = nn.Conv2d(20, 32, kernel_size=4)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(32, 32)
        self.fc2 = nn.Linear(32, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = F.relu(self.conv3(x))
        x1 = x.view(-1, 32)
        x2 = F.sigmoid(self.fc1(x1))
        x = F.dropout(x2, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x),x,x1,x2

class Noise_Net(nn.Module):
    def __init__(self):
        super(Noise_Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv3 = nn.Conv2d(20, 32, kernel_size=4)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(32, 64)
        self.fc2 = nn.Linear(64, 128)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = F.relu(self.conv3(x))
        x1 = x.view(-1, 32)
        x2 = F.sigmoid(self.fc1(x1))
        x = F.dropout(x2, training=self.training)
        x = self.fc2(x)
        return x

                # 2017/4/28 Encoder Network for fm_conv_overlay_noise.py
class E_Net_2(nn.Module):
    def __init__(self):
        super(E_Net_2, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 50, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(800, 240)
        self.fc2 = nn.Linear(240, 10)

    def forward(self, x):
        x1 = self.conv1(x)  # batch_size * 10* 24 * 24
        x = F.relu(F.max_pool2d(x1, 2))
        x2 = self.conv2_drop(self.conv2(x))  # batch_size * 8 * 8
        x = F.relu(F.max_pool2d(x2, 2))
        x = x.view(-1, 800)  # 50*16
        x3 = self.fc1(x)
        x = self.fc2(F.dropout(F.relu(x3), training=True))
        return F.log_softmax(x),x1, x2, x3

    #  for cgan series and fm_conv_overly_noise.py
class Ev_Net_conv(nn.Module):
    def __init__(self, ngpu, inchannel, main_gpu=0):
        super(Ev_Net_conv, self).__init__()
        self.ngpu = ngpu
        self.main_gpu = main_gpu
        self.pipeline = nn.Sequential(
            nn.Conv2d(inchannel, 48, 5),  # 28-> 24
            nn.ELU(),
            nn.MaxPool2d(2),
            nn.Conv2d(48, 128, 5),  # 12-> 8
            nn.ReLU(),
            nn.MaxPool2d(2),  # 8->4
            nn.Conv2d(128, 256, 3, 1, 0),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.pipeline2 = nn.Sequential(
            nn.Linear(256, 64),
            nn.ELU(),
            nn.Linear(64, 10),
            nn.Sigmoid()
        )

    def forward(self, x):
        if isinstance(x.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.pipeline, x, range(self.main_gpu, self.main_gpu + self.ngpu))
            output = output.view(-1, 256)
            output = nn.parallel.data_parallel(self.pipeline2, output,
                                               range(self.main_gpu, self.main_gpu + self.ngpu))
        else:
            output = self.pipeline(x)
            output = output.view(-1, 256)
            output = self.pipeline2(output)

        return output


## example
# class Combine_Net(nn.Module):
#     def __init__(self):
#         super(Combine_Net,self).__init__()
#         self.ngpu = ngpu
#         self.w1 = nn.Parameter(torch.FloatTensor([0]).float())
#         self.w2 = nn.Parameter(torch.FloatTensor([1]).float())
#     #self.w1 = self.w1.expand(28,28)
# 	#self.w2 = self.w2.expand(28,28)
#
#     def forward(self,x1,x2):
#         output = x1*self.w1.expand_as(x1) + x2*self.w2.expand_as(x2)
#         return output


"""Weight Initialization"""
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        m.weight.data= m.weight.data * 1/np.sqrt(m.in_features/2)

    # elif classname.find('BatchNorm') != -1:
    #     m.weight.data.normal_(0, 1)
    #     m.bias.data.fill_(0)


"""Label Convert"""
# For cgan_conv.py
def set_label(label_m,ms=28):
    # ms is the size of the picture 28 / 24
    label_m = np.nonzero(label_m)[1]
    # one dimension vector

    label_map = np.zeros([np.size(label_m),1, ms,ms])

    # for i, j in enumerate(label_m):
    label_map[range(0,np.size(label_m)),0,:,2*label_m+1]= 1

    return label_map

# For cgan_part_train.py for the label of labels
def set_label_ve_ma(label_m,total_size = 10):
    label_map = np.zeros([np.size(label_m),total_size])
    label_map[range(0,np.size(label_m)),label_m] = 1

    return label_map


# For script cgan_conv_5d.py
def set_label_ve(label_m, ms=28):
    # one dimension vector
    label_map = np.zeros([np.size(label_m),1, ms,ms]).astype('int')
    # for i, j in enumerate(label_m):
    label_map[range(0,np.size(label_m)),0,:,2*np.array(label_m)+1]= 1

    return label_map

# For script cgan_conv_5d.py
def set_label_ve_b(label_m, batch_size=50,ms=28):
    # one dimension vector
    label_map = np.zeros([batch_size,1, ms,ms]).astype('int')

    # for i, j in enumerate(label_m):
    label_map[range(0,batch_size),0,:,2*np.array(label_m)+1]= 1

    return label_map


def set_label_ma_3(label_m, ms):
    a = np.array([[0, 1, 2, 3], [4, 5, 6], [7, 8, 9]])
    b = np.array([[1, 5, 7], [2, 5, 8, 0], [3, 6, 9]])
    c = np.array([[1, 6, 8], [2, 4, 9], [3, 5, 0, 7]])

    label_map_list = np.zeros([10, 3])

    for i in range(10):
        for k, vector in enumerate([a, b, c]):
            for j in range(3):
                if (i in vector[j]):
                    label_map_list[i][k] = j
                    # break

    label_list = np.zeros([3,np.size(label_m),1,ms,ms]).astype('int')

    for i in range(3):
        map_list = np.array([label_map_list[num][i] for _,num in enumerate(label_m)])
        label_list[i] = set_label_ve(map_list.astype('int'),ms)

    return label_list

def set_label_f3(label_m,batch_size=50,ms = 28):
    label_map = torch.zeros([batch_size,1, ms,ms])
    for i in range(0,batch_size):
        aa = (label_m[i])
        aa = aa.resize(20,12)
        label_map[i,0,0:20,0:12]= aa.data

    return label_map

def set_label_cifar(label_m, batch_size = 100, ms = 32):
    label_map = torch.zeros([batch_size,1,ms,ms])
    for i in range(0,batch_size):
        aa = label_m[i]
        row_label, column_label = aa % 4, aa // 4
        label_map[i,0,row_label*8:(row_label+1)*8,column_label]=1
    return label_map

def set_condition(condition,channel,batch_size ):
    # batch_size = np.shape(condition.data.tolist())[0]
    # if(isinstance(channel,int)):
    #     ch = 1
    # else:
    #     ch = np.size(channel)
    shape = [batch_size,channel,28,28]
    c_v = torch.zeros(shape)

    c_v[0:batch_size,0:channel,0:24,0:24] = condition[:,0:channel,:,:]
    c_v = Variable(torch.FloatTensor(c_v)).cuda()#
    return c_v

def get_feature(f1,ch,batch_size):
    f1 = f1[:,0:ch,:,:]
    # f1.data.resize_(batch_size, ch, 24, 24) # print(np.shape(owntool.extract(f1)))
    g_f1 = f1.cuda()
    return  g_f1

def set_label_celebA(c,batch_size, image_size=64):
    c_list = torch.nonzero(c)
    label_m = torch.zeros([batch_size, 1,image_size,image_size])
    # assert batch_size==label_m.shape[0]
    for i in range(c_list.size()[0]):
        label_m[c_list[i,0], 0,:, c_list[i,1]] = 1
    return  label_m
