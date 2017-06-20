import torch
import torch.autograd as autograd
import torch.optim as optim
import numpy as np
import torch.nn as nn
import torch.functional as F


class G_Net_conv(nn.Module):
    def __init__(self,main_gpu=0, in_channel=100, out_channel = 1):
        super(G_Net_conv,self).__init__()
        self.nz = 100
        self.ngf = 64
        self.ngpu = 1
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

#  for cgan series and fm_conv_overly_noise.py
class D_Net_conv(nn.Module):
    def __init__(self,inchannel,main_gpu=0):
        super(D_Net_conv,self).__init__()
        # self.ngpu = ngpu
        self.ngpu = 1
        self.main_gpu = main_gpu
        self.ngf = 64
        self.pipeline = nn.Sequential(
            nn.Conv2d(inchannel,self.ngf,5), # 28-> 24
            nn.ELU(),
            nn.MaxPool2d(2),
            nn.Conv2d(self.ngf,self.ngf*2,5), # 12-> 8
            nn.ReLU(),
            nn.MaxPool2d(2), # 8->4
            nn.Conv2d(self.ngf*2,self.ngf*4,3,1,0),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.pipeline2 = nn.Sequential(
            nn.Linear(self.ngf*4,self.ngf),
            nn.ELU(),
            nn.Linear(self.ngf,1),
            nn.Sigmoid()
        )
    def forward(self,x):
        if isinstance(x.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.pipeline, x, range(self.main_gpu, self.main_gpu+self.ngpu))
            output = output.view(-1, self.ngf*4)
            output = nn.parallel.data_parallel(self.pipeline2, output, range(self.main_gpu,self.main_gpu+self.ngpu))
        else:
            output = self.pipeline(x)
            output = output.view(-1, self.ngf*4)
            output = self.pipeline2(output)

        return output
