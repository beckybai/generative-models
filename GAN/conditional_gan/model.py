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
from tensorflow.examples.tutorials.mnist import input_data
import torch.nn as nn
import torch.nn.functional as F
import shutil,sys



class D_Net_conv(nn.Module):
    def __init__(self,ngpu,inchannel):
        super(D_Net_conv,self).__init__()
        self.ngpu = ngpu
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
            output = nn.parallel.data_parallel(self.pipeline, x, range(self.ngpu))
            output = output.view(-1, 256)
            output = nn.parallel.data_parallel(self.pipeline2, output, range(self.ngpu))
        else:
            output = self.pipeline(x)
            output = output.view(-1, 256)
            output = self.pipeline2(output)

        return output

class G_Net_conv(nn.Module):
    def __init__(self,ngpu):
        super(G_Net_conv,self).__init__()
        self.nz = 110
        self.ngf = 64
        self.ngpu = ngpu
        self.pipeline= nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(     self.nz, self.ngf * 8, 4, 1, 0, bias=False),
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
            nn.ConvTranspose2d(self.ngf * 2,     1, 2, 2, 1, bias=False),
            # nn.BatchNorm2d(self.ngf),
            # nn.ReLU(True),
            # # state size. (self.ngf) x 32 x 32
            # nn.ConvTranspose2d(    self.ngf,      1, 4, 2, 1, bias=False),
            nn.Sigmoid()
        )

    def forward(self,x):
        if isinstance(x.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.pipeline, x, range(self.ngpu))
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
        # self.main = nn.Sequential(
        #     nn.Linear(110,256,bias=True),
        #     nn.BatchNorm1d(256),
        #     nn.Linear(256,512, bias=True),
        #     nn.ReLU(),
        #     nn.Linear(512,784,bias=True),
        #     nn.Sigmoid()
        # )
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


"""Weight Initialization"""

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        m.weight.data= m.weight.data * 1/np.sqrt(m.in_features/2)

    # elif classname.find('BatchNorm') != -1:
    #     m.weight.data.normal_(0, 1)
    #     m.bias.data.fill_(0)


def set_label(label_m):
    label_m = np.nonzero(label_m)[1]
    # one dimension vector

    label_map = np.zeros([np.size(label_m),1, 28,28])

    # for i, j in enumerate(label_m):
    label_map[range(0,np.size(label_m)),0,:,2*label_m+1]= 1

    return label_map


def set_label_5d(label_m):
    # one dimension vector
    label_map = np.zeros([np.size(label_m),1, 28,28])

    # for i, j in enumerate(label_m):
    label_map[range(0,np.size(label_m)),0,:,2*label_m+1]= 1

    return label_map


