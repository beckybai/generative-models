import torch
import torch.autograd as autograd
import torch.optim as optim
import numpy as np
import matplotlib as mpl
from datetime import datetime
import owntool

mpl.use('Agg')

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
from torch.autograd import Variable
from tensorflow.examples.tutorials.mnist import input_data
import torch.nn as nn
import torch.nn.functional as F
import shutil,sys
import mutil
import model
import data_convert

gpu = 2
ngpu = 1

torch.cuda.set_device(gpu)
mnist = input_data.read_data_sets('../../MNIST_data', one_hot=True)
mb_size = 64 # mini-batch_size
Z_dim = 100
X_dim = mnist.train.images.shape[1]
y_dim = mnist.train.labels.shape[1]
h_dim = 128
cnt = 0
mm = data_convert.owndata()

num = '0'
out_dir = 'out_fc_{}_{}/'.format(datetime.now(),num)
if not os.path.exists(out_dir):
    os.makedirs(out_dir)
    shutil.copyfile(sys.argv[0], out_dir + '/EVALUATOR_2_group_LOSS.py')

c_label = np.zeros(shape=[mb_size, y_dim], dtype='float32')

for i in range(0, 60, 6):
    c_label[i:(i + 6), i / 6] = 1.

sys.stdout = mutil.Logger(out_dir)
# else:
#     print("you have already creat one.")
#     exit(1)

#
#
# def xavier_init(size):
#     in_dim = size[0]
#     xavier_stddev = 1. / np.sqrt(in_dim / 2.)
#     return Variable(torch.randn(*size) * xavier_stddev, requires_grad=True)
in_channel=2
G = model.G_Net_conv(ngpu).cuda()
D = model.D_Net_conv(ngpu,1).cuda()
E = model.D_Net_conv(ngpu,2).cuda()

"""Weight Initialization"""


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    # elif classname.find('BatchNorm') != -1:
    #     m.weight.data.normal_(1.0, 0.02)
    #     m.bias.data.fill_(0)



G.apply(weights_init)
D.apply(weights_init)
E.apply(weights_init)

""" ===================== TRAINING ======================== """


G_solver = optim.Adam(G.parameters(), lr=1e-4)
D_solver = optim.Adam(D.parameters(), lr=1e-4)
E_solver = optim.Adam(E.parameters(), lr=1e-4)

ones_label = Variable(torch.ones(mb_size)).cuda()
zeros_label = Variable(torch.zeros(mb_size)).cuda()


criterion = nn.BCELoss()
criterion_r = nn.MarginRankingLoss(margin=0.1,size_average=False)

for it in range(100000):

    # Sample data
    z = Variable(torch.randn(mb_size, Z_dim)).cuda()
    X, c = mm.batch_next(mb_size)
    X = Variable(torch.from_numpy(X)).cuda()

    c_v = Variable(torch.from_numpy(model.set_label_ve_ma(c).astype('float32'))).cuda() # label for g c
    c_t = Variable(torch.from_numpy(model.set_label_ve(c).astype('float32'))).cuda() # label for d c(true)

    # Dicriminator forward-loss-backward-update
    D.zero_grad()
    G.zero_grad()
# For D
    x_g = torch.cat([z,c_v],1).t()
    x_g.data.resize_(mb_size, Z_dim+10, 1, 1)

    G_sample = G(x_g).detach()
    X.data.resize_(mb_size, 1, 28, 28)
    D_real = D(X)
    D_fake = D(G_sample)

    D_loss_real = criterion(D_real, ones_label)
    D_loss_fake = criterion(D_fake, zeros_label)
    D_loss = D_loss_real + D_loss_fake

    D_loss.backward()
    D_solver.step()

    # For G
    # Generator forward-loss-backward-update
    z = Variable(torch.randn(mb_size, Z_dim)).cuda()
    x_g = torch.cat([z, c_v], 1).t()
    x_g.data.resize_(mb_size, Z_dim + 10, 1, 1)

    G_sample = G(x_g)
    D_fake = D(G_sample)

    G_loss = criterion(D_fake, ones_label)

    G_loss.backward()
    G_solver.step()

    # Housekeeping - reset gradient
    D.zero_grad()
    G.zero_grad()

    # For E ( real data )
    E.zero_grad()
    G.zero_grad()
    ER_real = E(torch.cat([X,c_t],1))
    random_label = (np.random.rand(mb_size)*9+1).astype('int')
    c_f =  Variable(torch.from_numpy(model.set_label_ve((c+random_label)%10).astype('float32'))).cuda()
    ER_fake = E(torch.cat([X,c_f],1))
    # E_loss1 =  F.relu((ER_real-ER_fake)).sum()
    E_loss1 = criterion_r(ER_fake,ER_real,ones_label)
    # E_loss1 = criterion(ER_real,ones_label) + criterion(ER_fake,zeros_label)
    E_loss1.backward()
    E_solver.step()


# For E ( generated data )
    z = Variable(torch.randn(mb_size, Z_dim)).cuda()
    E.zero_grad()
    x_g = torch.cat([z,c_v],1).t()
    x_g.data.resize_(mb_size, Z_dim+10, 1, 1)
    G_sample =  G(x_g)
    E_real = E(torch.cat([G_sample,c_t],1))
    random_label = (np.random.rand(mb_size)*9 + 1).astype('int')
    c_f =  Variable(torch.from_numpy(model.set_label_ve((c+random_label)%10).astype('float32'))).cuda()
    E_fake = E(torch.cat([G_sample,c_f],1))
    # E_loss_real = criterion(E_real, ones_label)
    # E_loss_fake = criterion(E_fake, zeros_label)
    # E_loss2 = E_loss_real + E_loss_real
    # E_loss2 = F.relu(E_real - E_fake ).sum()
    E_loss2 = criterion_r(E_fake, E_real,ones_label)
    E_loss2.backward()
    # E_solver.step()
    G_solver.step()

    # Housekeeping - reset gradient
    D.zero_grad()
    G.zero_grad()
    E.zero_grad()


    # Print and plot every now and then
    if it % 500 == 0:
        print('Iter-{}; D_loss_real/fake: {}/{}; G_loss: {},E_loss:T:{}/F:{}'.format(it, D_loss_real.data.tolist(),D_loss_fake.data.tolist(), G_loss.data.tolist(), E_loss1.data.tolist(),E_loss2.data.tolist()))
        c = c_label
        c_v = Variable(torch.from_numpy(c.astype('float32'))).cuda()
        x_g = torch.cat([z, c_v], 1).t()
        x_g.data.resize_(mb_size, Z_dim + 10, 1, 1)
        samples = G(x_g)

        out_pic_dir = "{}{}.png".format(out_dir,it)
        owntool.save_picture(samples, out_pic_dir,image_size=28, column = 6)

    if it % 5000 == 0:
        torch.save(D.state_dict(),'{}/D_{}.model'.format(out_dir,it))
        torch.save(G.state_dict(),'{}/G_{}.model'.format(out_dir,it))
        torch.save(E.state_dict(),'{}/E_{}.model'.format(out_dir,it))