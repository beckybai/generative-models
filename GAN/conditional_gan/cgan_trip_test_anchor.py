import torch
import torch.autograd as autograd
import torch.optim as optim
import numpy as np
import matplotlib as mpl
from datetime import datetime
import owntool
from decimal import Decimal


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

main_gpu = 7
ngpu = 1

torch.cuda.set_device(main_gpu)
mnist = input_data.read_data_sets('../../MNIST_data', one_hot=True)
mb_size = 60 # mini-batch_size
Z_dim = 100
X_dim = mnist.train.images.shape[1]
y_dim = mnist.train.labels.shape[1]
h_dim = 128
cnt = 0
mm = data_convert.owndata()

num = '0'
out_dir = 'triplet_mnist10_{}_{}/'.format(datetime.now(),num)
if not os.path.exists(out_dir):
    os.makedirs(out_dir)
    shutil.copyfile(sys.argv[0], out_dir + '/tripnet_10dimension_e_p2.py')

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
E = model.Ev_Net_conv(ngpu,1).cuda()

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
half_label = Variable(torch.ones(mb_size)*0.5).cuda()

criterion = nn.BCELoss()
# criterion_r = nn.MarginRankingLoss(margin=0.1,size_average=False)
criterion_t = nn.TripletMarginLoss(p=1)
criterion_mse = nn.MSELoss()

for it in range(100000):

    # Sample data
    z = Variable(torch.randn(mb_size, Z_dim)).cuda()
    X, c = mm.batch_next(mb_size)
    X = Variable(torch.from_numpy(X)).cuda()

    c_v = Variable(torch.from_numpy(model.set_label_ve_ma(c).astype('float32'))).cuda() # label for g c
    # c_t = Variable(torch.from_numpy(model.set_label_ve(c).astype('float32'))).cuda() # label for d c(true)

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
    D_fake_real = D(X/2+G_sample/2)

    D_loss_fr = criterion_mse(D_fake_real,half_label)
    D_loss_real = criterion(D_real, ones_label)
    D_loss_fake = criterion(D_fake, zeros_label)
    D_loss = D_loss_real + D_loss_fake + D_loss_fr

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

# For E, triplet part
#     z = Variable(torch.randn(mb_size, Z_dim)).cuda()
    E.zero_grad()
    # x_g = torch.cat([z,c_v],1).t()
    # x_g.data.resize_(mb_size, Z_dim+10, 1, 1)
    # G_sample =  G(x_g)
    # E_real = E(G_sample)


    # genereted (another kind of data)
    # random_label = model.set_label_ve_ma((c+(np.random.rand(mb_size)*9 + 1).astype('int'))%10)
    # random_label = Variable(torch.from_numpy(random_label.astype('float32'))).cuda() # label for g c
    # x_g2 = torch.cat([z,random_label],1).t()
    # x_g2.data.resize_(mb_size, Z_dim+10,1,1)
    # G_sample2 = G(x_g2)
    # E_fake = E(G_sample2)
    X, c = mm.batch_next(mb_size, shuffle=False)
    X = Variable(torch.from_numpy(X.astype('float32'))).cuda()
    X.data.resize_(mb_size, 1, 28, 28)
    E_anch = E(X)

    X2, c2 = mm.batch_next(mb_size, shuffle=False)
    X2 = Variable(torch.from_numpy(X2.astype('float32'))).cuda()
    X2.data.resize_(mb_size, 1, 28, 28)
    E_real = E(X2)

    random_label = model.set_label_ve_ma((c.astype('int')+(np.random.rand(mb_size)*9 + 1).astype('int'))%10)
    random_label = Variable(torch.from_numpy(random_label.astype('float32'))).cuda() # label for g c
    X3, c3 = mm.batch_next(mb_size,label =[2,3,4,5,6,7,8,9,0,1], shuffle=False)
    X3 = Variable(torch.from_numpy(X3.astype('float32'))).cuda()
    # X3.data.resize_(mb_size, 1, 28, 28)
    E_fake = E(X3)

    E_loss = criterion_t(E_anch,E_real,E_fake)
    E_loss.backward()
    E_solver.step()

    # Housekeeping - reset gradient
    D.zero_grad()
    G.zero_grad()
    E.zero_grad()


# For E ( generated data )
    z = Variable(torch.randn(mb_size, Z_dim)).cuda()
    c_v = Variable(torch.from_numpy(model.set_label_ve_ma(c.astype('int')).astype('float32'))).cuda() # label for g c
    x_g = torch.cat([z,c_v],1).t()
    x_g.data.resize_(mb_size, Z_dim+10, 1, 1)
    G_sample =  G(x_g)
    E_real = E(G_sample)

    # genereted (another kind of data)
    random_label = model.set_label_ve_ma((c.astype('int')+(np.random.rand(mb_size)*9 + 1).astype('int'))%10)
    random_label = Variable(torch.from_numpy(random_label.astype('float32'))).cuda() # label for g c
    x_g2 = torch.cat([z,random_label],1).t()
    x_g2.data.resize_(mb_size, Z_dim+10,1,1)
    G_sample2 = G(x_g2)
    E_fake = E(G_sample2)

    E_anch = E(X)
    E_loss = criterion_t(E_anch,E_real,E_fake)
    E_loss.backward()
    # E_solver.step()
    G_solver.step()

    # Housekeeping - reset gradient
    D.zero_grad()
    G.zero_grad()
    E.zero_grad()

    # Print and plot every now and then
    if it % 500 == 0:
        print('Iter-{}; D_loss_real/fake: {}/{}; G_loss: {},E_loss:T:{}'.format(it, D_loss_real.data.tolist(),D_loss_fake.data.tolist(), G_loss.data.tolist(), E_loss.data.tolist()))
        # print("anchor\t:{}".format(E_anch[0:3]))
        # print("real\t:{}".format(E_real[0:3]))
        # print("fake\t:{}".format(E_fake[0:3]))
        c = c_label
        c_v = Variable(torch.from_numpy(c.astype('float32'))).cuda()
        x_g = torch.cat([z, c_v], 1).t()
        x_g.data.resize_(mb_size, Z_dim + 10, 1, 1)
        samples = G(x_g)


        out_pic_dir = "{}{}.png".format(out_dir,it)
        owntool.save_picture(samples[:60], out_pic_dir,image_size=28,column = 6)

    if it % 5000 == 0:
        X,c = mm.batch_next(20,shuffle=False)
        X = Variable(torch.from_numpy(X.astype('float32'))).cuda()
        X.data.resize_(20, 1, 28, 28)
        E_anch = E(X)
        c_label_tmp = c.astype('int')
        random_label = model.set_label_ve_ma((c_label_tmp + (np.random.rand(20) * 9 + 1).astype('int')) % 10)
        c_label_tmp = Variable(torch.from_numpy(model.set_label_ve_ma(c_label_tmp).astype('float32'))).cuda()
        random_label = Variable(torch.from_numpy(random_label.astype('float32'))).cuda()

        z = Variable(torch.randn(20, Z_dim)).cuda()

        x_g1 = torch.cat([z, c_label_tmp],1).t()
        x_g1.data.resize_(20, Z_dim+10,1,1)
        G_sample = G(x_g1)
        E_real = E(G_sample)

        x_g2 = torch.cat([z, random_label], 1).t()
        x_g2.data.resize_(20, Z_dim + 10, 1, 1)
        G_sample2 = G(x_g2)
        E_fake = E(G_sample2)
        # *** I am K K B
        # *** Why are you still a dog
        # np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})
        np.set_printoptions(precision=2,suppress=True)
        print("real")
        print( np.array(E_real.data.tolist()))
        print("anch")
        print( np.array(E_anch.data.tolist()))
        print("fake")
        print( np.array(E_fake.data.tolist()))
        torch.save(D.state_dict(),'{}/D_{}.model'.format(out_dir,it))
        torch.save(G.state_dict(),'{}/G_{}.model'.format(out_dir,it))
        torch.save(E.state_dict(),'{}/E_{}.model'.format(out_dir,it))