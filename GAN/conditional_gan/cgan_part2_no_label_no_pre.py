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
import mutil
import model
import data_convert
import random


gpu = 6
ngpu = 1

torch.cuda.set_device(gpu)
mnist = input_data.read_data_sets('../../MNIST_data', one_hot=True)
mb_size = 70 # mini-batch_size
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
    shutil.copyfile(sys.argv[0], out_dir + '/training_script.py')
# else:
#     print("you have already creat one.")
#     exit(1)
sys.stdout = mutil.Logger(out_dir)#


in_channel=1
G = model.G_Net_conv(ngpu).cuda()
D = model.D_Net_conv(ngpu,in_channel).cuda()

"""Weight Initialization"""


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    # elif classname.find('BatchNorm') != -1:
    #     m.weight.data.normal_(1.0, 0.02)
    #     m.bias.data.fill_(0)



G.apply(weights_init)
G.load_state_dict(torch.load('./out_fc_2017-04-26 23:27:27.269692_0/G_30000.model'))
D.apply(weights_init)
# Dout_fc_2017-04-26 23:27:27.269692_0
# G.load_state_dict(torch.load('./out_fc_2017-04-26 23:27:27.269692_0/D_30000.model'))

""" ===================== TRAINING ======================== """


G_solver = optim.Adam(G.parameters(), lr=1e-4)
D_solver = optim.Adam(D.parameters(), lr=1e-4)

ones_label = Variable(torch.ones(mb_size)).cuda()
zeros_label = Variable(torch.zeros(mb_size)).cuda()

criterion = nn.BCELoss()
label_select = np.array([5,6,7,8,9])
label_size = 5

for it in range(100000):

    # Sample data
    z = Variable(torch.randn(mb_size, Z_dim,1,1)).cuda()
    X, c = mm.batch_next(mb_size,label_select)
    X = Variable(torch.from_numpy(X.astype('float32'))).cuda()

    # c_v = Variable(torch.from_numpy(model.set_label_ve_ma(c.astype('int')).astype('float32'))).cuda()
    # c = Variable(torch.from_numpy(model.set_label_ve(c.astype('int')).astype('float32'))).cuda()

    # Dicriminator forward-loss-backward-update
    D.zero_grad()
    G.zero_grad()
    #
    # z.data.resize_(mb_size, Z_dim, 1, 1)

    G_sample = G(z)
    X.data.resize_(mb_size, 1, 28, 28)
    D_real = D(X)
    D_fake = D(G_sample)

    D_loss_real = criterion(D_real, ones_label)
    D_loss_fake = criterion(D_fake, zeros_label)
    D_loss = D_loss_real + D_loss_fake

    D_loss.backward()
    D_solver.step()

    # Housekeeping - reset gradient
    D.zero_grad()
    G.zero_grad()

    # Generator forward-loss-backward-update
    z = Variable(torch.randn(mb_size, Z_dim,1,1)).cuda()


    G_sample =  G(z)
    D_fake = D(G_sample)

    G_loss = criterion(D_fake, ones_label)

    G_loss.backward()
    G_solver.step()

    # Housekeeping - reset gradient
    D.zero_grad()
    G.zero_grad()

    # Print and plot every now and then
    if it % 2000 == 0:
        print('Iter-{}; D_loss_real/fake: {}/{}; G_loss: {}'.format(it, D_loss_real.data.tolist(),D_loss_fake.data.tolist(), G_loss.data.tolist()))
        # c = np.zeros(shape=[mb_size, y_dim], dtype='float32')
        # c = [int(10 * random.random()) for i in range(mb_size)]
        # for i,label_l in enumerate(label_select):
        #     c[i*12:(i+1)*12] = label_l*np.ones(12)
        #
        # c_v = Variable(torch.from_numpy(model.set_label_ve_ma(c).astype('float32'))).cuda()

        # c = Variable(torch.from_numpy(model.set_label(c).astype('float32'))).cuda()\
        z = Variable(torch.randn(mb_size, Z_dim,1,1)).cuda()
        samples = G(z)
        samples = samples.data.tolist()[:60]

        fig = plt.figure(figsize=(6, 10))
        gs = gridspec.GridSpec(10, 6)
        gs.update(wspace=0.05, hspace=0.05)

        for i, sample in enumerate(samples):
            ax = plt.subplot(gs[i])
            plt.axis('off')
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.set_aspect('equal')
            plt.imshow((np.array(sample)).reshape(28, 28), cmap='Greys_r')

        plt.savefig('{}/{}.png'.format(out_dir, str(cnt).zfill(3)), bbox_inches='tight')

        cnt += 1
        plt.close(fig)

        # for i in range(d_num):
        #     torch.save(D_list[i].state_dict(),'{}/D{}.model'.format(out_dir,i))
        # torch.save(G.state_dict(),'{}/G.model'.format(out_dir))

    if it % 10000 == 0:
        torch.save(G.state_dict(), '{}/G_{}.model'.format(out_dir, str(it)))
        torch.save(D.state_dict(), '{}/D_{}.model'.format(out_dir, str(it)))
