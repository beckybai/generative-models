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

sys.stdout = mutil.Logger()
gpu = 1
ngpu = 1

torch.cuda.set_device(gpu)
mnist = input_data.read_data_sets('../../MNIST_data', one_hot=True)
mb_size = 64 # mini-batch_size
Z_dim = 100
X_dim = mnist.train.images.shape[1]
y_dim = mnist.train.labels.shape[1]
h_dim = 128
cnt = 0

num = '0'
out_dir = 'out_fc_{}_{}/'.format(datetime.now(),num)
if not os.path.exists(out_dir):
    os.makedirs(out_dir)
    shutil.copyfile(sys.argv[0], out_dir + '/training_script.py')
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
D.apply(weights_init)

""" ===================== TRAINING ======================== """


G_solver = optim.Adam(G.parameters(), lr=1e-4)
D_solver = optim.Adam(D.parameters(), lr=1e-4)

ones_label = Variable(torch.ones(mb_size)).cuda()
zeros_label = Variable(torch.zeros(mb_size)).cuda()

criterion = nn.BCELoss()

for it in range(100000):

    # Sample data
    z = Variable(torch.randn(mb_size, Z_dim)).cuda()
    X, c = mnist.train.next_batch(mb_size)
    X = Variable(torch.from_numpy(X)).cuda()
    c = c%5

    c_v = Variable(torch.from_numpy(c.astype('float32'))).cuda()
    c = Variable(torch.from_numpy(model.set_label(c).astype('float32'))).cuda()

    # Dicriminator forward-loss-backward-update
    D.zero_grad()
    G.zero_grad()

    x_g = torch.cat([z,c_v],1).t()
    x_g.data.resize_(mb_size, Z_dim+10, 1, 1)

    G_sample = G(x_g)
    X.data.resize_(mb_size, 1, 28, 28)
    D_real = D(torch.cat([X,c],1))
    D_fake = D(torch.cat([G_sample, c],1))

    D_loss_real = criterion(D_real, ones_label)
    D_loss_fake = criterion(D_fake, zeros_label)
    D_loss = D_loss_real + D_loss_fake

    D_loss.backward()
    D_solver.step()

    # Housekeeping - reset gradient
    D.zero_grad()
    G.zero_grad()

    # Generator forward-loss-backward-update
    z = Variable(torch.randn(mb_size, Z_dim)).cuda()

    x_g = torch.cat([z,c_v],1).t()
    x_g.data.resize_(mb_size, Z_dim+10, 1, 1)

    G_sample =  G(x_g)
    D_fake = D(torch.cat([G_sample, c],1))

    G_loss = criterion(D_fake, ones_label)

    G_loss.backward()
    G_solver.step()

    # Housekeeping - reset gradient
    D.zero_grad()
    G.zero_grad()

    # Print and plot every now and then
    if it % 3000 == 0:
        print('Iter-{}; D_loss_real/fake: {}/{}; G_loss: {}'.format(it, D_loss_real.data.tolist(),D_loss_fake.data.tolist(), G_loss.data.tolist()))
        c = np.zeros(shape=[mb_size, y_dim], dtype='float32')
        c[:, np.random.randint(0, 10)] = 1.
        c = c%5
        c_v = Variable(torch.from_numpy(c.astype('float32'))).cuda()
        x_g = torch.cat([z, c_v], 1).t()
        x_g.data.resize_(mb_size, Z_dim + 10, 1, 1)

        # c = Variable(torch.from_numpy(model.set_label(c).astype('float32'))).cuda()
        samples = G(x_g)


        samples = samples.data.tolist()[:16]
        # pro_sample = np.average(pro_samples.data.tolist(),0)


        fig = plt.figure(figsize=(4, 4))
        gs = gridspec.GridSpec(4, 4)
        gs.update(wspace=0.05, hspace=0.05)

        for i, sample in enumerate(samples):
            ax = plt.subplot(gs[i])
            plt.axis('off')
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.set_aspect('equal')
            plt.imshow((np.array(sample)).reshape(28, 28), cmap='Greys_r')

        # ax = plt.subplot(gs[i])
        # plt.axis('on')

        # plt.imshow(pro_sample.reshape(32,32), cmap= 'Greys_r')
        # pro_sample, t2 = np.histogram(pro_sample, normed=True)
        #
        plt.savefig('{}/{}.png'.format(out_dir,str(cnt).zfill(3)), bbox_inches='tight')
        # fig = plt.figure()
        # gs = fig.add_subplot()

        # t2 = (t2[:-1] + t2[1:]) / 2
        # plt.plot(t2, pro_sample)  # 绘制统计所得到的概率密度
        # plt.show()
        # plt.savefig('{}/hehe_{}.png'.format(out_dir,str(cnt).zfill(3)), bbox_inches='tight')


        cnt += 1
        plt.close(fig)

        torch.save(D.state_dict(),'{}/D_{}.model'.format(out_dir,it))
        torch.save(G.state_dict(),'{}/G_{}.model'.format(out_dir,it))