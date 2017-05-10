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

G = model.G_Net().cuda()
D = model.D_Net().cuda()

G.apply(model.weights_init)
D.apply(model.weights_init)

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
    c = Variable(torch.from_numpy(c.astype('float32'))).cuda()

    # Dicriminator forward-loss-backward-update
    _,G_sample = G(torch.cat([z,c],1))
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
    _, G_sample =  G(torch.cat([z,c],1))
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
        c = Variable(torch.from_numpy(c)).cuda()
        [pro_samples,samples] = G(torch.cat([z, c],1))
        samples = samples.data.tolist()[:16]
        pro_sample = np.average(pro_samples.data.tolist(),0)


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

        ax = plt.subplot(gs[i])
        plt.axis('on')

        # plt.imshow(pro_sample.reshape(32,32), cmap= 'Greys_r')
        pro_sample, t2 = np.histogram(pro_sample, normed=True)

        plt.savefig('{}/{}.png'.format(out_dir,str(cnt).zfill(3)), bbox_inches='tight')
        fig = plt.figure()
        gs = fig.add_subplot()

        t2 = (t2[:-1] + t2[1:]) / 2
        plt.plot(t2, pro_sample)  # 绘制统计所得到的概率密度
        plt.show()
        plt.savefig('{}/hehe_{}.png'.format(out_dir,str(cnt).zfill(3)), bbox_inches='tight')


        cnt += 1
        plt.close(fig)
