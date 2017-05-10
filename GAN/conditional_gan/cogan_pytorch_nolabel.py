import torch
import torch.autograd as autograd
import torch.optim as optim
import numpy as np
import matplotlib as mpl
mpl.use('Agg')

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
from torch.autograd import Variable
from tensorflow.examples.tutorials.mnist import input_data
import torch.nn as nn
import torch.nn.functional as F


gpu = 2

torch.cuda.set_device(gpu)
mnist = input_data.read_data_sets('../../MNIST_data', one_hot=True)
mb_size = 64 # mini-batch_size
Z_dim = 100
X_dim = mnist.train.images.shape[1]
y_dim = mnist.train.labels.shape[1]
h_dim = 128
cnt = 0
lr = 1e-3

num = '2'
out_dir = 'out_fc_nolabel{}/'.format(num)
if not os.path.exists(out_dir):
    os.makedirs(out_dir)
# else:
#     print("you have already creat one.")
#     exit(1)



def xavier_init(size):
    in_dim = size[0]
    xavier_stddev = 1. / np.sqrt(in_dim / 2.)
    return Variable(torch.randn(*size) * xavier_stddev, requires_grad=True)


""" ==================== GENERATOR ======================== """

class G_Net(nn.Module):
    def __init__(self):
        # self.c = c
        super(G_Net,self).__init__()
        self.main = nn.Sequential(
            nn.Linear(110,256,bias=True),
            nn.ReLU(),

            nn.Linear(256,784,bias=True),
            nn.Sigmoid()
        )

    def forward(self,x):
        output = self.main(x)
        return output

""" ==================== DISCRIMINATOR ======================== """

Wxh = xavier_init(size=[X_dim + y_dim, h_dim])
bxh = Variable(torch.zeros(h_dim), requires_grad=True)

Why = xavier_init(size=[h_dim, 1])
bhy = Variable(torch.zeros(1), requires_grad=True)

class D_Net(nn.Module):
    def __init__(self):
        super(D_Net,self).__init__()
        self.main = nn.Sequential(
            nn.Linear(784,100,bias=True),
            nn.ReLU(),

            nn.Linear(100,1,bias=True),
            nn.Sigmoid()
        )

    def forward(self,x):
        output = self.main(x)
        return output

G = G_Net().cuda()
D = D_Net().cuda()

"""Weight Initialization"""

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        m.weight.data= m.weight.data * 1/np.sqrt(m.in_features/2)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

G.apply(weights_init)
D.apply(weights_init)

""" ===================== TRAINING ======================== """


G_solver = optim.Adam(G.parameters(), lr=1e-3)
D_solver = optim.Adam(D.parameters(), lr=1e-3)

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
    G_sample = G(torch.cat([z,c],1))
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
    z = Variable(torch.randn(mb_size, Z_dim)).cuda()
    G_sample =  G(torch.cat([z,c],1))
    D_fake = D(G_sample)

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
        samples = G(torch.cat([z, c],1)).data.tolist()[:16]

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


        plt.savefig('{}/{}.png'.format(out_dir,str(cnt).zfill(3)), bbox_inches='tight')
        cnt += 1
        plt.close(fig)


torch.save(D.state_dict(),'{}/D.model'.format(out_dir))
torch.save(G.state_dict(),'{}/G.model'.format(out_dir))