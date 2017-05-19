import torch
import torch.autograd as autograd
import torch.optim as optim
import numpy as np
import matplotlib as mpl
from datetime import datetime

# mpl.use('Agg')

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
from torch.autograd import Variable
from tensorflow.examples.tutorials.mnist import input_data
import torch.nn as nn
import torch.nn.functional as F
import shutil,sys
import mutil
import toy_model as model
import data_prepare


out_dir = './out/gan_{}/'.format(datetime.now())
if not os.path.exists(out_dir):
    os.makedirs(out_dir)
    shutil.copyfile(sys.argv[0], out_dir + '/training_script.py')
sys.stdout = mutil.Logger(out_dir)
gpu = 0
torch.cuda.set_device(gpu)
mb_size = 64 # mini-batch_size
mode_num = 4
data = data_prepare.Data_2D(mb_size,mode_num)
Z_dim = 2
X_dim = 2
h_dim = 128
cnt = 0

num = '0'

# else:
#     print("you have already creat one.")
#     exit(1)

G = model.G_Net(Z_dim,X_dim,h_dim).cuda()
D = model.D_Net(X_dim,1,h_dim).cuda()

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
    X = data.batch_next()
    X = Variable(torch.from_numpy(X.astype('float32'))).cuda()

    D_solver.zero_grad()
    # Dicriminator forward-loss-backward-update
    G_sample = G(z)
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
    G_sample = G(z)
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
        X = X.cpu().data.numpy()
        G_sample = G_sample.cpu().data.numpy()

        plt.scatter(X[:,0],X[:,1],alpha=0.5,edgecolors='blue')
        plt.scatter(G_sample[:,0], G_sample[:,1], alpha=0.5,edgecolors='red')
        plt.show()
        plt.savefig('{}/hehe_{}.png'.format(out_dir, str(cnt).zfill(3)), bbox_inches='tight')



