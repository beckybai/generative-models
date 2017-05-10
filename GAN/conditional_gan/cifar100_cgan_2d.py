import torch
import torch.autograd as autograd
import torch.optim as optim
import numpy as np

from datetime import datetime
import matplotlib as mpl
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


gpu = 0
ngpu = 2

torch.cuda.set_device(gpu)
# mnist = input_data.read_data_sets('../../MNIST_data', one_hot=True)
cifar_d = data_convert.cifar100()
mb_size = 100 # mini-batch_size
Z_dim = 100
X_dim = 32
y_dim = 1
cnt = 0

num = '0'
out_dir = './cifar100_result/basic_{}_{}/'.format(datetime.now(),num)
out_dir.replace(" ","_")

if not os.path.exists(out_dir):
    os.makedirs(out_dir)
    shutil.copyfile(sys.argv[0], out_dir + '/training_script.py')

sys.stdout = mutil.Logger(out_dir)
in_channel=2
d_num = 3

G = model.G_Net_conv(ngpu).cuda()
D_list = [ model.D_Net_conv(ngpu,in_channel).cuda() for i in range(d_num)]

"""Weight Initialization"""
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)

""" ===================== TRAINING ======================== """

d_num = 3
# avd_num = 1/d_num
G_solver = optim.Adam(G.parameters(), lr=1e-4)

D_solver_list = [optim.Adam(D_list[i].parameters(), lr=1e-4) for i in range(d_num)]

ones_label = Variable(torch.ones(mb_size)).cuda()
zeros_label = Variable(torch.zeros(mb_size)).cuda()

criterion = nn.BCELoss()
c_label = np.array(range(100))

def reset_d_grad():
    for net in D_list:
        net.zero_grad()
def step_d_optim():
    for opt in D_solver_list:
        opt.step()


for it in range(100000):
    # Sample data
    z = Variable(torch.randn(mb_size, Z_dim)).cuda()
    X, c = cifar_d(mb_size)
    X = Variable(torch.from_numpy(X)).cuda()
    # label_m = np.nonzero(c)[1]
    c_v = Variable(torch.from_numpy(model.set_label_ve_ma(c,100).astype('float32'))).cuda() # for the conditon of the generator
    label_m_list = model.set_label_cifar(c.astype('int'))
    c_list = [Variable(torch.from_numpy(label_m_list[i].astype('float32'))).cuda() for i in range(d_num)]

    # Dicriminator forward-loss-backward-update
    reset_d_grad()
    G.zero_grad()

    x_g = torch.cat([z,c_v],1).t()
    x_g.data.resize_(mb_size, Z_dim+10, 1, 1)

    G_sample = G(x_g).detach()
    X.data.resize_(mb_size, 1, X_dim, X_dim)

    D_real_list =[ D_list[i](torch.cat([X,c_list[i]],1)) for i in range(d_num)]
    D_fake_list =[ D_list[i](torch.cat([G_sample,c_list[i]],1)) for i in range(d_num)]
    D_loss_fake_list = [criterion(D_fake_list[i], zeros_label) for i in range(d_num)]
    D_loss_real_list = [criterion(D_real_list[i], ones_label) for i in range(d_num)]
    # D_loss_list =  D_loss_fake_list+D_loss_real_list
    # Ghost Code
    # for i in range(d_num):
    #     D_loss_list[i].backward()
    #     D_solver_list[i].step()

    for i in range(d_num):
        D_loss_real_list[i].backward()
        D_loss_fake_list[i].backward()
        D_solver_list[i].step()


    # step_d_optim()
    # Housekeeping - reset gradient
    reset_d_grad()
    G.zero_grad()

    # Generator forward-loss-backward-update
    z = Variable(torch.randn(mb_size, Z_dim)).cuda()

    x_g = torch.cat([z,c_v],1).t()
    x_g.data.resize_(mb_size, Z_dim+10, 1, 1)

    G_sample =  G(x_g)
    DG_loss_list =[ D_list[i](torch.cat([G_sample,c_list[i]],1)) for i in range(d_num)]


    G_loss = Variable(torch.zeros([1])).cuda()
    for i in range(d_num):
        G_loss = G_loss + criterion(DG_loss_list[i], ones_label)

    G_loss.backward()
    G_solver.step()

    # Housekeeping - reset gradient
    reset_d_grad()
    G.zero_grad()

    # Print and plot every now and then
    if it % 500 == 0:
        for i in range(d_num):
            print('Iter-{}; D_loss_real/fake: {}/{}; G_loss: {}'.format(it, D_loss_real_list[i].data.tolist(),
                                                                        D_loss_fake_list[i].data.tolist(), G_loss.data.tolist()))
        c = c_label
        c_v = Variable(torch.from_numpy(c.astype('float32'))).cuda()
        x_g = torch.cat([z, c_v], 1).t()
        x_g.data.resize_(mb_size, Z_dim + 10, 1, 1)
        samples = G(x_g)


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

        plt.savefig('{}/{}.png'.format(out_dir,str(cnt).zfill(3)), bbox_inches='tight')

        cnt += 1
        plt.close(fig)

        # for i in range(d_num):
        #     torch.save(D_list[i].state_dict(),'{}/D{}.model'.format(out_dir,i))
        # torch.save(G.state_dict(),'{}/G.model'.format(out_dir))

    if it % 10000==0:
        torch.save(G.state_dict(),'{}/G_{}.model'.format(out_dir,str(it)))
        for i in range(d_num):
            torch.save(D_list[i].state_dict(),'{}/D_{}_{}.model'.format(out_dir,str(it),str(i)))
