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
import owntool


gpu = 2
ngpu = 2

torch.cuda.set_device(gpu)
# mnist = input_data.read_data_sets('../../MNIST_data', one_hot=True)
cifar_d = data_convert.cifar100()
mb_size = 100 # mini-batch_size
Z_dim = 100
label_dim = 100
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
in_channel=4
d_num = 3

G = model.G_Net_conv_32(ngpu,in_channel = Z_dim+label_dim, out_channel = 3).cuda()
D = model.D_Net_conv(ngpu,in_channel).cuda()

"""Weight Initialization"""
# def weights_init(m):
#     classname = m.__class__.__name__
#     if classname.find('Conv') != -1:
#         m.weight.data.normal_(0.0, 0.02)






""" ===================== TRAINING ======================== """

d_num = 3
# avd_num = 1/d_num
G_solver = optim.Adam(G.parameters(), lr=1e-4)
D_solver = optim.Adam(D.parameters(), lr=1e-4)

ones_label = Variable(torch.ones(mb_size)).cuda()
zeros_label = Variable(torch.zeros(mb_size)).cuda()

criterion = nn.BCELoss()
c_label = np.array(range(100))

def reset_d_grad():
    D.zero_grad()
def step_d_optim():
    D_solver.step()

for it in range(100000):
    # Sample data
    z = Variable(torch.randn(mb_size, Z_dim)).cuda()
    X, c = cifar_d.batch_next(mb_size)
    X = Variable(torch.from_numpy(X)).cuda()
    # label_m = np.nonzero(c)[1]
    c_v = Variable(torch.from_numpy(model.set_label_ve_ma(c,100).astype('float32'))).cuda() # for the conditon of the generator
    label_m = model.set_label_cifar(c.astype('int'),mb_size,X_dim)
    c = Variable(label_m).cuda()

    # Dicriminator forward-loss-backward-update
    D.zero_grad()
    G.zero_grad()

    x_g = torch.cat([z,c_v],1).t()
    x_g.data.resize_(mb_size, Z_dim+label_dim, 1, 1)

    G_sample = G(x_g).detach()
    # X.data.resize_(mb_size, 1, X_dim, X_dim)
    D_real = D(torch.cat([X,c],1))
    D_fake = D(torch.cat([G_sample,c],1))
    D_loss_fake = criterion(D_fake, zeros_label)
    D_loss_real = criterion(D_real, ones_label)

    D_loss_real.backward()
    D_loss_fake.backward()
    D_solver.step()

    # step_d_optim()
    # Housekeeping - reset gradient
    D.zero_grad()
    G.zero_grad()

    # Generator forward-loss-backward-update
    z = Variable(torch.randn(mb_size, Z_dim)).cuda()
    x_g = torch.cat([z,c_v],1).t()
    x_g.data.resize_(mb_size, Z_dim+ label_dim, 1, 1)

    G_sample =  G(x_g)
    DG_loss = D(torch.cat([G_sample, c],1))
    G_loss = criterion(DG_loss, ones_label)
    G_loss.backward()
    G_solver.step()
    # Housekeeping - reset gradient
    D.zero_grad()
    G.zero_grad()

    # Print and plot every now and then
    if it % 500 == 0:
        print('Iter-{}; D_loss_real/fake: {}/{}; G_loss: {}'.format(it, D_loss_real.data.tolist(),
                                                                        D_loss_fake.data.tolist(), G_loss.data.tolist()))
        c = c_label
        c_v = Variable(torch.from_numpy(model.set_label_ve_ma(c,100).astype('float32'))).cuda()
        x_g = torch.cat([z, c_v], 1).t()
        x_g.data.resize_(mb_size, Z_dim + label_dim, 1, 1)
        samples = G(x_g)
        samples = samples.data.tolist()[:100]
        output_path = out_dir + "{}.png".format(it)
        owntool.save_color_picture_pixel(samples,output_path)

    if it % 10000==0:
        torch.save(G.state_dict(),'{}/G_{}.model'.format(out_dir,str(it)))
        torch.save(D.state_dict(),'{}/D_{}.model'.format(out_dir,str(it)))
