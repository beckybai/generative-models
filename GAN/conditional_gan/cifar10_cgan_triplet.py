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


gpu = 1
ngpu = 1

torch.cuda.set_device(gpu)
# mnist = input_data.read_data_sets('../../MNIST_data', one_hot=True)
cifar_d = data_convert.cifar10()
mb_size = 100 # mini-batch_size
Z_dim = 100
label_dim = 10
X_dim = 32
y_dim = 1
cnt = 0

num = '0'
out_dir = './cifar100_result/Triplet_{}_{}/'.format(datetime.now(),num)
out_dir.replace(" ","_")
model_dir = '/home/bike/2027/generative-models/GAN/conditional_gan/cifar100_result/cifar10_9w/'


if not os.path.exists(out_dir):
    os.makedirs(out_dir)
    shutil.copyfile(sys.argv[0], out_dir + '/training_script.py')

sys.stdout = mutil.Logger(out_dir)
in_channel=3
d_num = 3

G = model.G_Net_conv_32(ngpu,main_gpu = gpu, in_channel = Z_dim+label_dim, out_channel = 3).cuda()
D = model.D_Net_conv(ngpu,in_channel,main_gpu=gpu).cuda()
E = model.Ev_Net_conv(ngpu,in_channel,main_gpu=gpu).cuda()

# g_model = torch.load(model_dir+'G_90000.model')
# d_model = torch.load(model_dir+'D_90000.model')
# G.load_state_dict(g_model)
# D.load_state_dict(d_model)

"""Weight Initialization"""
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)

""" ===================== TRAINING ======================== """

# avd_num = 1/d_num
G_solver = optim.Adam(G.parameters(), lr=1e-4)
D_solver = optim.Adam(D.parameters(), lr=1e-4)
E_solver = optim.Adam(E.parameters(), lr=1e-4)

ones_label = Variable(torch.ones(mb_size)).cuda()
zeros_label = Variable(torch.zeros(mb_size)).cuda()
half_label = Variable(torch.ones(mb_size)*0.5).cuda()


criterion = nn.BCELoss()
criterion_t = nn.TripletMarginLoss(p=1)
criterion_mse = nn.MSELoss()
c_label = np.array(range(10))
c_label = c_label.repeat(10)


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
    c_v = Variable(torch.from_numpy(model.set_label_ve_ma(c,label_dim).astype('float32'))).cuda() # for the conditon of the generator
    label_m = torch.from_numpy(model.set_label_ve_b(c.astype('int'),ms=X_dim,batch_size= mb_size).astype('float32'))
    c = Variable(label_m).cuda()

    # Dicriminator forward-loss-backward-update
    D.zero_grad()
    G.zero_grad()

    x_g = torch.cat([z,c_v],1).t()
    x_g.data.resize_(mb_size, Z_dim+label_dim, 1, 1)

    G_sample = G(x_g).detach()
    # X.data.resize_(mb_size, 1, X_dim, X_dim)
    D_real = D(X)
    D_fake = D(G_sample)
    D_loss_fake = criterion(D_fake, zeros_label)
    D_loss_real = criterion(D_real, ones_label)
    D_fake_real = D(X/2+G_sample/2)
    D_loss_fr = criterion(D_fake_real,half_label)


    D_loss_real.backward()
    D_loss_fake.backward()
    D_loss_fr.backward()
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
    DG_loss = D(G_sample)
    G_loss = criterion(DG_loss, ones_label)
    G_loss.backward()
    G_solver.step()
    # Housekeeping - reset gradient
    D.zero_grad()
    G.zero_grad()

# E part
# For E, triplet part
#     z = Variable(torch.randn(mb_size, Z_dim)).cuda()
    E.zero_grad()
    X, c = cifar_d.batch_next(mb_size, shuffle=False)
    X = Variable(torch.from_numpy(X.astype('float32'))).cuda()
    # X.data.resize_(mb_size, 1, 32, 28)
    E_anch = E(X)

    X2, c2 = cifar_d.batch_next(mb_size, shuffle=False)
    X2 = Variable(torch.from_numpy(X2.astype('float32'))).cuda()
    # X2.data.resize_(mb_size, 1, 28, 28)
    E_real = E(X2)

    random_label = model.set_label_ve_ma((c.astype('int') + (np.random.rand(mb_size) * 9 + 1).astype('int')) % 10)
    random_label = Variable(torch.from_numpy(random_label.astype('float32'))).cuda()  # label for g c

    new_label_base = np.array(range(10))
    new_label = (new_label_base + (np.random.rand(10) * 9 + 1).astype('int'))%10
    X3, c3 = cifar_d.batch_next(mb_size, label=new_label, shuffle=False)
    X3 = Variable(torch.from_numpy(X3.astype('float32'))).cuda()
    # X3.data.resize_(mb_size, 1, 28, 28)
    E_fake = E(X3)

    E_loss = criterion_t(E_anch, E_real, E_fake)
    E_loss.backward()
    E_solver.step()

    # Housekeeping - reset gradient
    D.zero_grad()
    G.zero_grad()
    E.zero_grad()

    # For E ( generated data )
    z = Variable(torch.randn(mb_size, Z_dim)).cuda()
    c_v = Variable(torch.from_numpy(model.set_label_ve_ma(c.astype('int')).astype('float32'))).cuda()  # label for g c
    x_g = torch.cat([z, c_v], 1).t()
    x_g.data.resize_(mb_size, Z_dim + 10, 1, 1)
    G_sample = G(x_g)
    E_real = E(G_sample)

    # genereted (another kind of data)
    random_label = model.set_label_ve_ma((c.astype('int') + (np.random.rand(mb_size) * 9 + 1).astype('int')) % 10)
    random_label = Variable(torch.from_numpy(random_label.astype('float32'))).cuda()  # label for g c
    x_g2 = torch.cat([z, random_label], 1).t()
    x_g2.data.resize_(mb_size, Z_dim + 10, 1, 1)
    G_sample2 = G(x_g2)
    E_fake = E(G_sample2)

    E_anch = E(X)
    E_loss = criterion_t(E_anch, E_real, E_fake)
    E_loss.backward()
    # E_solver.step()
    G_solver.step()

    # Housekeeping - reset gradient
    D.zero_grad()
    G.zero_grad()
    E.zero_grad()

    # Print and plot every now and then
    if it % 500 == 0:
        print('Iter-{}; D_loss_real/fake: {}/{}; G_loss: {}, E_loss: {}'.
              format(it, D_loss_real.data.tolist(),
                                                                        D_loss_fake.data.tolist(), G_loss.data.tolist(),
                     E_loss.data.tolist()))
        c = c_label
        c_v = Variable(torch.from_numpy(
            model.set_label_ve_ma(c, label_dim).astype('float32'))).cuda()  # for the conditon of the generator
        x_g = torch.cat([z, c_v], 1).t()
        x_g.data.resize_(mb_size, Z_dim + label_dim, 1, 1)
        samples = G(x_g)
        samples = samples.data.tolist()[:100]
        output_path = out_dir + "{}.png".format(it)
        owntool.save_color_picture_pixel(samples,output_path)

    if it % 10000==0:
        torch.save(G.state_dict(),'{}/G_{}.model'.format(out_dir,str(it)))
        torch.save(D.state_dict(),'{}/D_{}.model'.format(out_dir,str(it)))
