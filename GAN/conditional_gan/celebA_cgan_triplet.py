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
# from tensorflow.examples.tutorials.mnist import input_data
import torch.nn as nn
import torch.nn.functional as F
import shutil,sys
import mutil
import model
import data_convert
import owntool

from pytimer import Timer


gpu =7
ngpu = 1

torch.cuda.set_device(gpu)
# mnist = input_data.read_data_sets('../../MNIST_data', one_hot=True)
celebA = data_convert.celebA()
mb_size = 100 # mini-batch_size
Z_dim = 100
label_dim = 18
X_dim = 64
y_dim = 1
cnt = 0

num = '0'
out_dir = './cifar100_result/basic_{}_{}/'.format(datetime.now(),num)
out_dir.replace(" ","_")

if not os.path.exists(out_dir):
    os.makedirs(out_dir)
    shutil.copyfile(sys.argv[0], out_dir + '/shuideguo.py')

sys.stdout = mutil.Logger(out_dir)
in_channel=4
d_num = 3

G = model.G_Net_conv_64(ngpu,main_gpu = gpu, in_channel = Z_dim+label_dim,out_channel=3).cuda()
G_model = torch.load("/home/bike/2027/generative-models/GAN/conditional_gan/cifar100_result/basic_2017-05-14 21:08:32.778985_0/G_60000.model")
D_model = torch.load("/home/bike/2027/generative-models/GAN/conditional_gan/cifar100_result/basic_2017-05-14 21:08:32.778985_0/D_60000.model")


D = model.D_Net_conv_64(ngpu,main_gpu=gpu,inchannel=3).cuda()
E = model.E_Net_conv_64(ngpu,main_gpu= gpu, inchannel=3, outchannel=10).cuda()

G.load_state_dict(G_model)
D.load_state_dict(D_model)

G.cuda()
D.cuda()
E.cuda()

timer = Timer()
timer.start()


"""Weight Initialization"""
# def weights_init(m):
#     classname = m.__class__.__name__
#     if classname.find('Conv') != -1:
#         m.weight.data.normal_(0.0, 0.02)

# def weights_init(m):
#     classname = m.__class__.__name__
#     if classname.find('Conv') != -1:
#         m.weight.data.normal_(0.0, 0.02)
#     elif classname.find('BatchNorm') != -1:
#         m.weight.data.normal_(1.0, 0.02)
#         m.weight.data.normal_(1.0, 0.02)
#         m.bias.data.fill_(0)
#
# G.apply(weights_init)
# D.apply(weights_init)
# E.apply(weights_init)

""" ===================== TRAINING ======================== """

d_num = 3
# avd_num = 1/d_num
G_solver = optim.Adam(G.parameters(), lr=1e-4)
D_solver = optim.Adam(D.parameters(), lr=1e-4)
E_solver = optim.Adam(E.parameters(), lr=1e-5)

ones_label = Variable(torch.ones(mb_size)).cuda()
zeros_label = Variable(torch.zeros(mb_size)).cuda()

criterion = nn.BCELoss()
# criterion_mse = nn.MSELoss()
criterion_t = nn.TripletMarginLoss(p=1)
# c_label = np.array(range(10))
# c_label = c_label.repeat(10)

half_label = Variable(torch.ones(mb_size)*0.5).cuda()

def reset_d_grad():
    D.zero_grad()
def step_d_optim():
    D_solver.step()

for it in range(100000):

    timer.checkpoint('start D')
    # Sample data
    z = Variable(torch.randn(mb_size, Z_dim)).cuda()
    X, c = celebA.batch_next(mb_size,0)
    X = Variable(X).cuda()
    c_d = Variable(model.set_label_celebA(c.float(),mb_size)).cuda()
    c = Variable(c).cuda()

    # Dicriminator forward-loss-backward-update
    D.zero_grad()
    G.zero_grad()

    x_g = torch.cat([z,c.float()],1).t()
    x_g.data.resize_(mb_size, Z_dim+label_dim, 1, 1)

    G_sample = G(x_g).detach()
    # X.data.resize_(mb_size, 1, X_dim, X_dim)
    D_real = D(X)
    D_fake = D(G_sample)
    # D_fake_real = D(X/2+G_sample/2)

    # D_loss_fr = criterion(D_fake_real,half_label)
    D_loss_fake = criterion(D_fake, zeros_label)
    D_loss_real = criterion(D_real, ones_label)
    # D_loss_a = criterion(D_a_fake_real, ones_label*a)

    D_loss = D_loss_real+ D_loss_fake
    D_loss.backward()
    D_solver.step()

    # step_d_optim()
    # Housekeeping - reset gradient
    D_solver.zero_grad()
    G_solver.zero_grad()

    timer.checkpoint('start G')
    # Generator forward-loss-backward-update
    z = Variable(torch.randn(mb_size, Z_dim)).cuda()
    x_g = torch.cat([z,c.float()],1).t()
    x_g.data.resize_(mb_size, Z_dim+ label_dim, 1, 1)

    G_sample =  G(x_g)
    DG_loss = D(G_sample)
    G_loss = criterion(DG_loss, ones_label)
    G_loss.backward()
    for param_group in G_solver.param_groups:
        param_group['lr'] = 1e-4
    G_solver.step()
    # Housekeeping - reset gradient
    D_solver.zero_grad()
    G_solver.zero_grad()

# Add Evaluator to the network as the condition information.
#
    if it % 5 == 0:
        timer.checkpoint('start Ev')
        E_solver.zero_grad()
        X, c = celebA.batch_next(mb_size, label = 1,shuffle=True)
        X = Variable(X).cuda()
        # X.data.resize_(mb_size, 1, 32, 28)
        E_anch = E(X)

        X2, c2 = celebA.batch_next(mb_size, label= c, shuffle=False)
        X2 = Variable(X2).cuda()
        E_real = E(X2)

        X3, c3 = celebA.batch_next(mb_size, label=c, shuffle=False, Negative = True)
        X3 = Variable(X3).cuda()
        E_fake = E(X3)

        E_loss_t = criterion_t(E_anch, E_real, E_fake)
        E_loss_t.backward()
        E_solver.step()

        # Housekeeping - reset gradient
        # D.zero_grad()
        G_solver.zero_grad()
        E_solver.zero_grad()

        # For E ( generated data )
        timer.checkpoint('start EG')
        z = Variable(torch.randn(mb_size, Z_dim)).cuda()
        c = Variable(c).cuda()
        x_g = torch.cat([z, c.float()], 1).t()
        x_g.data.resize_(mb_size, Z_dim + label_dim, 1, 1)
        G_sample = G(x_g)
        E_real = E(G_sample)

        # X3, c3 = celebA.batch_next(mb_size, label= c.data, shuffle=False, Negative = True)
        c3 = Variable(c3).cuda()
        x_g2 = torch.cat([z, c3], 1).t()
        x_g2.data.resize_(mb_size, Z_dim + label_dim, 1, 1)
        G_sample2 = G(x_g2)
        E_fake = E(G_sample2)

        E_anch = E(X)
        E_loss_g = criterion_t(E_anch, E_real, E_fake)
        E_loss_g.backward()
        for param_group in G_solver.param_groups:
            param_group['lr'] = 1e-6
        G_solver.step()

        timer.checkpoint('End EG')

    # Housekeeping - reset gradient

    # Print and plot every now and then
    if it % 100 == 0:
        print('Iter-{}; D_loss_real/fake: {}/{}; G_loss: {}: E_loss : {}/{}'.format(it, D_loss_real.data.tolist(),
                                                                        D_loss_fake.data.tolist(), G_loss.data.tolist(),
                                                                                 E_loss_t.data.tolist(), E_loss_g.data.tolist()
                                                                               ))
        print(c)
        x_g = torch.cat([z, c.float()], 1).t()
        x_g.data.resize_(mb_size, Z_dim + label_dim, 1, 1)
        samples = G(x_g)
        samples = samples.data.tolist()[:mb_size]
        output_path = out_dir + "{}.png".format(it)
        output_path_real = out_dir +  "{}_standard.png".format(it)
        owntool.save_color_picture_pixel(samples,output_path,image_size=64,column=10,mb_size = 100)
        owntool.save_color_picture_pixel(X.data.tolist(),output_path_real,image_size = 64,column=10,mb_size = 100)
        timer.summary()
        timer.reset()

    if it % 10000==0:
        for param_group in D_solver.param_groups:
            param_group['lr'] = param_group['lr']*0.9
        torch.save(G.state_dict(),'{}/G_{}.model'.format(out_dir,str(it)))
        torch.save(D.state_dict(),'{}/D_{}.model'.format(out_dir,str(it)))
        # torch.save(E.state_dict(),'{}/E_{}.model'.format(out_dir,str(it)))
