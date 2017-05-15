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
from be_gan_models import *
import torchvision.utils as vutils
from collections import deque

gpu =4
ngpu = 1
gpu_ids = [4,5]

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

# G = model.G_Net_conv_64(ngpu,main_gpu = gpu, in_channel = Z_dim+label_dim,out_channel=3).cuda()
G_model = torch.load("/home/bike/2027/generative-models/GAN/conditional_gan/cifar100_result/basic_2017-05-15 19:57:38.738341_0/G_10000.model")
D_model = torch.load("/home/bike/2027/generative-models/GAN/conditional_gan/cifar100_result/basic_2017-05-15 19:57:38.738341_0/D_10000.model")
# D = model.D_Net_conv_64(ngpu,main_gpu=gpu,inchannel=3).cuda()
D_hidden_layer = 128
conv_hidden_num = 128


repeat_num = int(np.log2(X_dim)) - 2

D = DiscriminatorCNN(input_channel=3, z_num= D_hidden_layer, repeat_num=repeat_num, hidden_num=conv_hidden_num, num_gpu=gpu_ids)
G = GeneratorCNN(label_dim+Z_dim, D.conv2_input_dim, output_num=3, repeat_num=repeat_num, hidden_num=conv_hidden_num, num_gpu=gpu_ids)
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
beta1 = 0.5
beta2 = 0.999
G_solver = optim.Adam(G.parameters(), lr=1e-4,betas=(beta1, beta2))
D_solver = optim.Adam(D.parameters(), lr=1e-4,betas=(beta1, beta2))
E_solver = optim.Adam(E.parameters(), lr=1e-5,betas=(beta1, beta2))

ones_label = Variable(torch.ones(mb_size)).cuda()
zeros_label = Variable(torch.zeros(mb_size)).cuda()

criterion = nn.BCELoss()
# criterion_mse = nn.MSELoss()
criterion_t = nn.TripletMarginLoss(p=1)

half_label = Variable(torch.ones(mb_size)*0.5).cuda()

x_fixed, label = celebA.batch_next(mb_size,label=1,shuffle=True)
x_fixed = Variable(x_fixed).cuda()
# x_fixed = self._get_variable(next(data_loader))
vutils.save_image(x_fixed.data, '{}/x_fixed.png'.format(out_dir))

k_t = 0
prev_measure = 1
lr_update_step = 10000
measure_history = deque([0] * lr_update_step, lr_update_step)
gamma = 0.5
lambda_k = 0.001


def generate( inputs, path, idx=None):
    path = '{}{}_G.png'.format(path, idx)
    x = G(inputs)
    vutils.save_image(x.data, path)
    print("[*] Samples saved: {}".format(path))
    return x


def autoencode( inputs, path, idx=None, x_fake=None):
    x_path = '{}{}_D.png'.format(path, idx)
    x = D(inputs)
    vutils.save_image(x.data, x_path)
    print("[*] Samples saved: {}".format(x_path))

    if x_fake is not None:
        x_fake_path = '{}{}_D_fake.png'.format(path, idx)
        x = D(x_fake)
        vutils.save_image(x.data, x_fake_path)
        print("[*] Samples saved: {}".format(x_fake_path))

def reset_d_grad():
    D.zero_grad()
def step_d_optim():
    D_solver.step()

X, c_fixed = celebA.batch_next(mb_size,0)
z_fixed = Variable(torch.randn(mb_size, Z_dim), volatile=True).cuda()
c_fixed = Variable(c_fixed, volatile=True).cuda()
zc_fixed = torch.cat([z_fixed, c_fixed.float()],1)


for it in range(100000):

    timer.checkpoint('start D')
    # Sample data
    # ============ Train D ============#
    D_solver.zero_grad()
    z = Variable(torch.randn(mb_size, Z_dim)).cuda()
    X, c = celebA.batch_next(mb_size,0)
    X = Variable(X).cuda()
    AE_x = D(X)
    d_loss_real = torch.mean(torch.abs(AE_x - X))
    # z = Variable(torch.FloatTensor(mb_size, Z_dim))

    c_d = Variable(model.set_label_celebA(c.float(),mb_size)).cuda()
    c = Variable(c).cuda()

    # Dicriminator forward-loss-backward-update
    fake_image = G(torch.cat([z,c.float()],1))
    AE_fake = D(fake_image.detach())
    d_loss_fake = torch.mean(torch.abs(AE_fake - fake_image.detach()))

    d_loss = d_loss_real - k_t * d_loss_fake
    d_loss.backward()
    D_solver.step()

    # ============ Train G ============#
    # zero the grad buffer
    G_solver.zero_grad()
    timer.checkpoint('start G')
    # Generator forward-loss-backward-update
    z = Variable(torch.randn(mb_size, Z_dim)).cuda()
    x_g = torch.cat([z,c.float()],1)
    fake_image = G(x_g)
    AE_fake = D(fake_image)
    g_loss = torch.mean(torch.abs(AE_fake - fake_image))

    g_loss.backward()
    G_solver.step()

    g_d_balance = (gamma * d_loss_real - d_loss_fake).data[0]
    k_t += lambda_k * g_d_balance
    k_t = max(min(1, k_t), 0)

    measure = d_loss_real.data[0] + abs(g_d_balance)
    measure_history.append(measure)

    if it % 100 == 0:
        x_fake = generate(zc_fixed, out_dir, idx=it) # get picture G
        autoencode(x_fixed, out_dir, idx=it, x_fake=x_fake) # get the reconstructed picture D

    # G_sample =  G(x_g)
    # DG_loss = D(G_sample)
    # G_loss = criterion(DG_loss, ones_label)
    # G_loss.backward()
    # for param_group in G_solver.param_groups:
    #     param_group['lr'] = 1e-4
    # G_solver.step()
    # # Housekeeping - reset gradient
    # D_solver.zero_grad()
    # G_solver.zero_grad()

    # ============ Train Triplet Net ============#
# Add Evaluator to the network as the condition information.
#
    # timer.checkpoint('start Ev')
    # E_solver.zero_grad()
    # X, c = celebA.batch_next(mb_size, label = 1,shuffle=True)
    # X = Variable(X).cuda()
    # # X.data.resize_(mb_size, 1, 32, 28)
    # E_anch = E(X)
    #
    # X2, c2 = celebA.batch_next(mb_size, label= c, shuffle=False)
    # X2 = Variable(X2).cuda()
    # E_real = E(X2)
    #
    # X3, c3 = celebA.batch_next(mb_size, label=c, shuffle=False, Negative = True)
    # X3 = Variable(X3).cuda()
    # E_fake = E(X3)
    #
    # E_loss = criterion_t(E_anch, E_real, E_fake)
    # E_loss.backward()
    # E_solver.step()
    #
    # # Housekeeping - reset gradient
    # # D.zero_grad()
    # G_solver.zero_grad()
    # E_solver.zero_grad()
    #
    # # For E ( generated data )
    # timer.checkpoint('start EG')
    # z = Variable(torch.randn(mb_size, Z_dim)).cuda()
    # c = Variable(c).cuda()
    # x_g = torch.cat([z, c.float()], 1).t()
    # x_g.data.resize_(mb_size, Z_dim + label_dim, 1, 1)
    # G_sample = G(x_g)
    # E_real = E(G_sample)
    #
    # # X3, c3 = celebA.batch_next(mb_size, label= c.data, shuffle=False, Negative = True)
    # c3 = Variable(c3).cuda()
    # x_g2 = torch.cat([z, c3], 1).t()
    # x_g2.data.resize_(mb_size, Z_dim + label_dim, 1, 1)
    # G_sample2 = G(x_g2)
    # E_fake = E(G_sample2)
    #
    # E_anch = E(X)
    # E_loss = criterion_t(E_anch, E_real, E_fake)
    # E_loss.backward()
    # for param_group in G_solver.param_groups:
    #     param_group['lr'] = 1e-6
    # G_solver.step()
    #
    # timer.checkpoint('End EG')



    # Housekeeping - reset gradient

    # Print and plot every now and then
    if it % 500 == 0:
        print('Iter-{}; D_loss_real/fake: {}/{}; G_loss: {}'.format(it, d_loss_real.data.tolist(),
                                                                        d_loss_fake.data.tolist(), g_loss.data.tolist(),
                                                                               ))
        print(c)
        x_g = torch.cat([z, c.float()], 1)
        # x_g.data.resize_(mb_size, Z_dim + label_dim, 1, 1)
        samples = G(x_g)
        samples = samples.data.tolist()[:mb_size]
        output_path = out_dir + "{}.png".format(it)
        output_path_real = out_dir +  "{}_standard.png".format(it)
        owntool.save_color_picture_pixel(samples,output_path,image_size=64,column=10,mb_size = 100)
        owntool.save_color_picture_pixel(X.data.tolist(),output_path_real,image_size = 64,column=10,mb_size = 100)
        timer.summary()
        timer.reset()

    if it % 2500==0:
        for param_group in D_solver.param_groups:
            param_group['lr'] = param_group['lr']*0.9
        torch.save(G.state_dict(),'{}/G_{}.model'.format(out_dir,str(it)))
        torch.save(D.state_dict(),'{}/D_{}.model'.format(out_dir,str(it)))
        # torch.save(E.state_dict(),'{}/E_{}.model'.format(out_dir,str(it)))
