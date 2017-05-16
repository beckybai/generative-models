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
#from pytimer import Timer
from be_gan_models import *
import torchvision.utils as vutils
from collections import deque

gpu = 6
ngpu = 1
gpu_ids = [0]

torch.cuda.set_device(gpu)
# mnist = input_data.read_data_sets('../../MNIST_data', one_hot=True)
celebA = data_convert.celebA()
mb_size = 16 # mini-batch_size
Z_dim = 100
label_dim = 18
X_dim = 64
y_dim = 1
cnt = 0

num = '0'
out_dir = './cifar100_result/pretrain_triplet_{}_{}/'.format(datetime.now(),num)
out_dir.replace(" ","_")

if not os.path.exists(out_dir):
    os.makedirs(out_dir)
    shutil.copyfile(sys.argv[0], out_dir + '/pretrain_uncondition_triplet_large_lr.py')

sys.stdout = mutil.Logger(out_dir)
in_channel=4
d_num = 3

# G = model.G_Net_conv_64(ngpu,main_gpu = gpu, in_channel = Z_dim+label_dim,out_channel=3).cuda()
#G_model = torch.load("/home/bike/generative-models/GAN/conditional_gan/cifar100_result/basic_2017-05-16 17:33:50.051268_0/G_42500.model")
#D_model = torch.load("/home/bike/generative-models/GAN/conditional_gan/cifar100_result/basic_2017-05-16 17:33:50.051268_0/D_42500.model")
# D = model.D_Net_conv_64(ngpu,main_gpu=gpu,inchannel=3).cuda()
#G_model = torch.load("/home/bike/generative-models/GAN/conditional_gan/cifar100_result/basic_2017-05-16 15:52:13.303939_0/G_10000.model")
#D_model = torch.load("/home/bike/generative-models/GAN/conditional_gan/cifar100_result/basic_2017-05-16 15:52:13.303939_0/D_10000.model")
#D_model = torch.load("/home/bike/data/beganD_60000.model")
#G_model = torch.load("/home/bike/data/beganG_60000.model")


D_hidden_layer = 128
conv_hidden_num = 128


repeat_num = int(np.log2(X_dim)) - 2

#D = DiscriminatorCNN(input_channel=3, z_num= D_hidden_layer, repeat_num=repeat_num, hidden_num=conv_hidden_num, num_gpu=gpu_ids)
#G = GeneratorCNN(label_dim+Z_dim, D.conv2_input_dim, output_num=3, repeat_num=repeat_num, hidden_num=conv_hidden_num, num_gpu=gpu_ids)
E = model.E_Net_conv_64(ngpu,main_gpu= gpu, inchannel=3, outchannel=10).cuda()
print(E)

#G.load_state_dict(G_model)
#D.load_state_dict(D_model)

#G.cuda()
#D.cuda()
E.cuda()

#timer = Timer()
#timer.start()


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
lr = 1e-5
#G_solver = optim.Adam(G.parameters(), lr=1e-5,betas=(beta1, beta2))
#D_solver = optim.Adam(D.parameters(), lr=1e-5,betas=(beta1, beta2))
E_solver = optim.Adam(E.parameters(), lr=1e-5,betas=(beta1, beta2))

ones_label = Variable(torch.ones(mb_size)).cuda()
zeros_label = Variable(torch.zeros(mb_size)).cuda()

criterion = nn.BCELoss()
# criterion_mse = nn.MSELoss()
criterion_t = nn.TripletMarginLoss(p=1)

#half_label = Variable(torch.ones(mb_size)*0.5).cuda()

x_fixed, label = celebA.batch_next(mb_size,label=1,shuffle=True)
x_fixed = Variable(x_fixed).cuda()
# x_fixed = self._get_variable(next(data_loader))
# vutils.save_image(x_fixed.data, '{}/x_fixed.png'.format(out_dir))
owntool.save_color_picture_pixel(x_fixed.data.tolist(), '{}/x_fixed.png'.format(out_dir), image_size=64, column=4, mb_size=mb_size)

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
    owntool.save_color_picture_pixel(x.data.tolist(), x_path, image_size=64, column=4, mb_size=mb_size)
    # vutils.save_image(x.data, x_path)
    print("[*] Samples saved: {}".format(x_path))

    if x_fake is not None:
        x_fake_path = '{}{}_D_fake.png'.format(path, idx)
        x = D(x_fake)
        owntool.save_color_picture_pixel(x.data.tolist(),x_fake_path,image_size=64,column=4,mb_size = mb_size)
        # vutils.save_image(x.data, x_fake_path)
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

 #   timer.checkpoint('start D')
    # Sample data
    # ============ Train D ============#

    if it % 10 == 0:
 #       timer.checkpoint('start Ev')
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
      
        E_solver.zero_grad()
            
    #    G_solver.step()

#        timer.checkpoint('End EG')

    # Print and plot every now and then
    if it % 500 == 0:
        print('Iter-{};E_loss: {}'.format(it,E_loss_t.data.tolist(),
                              ))
    
      
        # print(c)
       # timer.summary()
#        timer.reset()

    if it % 2500==0:
        for param_group in D_solver.param_groups:
            param_group['lr'] = param_group['lr']*0.9
 
        torch.save(E.state_dict(),'{}/E_{}.model'.format(out_dir,str(it)))
