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

gpu = 0
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
out_dir = './cifar100_result/regression_BE_{}_{}/'.format(datetime.now(),num)
out_dir.replace(" ","_")

if not os.path.exists(out_dir):
    os.makedirs(out_dir)
    shutil.copyfile(sys.argv[0], out_dir + '/shuideguo.py')

sys.stdout = mutil.Logger(out_dir)
in_channel=4
d_num = 3

# G = model.G_Net_conv_64(ngpu,main_gpu = gpu, in_channel = Z_dim+label_dim,out_channel=3).cuda()
G_model = torch.load("/home/bike/2027/generative-models/GAN/conditional_gan/cifar100_result/basic_2017-05-16 16:34:10.009671_0/G_40000.model")
D_model = torch.load("/home/bike/2027/generative-models/GAN/conditional_gan/cifar100_result/basic_2017-05-16 16:34:10.009671_0/D_40000.model")
# D = model.D_Net_conv_64(ngpu,main_gpu=gpu,inchannel=3).cuda()
D_hidden_layer = 128
conv_hidden_num = 128


repeat_num = int(np.log2(X_dim)) - 2

D = DiscriminatorCNN_Part1(input_channel=3, z_num= D_hidden_layer, repeat_num=repeat_num, hidden_num=conv_hidden_num, num_gpu=gpu_ids)
G = GeneratorCNN(label_dim+Z_dim, D.conv2_input_dim, output_num=3, repeat_num=repeat_num, hidden_num=conv_hidden_num, num_gpu=gpu_ids)
D_G = DiscriminatorCNN_GAN(input_channel=3, conv2_input_dim=D.conv2_input_dim, repeat_num=repeat_num, hidden_num=conv_hidden_num, num_gpu=gpu_ids )
D_L = DiscriminatorCNN_Label(input_channel=3, conv2_input_dim=D.conv2_input_dim, repeat_num=1, hidden_num=conv_hidden_num, num_gpu=gpu_ids, output_dim = label_dim)
# E = model.E_Net_conv_64(ngpu,main_gpu= gpu, inchannel=3, outchannel=10).cuda()

G.load_state_dict(G_model)
D.load_state_dict(D_model)

G.cuda()
D.cuda()
D_G.cuda()
D_L.cuda()
# E.cuda()

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
lr = 1e-4
G_solver = optim.Adam(G.parameters(), lr=1e-4, betas=(beta1, beta2))
D_solver = optim.Adam(D.parameters(), lr=1e-4, betas=(beta1, beta2))
DG_solver = optim.Adam(D_G.parameters(), lr=1e-4, betas=(beta1, beta2))
DL_solver = optim.Adam(D_L.parameters(), lr=1e-4, betas=(beta1, beta2))
# E_solver = optim.Adam(E.parameters(), lr=1e-5,betas=(beta1, beta2))

ones_label = Variable(torch.ones(mb_size)).cuda()
zeros_label = Variable(torch.zeros(mb_size)).cuda()

criterion = nn.BCELoss()
# criterion_mse = nn.MSELoss()
# criterion_t = nn.TripletMarginLoss(p=1)

# half_label = Variable(torch.ones(mb_size)*0.5).cuda()

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
    x_tmp = D(inputs)
    x = D_G(x_tmp)
    owntool.save_color_picture_pixel(x.data.tolist(), x_path, image_size=64, column=4, mb_size=mb_size)
    # vutils.save_image(x.data, x_path)
    print("[*] Samples saved: {}".format(x_path))

    if x_fake is not None:
        x_fake_path = '{}{}_D_fake.png'.format(path, idx)
        x_tmp = D(x_fake)
        x = D_G(x_tmp)
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
criterion_sm = nn.SoftMarginLoss()

for it in range(100000):

    timer.checkpoint('start D')
    # Sample data
    # ============ Train D ============#
    D_solver.zero_grad()
    DG_solver.zero_grad()
    DL_solver.zero_grad()
    z = Variable(torch.randn(mb_size, Z_dim)).cuda()
    X, c = celebA.batch_next(mb_size,0)
    X = Variable(X).cuda()
    AE_x = D_G(D(X))
    d_loss_real = torch.mean(torch.abs(AE_x - X))
    # z = Variable(torch.FloatTensor(mb_size, Z_dim))
    c_d = Variable(model.set_label_celebA(c.float(),mb_size)).cuda()
    c = Variable(c).cuda()
    c = c.float()
    C_x = D_L(D(X).detach()) # the condition information will not influence the encoder
    d_loss_condition = criterion_sm(C_x,c)

    # Dicriminator forward-loss-backward-update
    fake_image = G(torch.cat([z,c],1))
    AE_fake = D_G(D(fake_image.detach()))
    d_loss_fake = torch.mean(torch.abs(AE_fake - fake_image.detach()))

    d_loss = d_loss_real - k_t * d_loss_fake
    d_loss.backward()
    d_loss_condition.backward()
    D_solver.step()
    DG_solver.step()
    DL_solver.step()

    # ============ Train G ============#
    # zero the grad buffer
    G_solver.zero_grad()
    DL_solver.zero_grad()
    timer.checkpoint('start G')
    # Generator forward-loss-backward-update
    z = Variable(torch.randn(mb_size, Z_dim)).cuda()
    x_g = torch.cat([z,c],1)
    fake_image = G(x_g)

    AE_fake = D_G(D(fake_image))
    C_x = D_L(D(fake_image))
    g_loss = torch.mean(torch.abs(AE_fake - fake_image))
    d_loss_condition_g = torch.mean(torch.abs(C_x-c))

    d_loss_condition_g.backward(retain_variables=True)
    g_loss.backward(retain_variables=False)
    G_solver.step()
    DL_solver.step()

    g_d_balance = (gamma * d_loss_real - d_loss_fake).data[0]
    k_t += lambda_k * g_d_balance
    k_t = max(min(1, k_t), 0)

    measure = d_loss_real.data[0] + abs(g_d_balance)
    measure_history.append(measure)

    if it % lr_update_step == lr_update_step - 1:
        cur_measure = np.mean(measure_history)
        if cur_measure > prev_measure * 0.9999:
            lr *= 0.5
            for param_group in D_solver.param_groups:
                param_group['lr'] = param_group['lr'] * 0.5
            for param_group in G_solver.param_groups:
                param_group['lr'] = param_group['lr'] * 0.5
        prev_measure = cur_measure

    # ============ Train Condition ============#
    # Print and plot every now and then
    if it % 500 == 0:
        print('Iter-{}; D_loss_real/fake: {}/{}; G_loss: {},C_loss: {}/{}, measure:{}, k_t : {}, lr = {}'.format(it, d_loss_real.data.tolist(),
                                                                d_loss_fake.data.tolist(), g_loss.data.tolist(),
                                                                d_loss_condition.data.tolist(),d_loss_condition_g.data.tolist(),
                                                                measure,k_t,lr
                                                                               ))
        x_fake = generate(zc_fixed, out_dir, idx=it) # get picture G
        autoencode(x_fixed, out_dir, idx=it, x_fake=x_fake) # get the reconstructed picture D
        # print(c)
        x_g = torch.cat([z, c.float()], 1)
        # x_g.data.resize_(mb_size, Z_dim + label_dim, 1, 1)
        samples = G(x_g)
        samples = samples.data.tolist()[:mb_size]
        output_path = out_dir + "{}.png".format(it)
        output_path_real = out_dir +  "{}_standard.png".format(it)
        owntool.save_color_picture_pixel(samples,output_path,image_size=64,column=4,mb_size = mb_size)
        owntool.save_color_picture_pixel(X.data.tolist(),output_path_real,image_size = 64,column=4,mb_size = mb_size)
        timer.summary()
        timer.reset()

    if it % 2500==0:
        for param_group in D_solver.param_groups:
            param_group['lr'] = param_group['lr']*0.9
        torch.save(G.state_dict(),'{}/G_{}.model'.format(out_dir,str(it)))
        torch.save(D.state_dict(),'{}/D_{}.model'.format(out_dir,str(it)))
        torch.save(D_G.state_dict(),'{}/DG_{}.model'.format(out_dir,str(it)))
        torch.save(D_L.state_dict(),'{}/DL_{}.model'.format(out_dir,str(it)))

