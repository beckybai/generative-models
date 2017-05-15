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


gpu = 0
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
    shutil.copyfile(sys.argv[0], out_dir + '/training_script.py')

sys.stdout = mutil.Logger(out_dir)
in_channel=4
d_num = 3

G = model.G_Net_conv_64(ngpu,main_gpu = gpu, in_channel = Z_dim+label_dim,out_channel=3).cuda()
G2 = model.G_Net_conv_64(ngpu,main_gpu = gpu, in_channel = Z_dim+label_dim,out_channel=3).cuda()
D = model.D_Net_conv_64(ngpu,main_gpu=gpu,inchannel=4).cuda()
G_model = torch.load("/home/bike/2027/generative-models/GAN/conditional_gan/cifar100_result/basic_2017-05-15 15:56:23.006476_0/G_20000.model")
G_model2 = torch.load('/home/bike/2027/generative-models/GAN/conditional_gan/cifar100_result/basic_2017-05-14 12:01:45.280474_0_good/G_80000.model') # naive _ cgan


G.load_state_dict(G_model)
G2.load_state_dict(G_model2)
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
criterion_mse = nn.MSELoss()

# c_label = np.array(range(10))
# c_label = c_label.repeat(10)

half_label = Variable(torch.ones(mb_size)*0.5).cuda()

def reset_d_grad():
    D.zero_grad()
def step_d_optim():
    D_solver.step()

for it in range(10):
    # Sample data
    z = Variable(torch.randn(mb_size, Z_dim)).cuda()
    X, c = celebA.batch_next(10,0)
    X = X.repeat(10,1,1,1)
    c = c.repeat(10,1)
    # c = Variable(c).cuda()

    X2, c2 = celebA.batch_next(10,0)
    X2 = X2.repeat(10,1,1,1)
    c2 = c2.repeat(10,1)
    # c2 = Variable(c2).cuda()

    c_t = torch.ceil((c.float() + c2.float())/2)
    c_t = Variable(c_t).cuda()

    x_g = torch.cat([z,c_t.float()],1).t()
    x_g.data.resize_(mb_size, Z_dim+label_dim, 1, 1)
    samples = G(x_g)
    # samples2 = G2(x_g)
    samples = samples.data.tolist()[:mb_size]
    # samples2 = samples2.data.tolist()[:mb_size]
    output_path = out_dir + "{}.png".format(it)
    output_path2 = out_dir + "{}_standard2.png".format(it)

    output_path_real = out_dir + "{}_standard.png".format(it)

    owntool.save_color_picture_pixel(samples, output_path, image_size=64, column=10)
    owntool.save_color_picture_pixel(X2.tolist(), output_path2, image_size=64, column=10)
    owntool.save_color_picture_pixel(X.tolist(), output_path_real, image_size=64, column=10)
    # Print and plot every now and then
