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


gpu =4
ngpu = 1

torch.cuda.set_device(gpu)
# mnist = input_data.read_data_sets('../../MNIST_data', one_hot=True)
celebA = data_convert.celebA()
mb_size = 64 # mini-batch_size
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
D = model.D_Net_conv_64(ngpu,main_gpu=gpu,inchannel=4).cuda()

"""Weight Initialization"""
# def weights_init(m):
#     classname = m.__class__.__name__
#     if classname.find('Conv') != -1:
#         m.weight.data.normal_(0.0, 0.02)

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

G.apply(weights_init)
D.apply(weights_init)

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

for it in range(100000):
    # Sample data
    z = Variable(torch.randn(mb_size, Z_dim)).cuda()
    X, c = celebA.batch_next(mb_size,0)
    X = Variable(X).cuda()
    c_d = Variable(model.set_label_celebA(c.float(),mb_size)).cuda()
    c = Variable(c).cuda()
    # label_m = np.nonzero(c)[1]
    # c_v = Variable(torch.from_numpy(model.set_label_ve_ma(c,label_dim).astype('float32'))).cuda() # for the conditon of the generator
    # label_m = torch.from_numpy(model.set_label_ve_b(c.astype('int'),ms=X_dim,batch_size= mb_size).astype('float32'))
    # c = Variable(label_m).cuda()
    # c_d = model.set_label_celebA(c)

    # Dicriminator forward-loss-backward-update
    D.zero_grad()
    G.zero_grad()

    x_g = torch.cat([z,c.float()],1).t()
    x_g.data.resize_(mb_size, Z_dim+label_dim, 1, 1)

    G_sample = G(x_g).detach()
    # X.data.resize_(mb_size, 1, X_dim, X_dim)
    D_real = D(torch.cat([X,c_d],1))
    D_fake = D(torch.cat([G_sample,c_d],1))
    D_fake_real = D(torch.cat([X/2+G_sample/2,c_d],1))

    # a = np.random.rand()
    # D_a_fake_real = D(torch.cat([X*a + G_sample*(1-a),c_d.float()],1))

    # add some other noise.

    D_loss_fr = criterion(D_fake_real,half_label)
    D_loss_fake = criterion(D_fake, zeros_label)
    D_loss_real = criterion(D_real, ones_label)
    # D_loss_a = criterion(D_a_fake_real, ones_label*a)

    D_loss_real.backward()
    D_loss_fake.backward()
    D_loss_fr.backward()
    # D_loss_a.backward()
    D_solver.step()

    # step_d_optim()
    # Housekeeping - reset gradient
    D.zero_grad()
    G.zero_grad()

    # Generator forward-loss-backward-update
    z = Variable(torch.randn(mb_size, Z_dim)).cuda()
    x_g = torch.cat([z,c.float()],1).t()
    x_g.data.resize_(mb_size, Z_dim+ label_dim, 1, 1)

    G_sample =  G(x_g)
    DG_loss = D(torch.cat([G_sample, c_d],1))
    G_loss = criterion(DG_loss, ones_label)
    G_loss.backward()
    G_solver.step()
    # Housekeeping - reset gradient
    D.zero_grad()
    G.zero_grad()

    # Print and plot every now and then
    if it % 500 == 0:
        print('Iter-{}; D_loss_real/fake: {}/{}; G_loss: {}, D_hf:{}'.format(it, D_loss_real.data.tolist(),
                                                                        D_loss_fake.data.tolist(), G_loss.data.tolist(),
                                                                              D_loss_fr.data.tolist()))
        # c = c
        # c_v = Variable(torch.from_numpy(
        #     model.set_label_ve_ma(c, label_dim).astype('float32'))).cuda()  # for the conditon of the generator
        print(c)
        x_g = torch.cat([z, c.float()], 1).t()
        x_g.data.resize_(mb_size, Z_dim + label_dim, 1, 1)
        samples = G(x_g)
        samples = samples.data.tolist()[:mb_size]
        output_path = out_dir + "{}.png".format(it)
        output_path_real = out_dir +  "{}_standard.png".format(it)
        owntool.save_color_picture_pixel(samples,output_path,image_size=64,column=8)
        owntool.save_color_picture_pixel(X.data.tolist(),output_path_real,image_size = 64,column=8)

    if it % 10000==0:
        for param_group in D_solver.param_groups:
            param_group['lr'] = param_group['lr']*0.9
        torch.save(G.state_dict(),'{}/G_{}.model'.format(out_dir,str(it)))
        torch.save(D.state_dict(),'{}/D_{}.model'.format(out_dir,str(it)))
