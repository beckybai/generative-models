import torch
import torch.autograd as autograd
import torch.optim as optim
import numpy as np
import matplotlib as mpl
from datetime import datetime

mpl.use('Agg')

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F
import shutil, sys
import mutil
import toy_model as model
import data_prepare

out_dir = './out/gan_lr_small_{}'.format(datetime.now())
out_dir = out_dir.replace(" ", "_")
print(out_dir)

if not os.path.exists(out_dir):
    os.makedirs(out_dir)
    shutil.copyfile(sys.argv[0], out_dir + '/training_script.py')

sys.stdout = mutil.Logger(out_dir)
gpu = 3
torch.cuda.set_device(gpu)
mb_size = 600  # mini-batch_size
mode_num = 2

distance = 10
data = data_prepare.Data_2D_Circle(mb_size, mode_num, distance)

Z_dim = 2
X_dim = 2
h_dim = 128
c_dim = mode_num * mode_num
cnt = 0

num = '0'

# else:
#     print("you have already creat one.")
#     exit(1)

G = model.G_Net(Z_dim , X_dim, h_dim).cuda()
D = model.D_Net(X_dim , 1, h_dim).cuda()
model_load_path = "/home/bike/2027/generative-models/GAN/conditional_gan/toys/out/gan_2017-05-22_14:26:27.855839"
E = model.E_Net(X_dim + c_dim, 10, h_dim).cuda()
G_model = torch.load("{}/G.model".format(model_load_path))
D_model = torch.load("{}/D.model".format(model_load_path))
# G_fake = model.Direct_Net(X_dim+c_dim, 1, h_dim).cuda()
# G.apply(model.weights_init)
# D.apply(model.weights_init)
# G.load_state_dict(G_model)
# D.load_state_dict(D_model)
D.apply(model.weights_init)
G.apply(model.weights_init)
E.apply(model.weights_init)
""" ===================== TRAINING ======================== """

lr = 1e-4
G_solver = optim.Adam(G.parameters(), lr=lr)
D_solver = optim.Adam(D.parameters(), lr=lr)
E_solver = optim.Adam(E.parameters(), lr=lr)

ones_label = Variable(torch.ones(mb_size)).cuda()
zeros_label = Variable(torch.zeros(mb_size)).cuda()

criterion = nn.BCELoss()
criterion_t = nn.TripletMarginLoss(p=1)

grads = {}
skip = (slice(None, None, 3), slice(None, None, 3))


def save_grad(name):
    def hook(grad):
        grads[name] = grad

    return hook


# z_fixed = Variable(torch.randn(20, Z_dim), volatile=False).cuda()
# c_fixed = np.array(range(0,4))
# c_fixed = Variable(torch.from_numpy(mutil.label_num2vec(np.repeat(c_fixed,5)).astype("float32")),volatile=False).cuda()
# zc_fixed = torch.cat([z_fixed, c_fixed], 1)

z_fixed = torch.randn(20, Z_dim)
c_fixed = np.array(range(0, mode_num * mode_num))
c_fixed = Variable(
    torch.from_numpy(mutil.label_num2vec(np.repeat(c_fixed, mb_size // (mode_num * mode_num))).astype("float32")),
    volatile=False).cuda()
# zc_fixed = torch.cat([z_fixed, c_fixed],1)
# zc_fixed = Variable(zc_fixed, volatile=False).cuda()

grid_num = 100
y_fixed, x_fixed = np.mgrid[-20:20:float(40)/grid_num, -20:20:float(40)/grid_num]
x_fixed, y_fixed = x_fixed.reshape(grid_num * grid_num, 1), y_fixed.reshape(grid_num * grid_num, 1)
mesh_fixed_cpu = np.concatenate([x_fixed, y_fixed], 1)
mesh_fixed = Variable(torch.from_numpy(mesh_fixed_cpu.astype("float32")).cuda())


# mesh_fixed.register_hook(save_grad('Mesh'))

def get_grad(input, label, name, c=False, is_z=True, need_sample = False):

    D.zero_grad()
    if (is_z):
        sample = G(input)
    else:
        input.requires_grad= True
        sample = input
    sample.register_hook(save_grad(name))
    if (c):
        d_result = D(sample)
    else:
        d_result = D(sample)

    ones_label_tmp = Variable(torch.ones([d_result.data.size()[0], 1])).cuda()
    loss_real = criterion(d_result, ones_label_tmp * label)
    loss_real.backward()
    D.zero_grad()
    G.zero_grad()
    if(need_sample):
        return d_result,sample
    else:
        return d_result


for it in range(200000):
    # Sample data
    z = Variable(torch.randn(mb_size, Z_dim)).cuda()
    X, c = data.batch_next(old_label=False,need_label=True)  # with label
    X = Variable(torch.from_numpy(X.astype('float32'))).cuda()
    c = np.zeros(mb_size)
    c = Variable(torch.from_numpy(mutil.label_num2vec(c.astype('int'),max_label=mode_num*mode_num-1).astype('float32'))).cuda()
    # c = Variable(torch.zeros([mb_size,1])).cuda()

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
    z = Variable(torch.randn(mb_size, Z_dim).cuda(), requires_grad=True)
    # print(c.cpu().data.numpy().shape)
    G_sample = G(z)
    G_sample.register_hook(save_grad('G'))
    # G_sample.requires_grad= True
    D_fake = D(G_sample)
    G_loss = criterion(D_fake, ones_label)

    G_loss.backward()
    G_solver.step()
    # print(grads['G'])

    # Housekeeping - reset gradient
    D.zero_grad()
    G.zero_grad()
    # triplet part

    if it % 5000 == 0:
        #	print(zc_fixed_cpu)
        lr = lr * 0.5
        for param_group in G_solver.param_groups:
            param_group['lr'] = param_group['lr'] * 0.8
        for param_group in D_solver.param_groups:
            param_group['lr'] = param_group['lr'] * 0.8
        for param_group in E_solver.param_groups:
            param_group['lr'] = param_group['lr'] * 0.2

    # Print and plot every now and then
    if it % 1000 == 0:
        fig, ax = plt.subplots()

        print('Iter-{}; D_loss_real/fake: {}/{}; G_loss: {}'.format(it, D_loss_real.data.tolist(),
                                                                    D_loss_fake.data.tolist(), G_loss.data.tolist(),
                                                                   ))
        X = X.cpu().data.numpy()
        G_sample_cpu = G_sample.cpu().data.numpy()
        d_g_sample_cpu = get_grad(G_sample.detach(), 1, 'G', c=False)

        gd_cpu = -grads['G'].cpu().data.numpy()

        d_mesh = (get_grad(mesh_fixed, 1, 'mesh', is_z=False)).cpu().data.numpy()
        print(d_mesh)

        #	ax.quiver(zc_fixed_cpu[:, 0], zc_fixed_cpu[:, 1], gd_fixed_cpu[:, 0], gd_fixed_cpu[:, 1],
        #	          d_z_fixed.cpu().data.numpy(), units='xy')
        gd_mesh_cpu = -grads['mesh'].cpu().data.numpy()
        # print(gd_mesh_cpu.shape)
        #
        gd_mesh_cpu_x, gd_mesh_cpu_y = np.expand_dims(gd_mesh_cpu[:, 0], 1).reshape(grid_num, grid_num), np.expand_dims(
            gd_mesh_cpu[:, 1], 1).reshape(grid_num, grid_num)
        d_mesh = d_mesh.reshape(grid_num, grid_num)
        x_fixed, y_fixed = x_fixed.reshape(grid_num, grid_num), y_fixed.reshape(grid_num, grid_num)
        ax.quiver(x_fixed[::3, ::3], y_fixed[::3, ::3], gd_mesh_cpu_x[::3, ::3], gd_mesh_cpu_y[::3, ::3],
                  d_mesh[::3, ::3], units='xy')

        #	ax.quiver(zc_fixed_cpu[:, 0], zc_fixed_cpu[:, 1], gd_fixed_cpu[:, 0], gd_fixed_cpu[:, 1],
        #	          d_z_fixed.cpu().data.numpy(), units='xy')
        # gd_mesh_cpu = -grads['mesh'].cpu().data.numpy()
        # print(gd_mesh_cpu.shape)
        #
        # gd_mesh_cpu_x, gd_mesh_cpu_y = np.expand_dims(gd_mesh_cpu[:, 0], 1).reshape(grid_num, grid_num), np.expand_dims(
        # 	gd_mesh_cpu[:, 1], 1).reshape(grid_num, grid_num)
        # d_mesh = d_mesh.reshape(grid_num, grid_num)
        # x_fixed, y_fixed = x_fixed.reshape(grid_num, grid_num), y_fixed.reshape(grid_num, grid_num)
        # ax.quiver(x_fixed[::3, ::3], y_fixed[::3, ::3], gd_mesh_cpu_x[::3, ::3], gd_mesh_cpu_y[::3, ::3],
        #           d_mesh[::3, ::3], units='xy')
        #
        # print(np.abs(gd_fixed_cpu).mean())
        #
        ax.set(aspect=1, title='4 mode. triplet lr=1e-4')
        #		plt.scatter(zc_fixed_cpu[:, 0], zc_fixed_cpu[:, 1], s=1, color='yellow')
        plt.scatter(X[:, 0], X[:, 1], s=1, edgecolors='blue', color='blue')
        plt.scatter(G_sample_cpu[:, 0], G_sample_cpu[:, 1], s=1, color='red', edgecolors='red')
        plt.show()
        plt.ylim((-20, 20))
        plt.xlim((-20, 20))

        plt.savefig('{}/hehe_{}.png'.format(out_dir, str(cnt).zfill(3)), bbox_inches='tight')
        if it % 5000 == 0:
            plt.savefig('{}/hehe_{}.pdf'.format(out_dir, str(cnt).zfill(3)), bbox_inches='tight')
            test_command = os.system("convert -quality 100 -delay 20 {}/hehe_*.png {}/hehe_video.mp4".format(out_dir, out_dir))

        ax.quiver(G_sample_cpu[:, 0], G_sample_cpu[:, 1], gd_cpu[:, 0], gd_cpu[:, 1], d_g_sample_cpu.cpu().data.numpy(),
                  units='xy')
        plt.savefig('{}/haha_{}.png'.format(out_dir, str(cnt).zfill(3)), bbox_inches='tight')

        if it % 5000 == 0:
            plt.savefig('{}/haha_{}.pdf'.format(out_dir, str(cnt).zfill(3)), bbox_inches='tight')
            test_command = os.system("convert -quality 100 -delay 20 {}/haha_*.png {}/haha_video.mp4".format(out_dir, out_dir))

        # plt.imshow(np.rollaxis(sample,0,3), interpolation='nearest')
        plt.close()
        cnt += 1

        torch.save(G.state_dict(), "{}/G.model".format(out_dir))
        torch.save(D.state_dict(), "{}/D.model".format(out_dir))
