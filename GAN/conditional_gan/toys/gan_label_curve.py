# from curve_triplet
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

out_dir = './out/cgan_label_right_{}'.format(datetime.now())
out_dir = out_dir.replace(" ", "_")
print(out_dir)

if not os.path.exists(out_dir):
    os.makedirs(out_dir)
    shutil.copyfile(sys.argv[0], out_dir + '/training_script.py')

sys.stdout = mutil.Logger(out_dir)
gpu = 0
torch.cuda.set_device(gpu)
mb_size = 600  # mini-batch_size
mode_num = 3

distance = 10
R = np.array([1, 2, 3])*2
theta = np.array([[-15, 15], [75, 105], [165, 195]])
# theta[:, 0] = theta[:, 0] - 15
# theta[:, 1] = theta[:, 1] + 15
data = data_prepare.Data_2D_Curve(mb_size, theta, R)
grid_num = 100
data_mesh = data_prepare.Data_2D_Curve(grid_num*grid_num, theta,R)

Z_dim = 2
X_dim = 2
h_dim = 128
c_dim = mode_num
cnt = 0

num = '0'

# else:
#     print("you have already creat one.")
#     exit(1)

G = model.G_Net(Z_dim+c_dim, X_dim, h_dim).cuda()
D = model.D_Net(X_dim+c_dim, 1, h_dim).cuda()
# E = model.E_Net(X_dim+c_dim, 1, h_dim).cuda()
# G_fake = model.Direct_Net(X_dim+c_dim, 1, h_dim).cuda()
# path = "/home/bike/2027/generative-models/GAN/conditional_gan/toys/out/gan_label_curve_triplet_2017-05-27_05:07:26.528281"
# D_model = torch.load(path+"/D.model")
# G_model = torch.load(path+"/G.model")
# E_model = torch.load(path+'/E.model')

# D.load_state_dict(D_model)
# E.load_state_dict(E_model)
# G.load_state_dict(G_model)

G.apply(model.weights_init)
D.apply(model.weights_init)
# E.apply(model.weights_init)

""" ===================== TRAINING ======================== """

lr = 1e-4
G_solver = optim.Adam(G.parameters(), lr=lr)
D_solver = optim.Adam(D.parameters(), lr=lr)
# E_solver = optim.Adam(E.parameters(), lr=lr*10)

ones_label = Variable(torch.ones(mb_size)).cuda()
zeros_label = Variable(torch.zeros(mb_size)).cuda()

criterion = nn.BCELoss()
criterion_t = nn.TripletMarginLoss(p=1)

grads = {}
skip = (slice(None, None, 3), slice(None, None, 3))


def save_grad(name):
    def hook(grad):
        grads[name] = grad.data.cpu().numpy()

    return hook

def n2v(mm):
    return Variable(torch.from_numpy(mm.astype('float32'))).cuda()

def cat(m1,m2):
    return torch.cat([m1,m2],1)

# z_fixed = Variable(torch.randn(20, Z_dim), volatile=False).cuda()
# c_fixed = np.array(range(0,4))
# c_fixed = Variable(torch.from_numpy(mutil.label_num2vec(np.repeat(c_fixed,5)).astype("float32")),volatile=False).cuda()
# zc_fixed = torch.cat([z_fixed, c_fixed], 1)

z_fixed = Variable(torch.randn(mb_size, Z_dim)).cuda()
c_fixed = np.array(range(0, mode_num))
c_fixed = Variable(torch.from_numpy(mutil.label_num2vec(np.repeat(c_fixed, mb_size // (mode_num))).astype("float32")),
                   volatile=False).cuda()
# zc_fixed = torch.cat([z_fixed, c_fixed],1)
# zc_fixed = Variable(zc_fixed, volatile=False).cuda()



x_limit = 10
y_limit = 10
unit = x_limit / (float(grid_num))*2

y_fixed, x_fixed = np.mgrid[-x_limit:x_limit:unit, -y_limit:y_limit:unit]

x_fixed, y_fixed = x_fixed.reshape(grid_num * grid_num, 1), y_fixed.reshape(grid_num * grid_num, 1)
mesh_fixed_cpu = np.concatenate([x_fixed, y_fixed], 1)
mesh_fixed = Variable(torch.from_numpy(mesh_fixed_cpu.astype("float32")).cuda())


# mesh_fixed.register_hook(save_grad('Mesh'))

def get_grad(input, label, name, c=None, is_z=True, need_sample = False,loss = False):

    D.zero_grad()
    if (is_z):
        sample = G(torch.cat([input,c],1))
    else:
        input.requires_grad= True
        sample = input
    sample.register_hook(save_grad(name))
    if c is not None:
        d_result = D(torch.cat([sample,c],1))
    else:
        d_result = D(sample)

    ones_label_tmp = Variable(torch.ones([d_result.data.size()[0], 1])*label).cuda()
    loss_real = criterion(d_result, ones_label_tmp)
    loss_real.backward()
    D.zero_grad()
    G.zero_grad()
    if(need_sample):
        return d_result,sample
    elif (loss):
        return d_result, loss_real
    else:
        return d_result

def get_grad_E(input1,input2,input3, c1,c2,name,is_z = True,sample_seq=0):
    E.zero_grad()
    G.zero_grad()
    if not is_z:
        input1.requires_grad=True
        input2.requires_grad=True # may be not necessary
        input3.requires_grad=True # may be not necessary
        mesh_label = Variable(torch.ones([grid_num*grid_num,1])).cuda()

    input2.register_hook(save_grad(name))
    e_anch = E(torch.cat([input1,c1],1))
    e_real = E(torch.cat([input2,c1],1))
    e_fake = E(torch.cat([input3,c2],1))
    if not is_z:
        e_loss = criterion_t(e_anch,e_real,e_fake)
        # E_loss_t_a = criterion(e_anch, mesh_label)
        # E_loss_t_r = criterion(e_real, mesh_label)
        # E_loss_t_f = criterion(e_fake, mesh_label*0)
        # print("E_anch{}/{}/{}".format(E_loss_t_a.data.tolist(),E_loss_t_r.data.tolist(),E_loss_t_f.data.tolist()))
    else:
        # E_loss_t_a = criterion(e_anch,ones_label)
        # E_loss_t_r = criterion(e_real,ones_label)
        # E_loss_t_f = criterion(e_fake,zeros_label)
        e_loss =criterion_t(e_anch,e_real,e_fake)

    # e_loss = E_loss_t_a + E_loss_t_r + E_loss_t_f
    # e_loss = criterion_t(e_anch,e_real,e_fake)
    # e_loss.backward()
    # E_solver,backward()
    # loss_triplet = criterion_t()
    if(sample_seq):
        e_loss.backward(retain_variables = False)
    else:
        e_loss.backward(retain_variables=True) # for just keep the vairable

for it in range(100000):
    # Sample data
    z = Variable(torch.randn(mb_size, Z_dim)).cuda()
    X,c = data.batch_next(has_label=True)  # with label
    X = Variable(torch.from_numpy(X.astype('float32'))).cuda()
    c = Variable(torch.from_numpy(mutil.label_num2vec(c.astype('int')).astype('float32'))).cuda()

    D_solver.zero_grad()
    # Dicriminator forward-loss-backward-update
    G_sample = G(torch.cat([z,c],1))
    D_real = D(torch.cat([X,c],1))
    D_fake = D(torch.cat([G_sample,c],1))

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
    G_sample = G(torch.cat([z,c],1))
    G_sample.register_hook(save_grad('G'))
    # G_sample.requires_grad= True
    D_fake = D(torch.cat([G_sample,c],1))
    G_loss = criterion(D_fake, ones_label)

    G_loss.backward()
    G_solver.step()
    # print(grads['G'])

    # Housekeeping - reset gradient
    D.zero_grad()
    G.zero_grad()
    # E_solver.zero_grad()


    if it % 5000 == 0:
        #	print(zc_fixed_cpu)
        lr = lr * 0.9
        for param_group in G_solver.param_groups:
            param_group['lr'] = lr
        for param_group in D_solver.param_groups:
            param_group['lr'] = lr
        # for param_group in E_solver.param_groups:
        #     param_group['lr'] = lr*3

    mc = [None] * mode_num
    for i in range(mode_num):
        mc[i] = n2v(mutil.label_num2vec((np.ones([grid_num * grid_num, 1]) * i).astype('int')))

    # Print and plot every now and then
    if it % 1000 == 0:
        fig, ax = plt.subplots()

        print('Iter-{}; D_loss_real/fake: {}/{}; G_loss: {};'.format(it, D_loss_real.data.tolist(),
                                                                    D_loss_fake.data.tolist(), G_loss.data.tolist(),
                                                                                ))

        # data prepare
        # the total data.
        X_cpu = X.cpu().data.numpy()
        d_g_sample_cpu, G_sample = get_grad(z_fixed, 1, 'G', c=c_fixed,is_z=True,need_sample=True)
        G_sample_cpu = G_sample.cpu().data.numpy()
        # d_mesh = (get_grad(mesh_fixed, 1, 'mesh', is_z=False)).cpu().data.numpy()
        # gd_mesh_cpu = -grads['mesh'].cpu().data.numpy()
        # gd_mesh_cpu_x, gd_mesh_cpu_y = np.expand_dims(gd_mesh_cpu[:, 0], 1).reshape(grid_num, grid_num), np.expand_dims(
        # 	gd_mesh_cpu[:, 1], 1).reshape(grid_num, grid_num)
        # d_mesh = d_mesh.reshape(grid_num, grid_num)
        # x_fixed, y_fixed = x_fixed.reshape(grid_num, grid_num), y_fixed.reshape(grid_num, grid_num)
        # ax.quiver(x_fixed[::3, ::3], y_fixed[::3, ::3], gd_mesh_cpu_x[::3, ::3], gd_mesh_cpu_y[::3, ::3],
        #           d_mesh[::3, ::3], units='xy')
        # ax.set(aspect=1, title='3 mode Eloss{}_Dloss_{}.png'.format(np.round(E_loss_t.cpu().data.numpy(),2),np.round(float(D_loss_fake.cpu().data.numpy()),2)))
        plt.scatter(X_cpu[:, 0], X_cpu[:, 1], s=1, edgecolors='blue', color='blue')
        plt.scatter(G_sample_cpu[:, 0], G_sample_cpu[:, 1], s=1, color='red', edgecolors='red')
        plt.show()
        ax.set(adjustable='box-forced', aspect='equal')
        ax.set(title='cgan_{}'.format(it))
        plt.ylim((-y_limit, y_limit))
        plt.xlim((-x_limit, x_limit))
        if(it%5000==0):
            plt.savefig('{}/hehe_{}.pdf'.format(out_dir, str(cnt).zfill(3)), bbox_inches='tight')
        plt.savefig('{}/hehe_{}.png'.format(out_dir, str(cnt).zfill(3)), bbox_inches='tight')

        gd_cpu = -grads['G']
        ax.quiver(G_sample_cpu[:, 0], G_sample_cpu[:, 1], gd_cpu[:, 0], gd_cpu[:, 1], d_g_sample_cpu.cpu().data.numpy(),
                  units='xy')

        ax.set(title='3 mode CGAN_{}'.format(it))
        if(it%5000==0):
            plt.savefig('{}/haha_{}.pdf'.format(out_dir, str(cnt).zfill(3)), bbox_inches='tight')
        plt.savefig('{}/haha_{}.png'.format(out_dir, str(cnt).zfill(3)), bbox_inches='tight')
        plt.close()
        cnt += 1

        test_command = os.system("convert -quality 100 -delay 20 {}/hehe_*.png {}/video_hehe.mp4".format(out_dir, out_dir))
        test_command = os.system("convert -quality 100 -delay 20 {}/haha_*.png {}/video_haha.mp4".format(out_dir, out_dir))

        torch.save(G.state_dict(), "{}/G.model".format(out_dir))
        torch.save(D.state_dict(), "{}/D.model".format(out_dir))

        pic_num = 0

        gs = gridspec.GridSpec(1, 3)
        gs.update(wspace=0.5)

        d_mesh = [None]*mode_num
        loss_mesh = [None]*mode_num

        for i in range(mode_num):
            d_mesh[i], loss_mesh[i]=(get_grad(mesh_fixed.detach(),1,'mesh_{}'.format(i),c=mc[i],is_z=False, loss=True))
            d_mesh[i] = d_mesh[i].cpu().data.numpy()
            loss_mesh[i] = loss_mesh[i].cpu().data.numpy()

        for i in range(mode_num):
            ax = plt.subplot2grid((1,3),(0, i))
            # plt.axis('off')
            # ax.set_aspect('equal')
            ax.set(title ='mesh_{}'.format(i))
            ax.set(adjustable='box-forced', aspect='equal')
            gd_mesh_cpu = -grads['mesh_{}'.format(i)]

            gd_mesh_cpu_x, gd_mesh_cpu_y = np.expand_dims(gd_mesh_cpu[:, 0], 1).reshape(grid_num,
                                                                                        grid_num), np.expand_dims(
                gd_mesh_cpu[:, 1], 1).reshape(grid_num, grid_num)
            d_mesh[i] = d_mesh[i].reshape(grid_num, grid_num)
            x_fixed, y_fixed = x_fixed.reshape(grid_num, grid_num), y_fixed.reshape(grid_num, grid_num)
            ax.quiver(x_fixed[::3, ::3], y_fixed[::3, ::3], gd_mesh_cpu_x[::3, ::3], gd_mesh_cpu_y[::3, ::3], d_mesh[i],units='xy')

            plt.scatter(X_cpu[:, 0], X_cpu[:, 1], s=1, edgecolors='blue', color='blue')
            plt.scatter(G_sample_cpu[:, 0], G_sample_cpu[:, 1], s=1, color='red', edgecolors='red')
            plt.ylim((-y_limit, y_limit))
            plt.xlim((-x_limit, x_limit))
            pic_num+=1
            # plt.tight_layout()

            plt.savefig('{}/huhu_{}.png'.format(out_dir, str(cnt).zfill(3)), bbox_inches='tight')
            if it % 5000 == 0:
                plt.savefig('{}/huhu_{}.pdf'.format(out_dir, str(cnt).zfill(3)), bbox_inches='tight')
            # plt.imshow(np.rollaxis(sample,0,3), interpolation='nearest')

            test_command = os.system("convert -quality 100 -delay 20 {}/huhu_*.png {}/video_huhu.mp4".format(out_dir, out_dir))

        plt.close()