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

out_dir = './out/d3_naive{}'.format(datetime.now())
out_dir = out_dir.replace(" ", "_")
print(out_dir)

if not os.path.exists(out_dir):
    os.makedirs(out_dir)
    shutil.copyfile(sys.argv[0], out_dir + '/training_script.py')
    shutil.copyfile("./toy_model.py", out_dir + "/toy_model.py")
    shutil.copyfile("./data_prepare.py", out_dir + "/data_prepare.py")
    shutil.copyfile("./gan_reproduce.py", out_dir+ "/gan_reproduce.py")
    shutil.copyfile("./gan.py", out_dir+ "/gan.py")

sys.stdout = mutil.Logger(out_dir)
gpu = 0
torch.cuda.set_device(gpu)
mb_size = 100  # mini-batch_size
# mode_num = 2
sample_point = 10000

# distance = 10
# start_points = np.array([[0,0],[0,1],[0,2]])
# end_points = np.array([[1,0],[1,1],[1,2]])
start_points = np.array([[0,0]])
end_points = np.array([[1,0]])
Z_dim = 10
X_dim = 10
h_dim = 16
dim = 10

# data = data_prepare.Straight_Line(90, start_points, end_points, type=1)

# attention ! here mode_num and dimension have nearly the same physical meaning!!!
data = data_prepare.Data_HD_Circle(mb_size,mode_num=2*dim, R = 2,dimension = dim)
data_draw_m = data_prepare.Data_HD_Circle(2*dim, R=2,mode_num=2*dim, dimension = dim)
# data_draw_m = data_prepare.Data_2D_Circle(8,R=2)
data_draw = data_draw_m.batch_next()

z_draw = Variable(torch.randn(sample_point, Z_dim)).cuda()



# c_dim = mode_num * mode_num
cnt = 0

num = '0'
# else:
#     print("you have already creat one.")
#     exit(1)
grid_num = 100

top_line = 3
down_line =-3
td_interval = (top_line-down_line)/100.0

left_line = -3
right_line = 3
lr_interval = (right_line-left_line)/100.0



G = model.G_Net(Z_dim, X_dim, h_dim).cuda()
D = model.D_Net(X_dim, 1, h_dim).cuda()

# G_fake = model.Direct_Net(X_dim+c_dim, 1, h_dim).cuda()
# G.apply(model.weights_init)
# D.apply(model.weights_init)

""" ===================== TRAINING ======================== """

lr = 1e-4
G_solver = optim.Adam(G.parameters(), lr=1e-4,betas=[0.5,0.999])
# G_solver = optim.SGD(G.parameters(), lr=1e-3)
D_solver = optim.Adam(D.parameters(), lr=1e-4,betas=[0.5,0.999])
# D_solver = optim.SGD(D.parameters(), lr=1e-3)

ones_label = Variable(torch.ones(mb_size)).cuda()
zeros_label = Variable(torch.zeros(mb_size)).cuda()

criterion = nn.BCELoss()

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
# c_fixed = np.array(range(0, mode_num * mode_num))
# c_fixed = Variable(
#     torch.from_numpy(mutil.label_num2vec(np.repeat(c_fixed, mb_size // (mode_num * mode_num))).astype("float32")),
#     volatile=False).cuda()
# zc_fixed = torch.cat([z_fixed, c_fixed],1)
# zc_fixed = Variable(zc_fixed, volatile=False).cuda()


y_fixed, x_fixed = np.mgrid[down_line:top_line:td_interval, right_line:left_line:-lr_interval]

# y_fixed, x_fixed = np.mgrid[ left_line:right_line:lr_interval,top_line:down_line:-td_interval]

x_fixed, y_fixed = x_fixed.reshape(grid_num * grid_num, 1), y_fixed.reshape(grid_num * grid_num, 1)
mesh_fixed_cpu = np.zeros([grid_num*grid_num,dim])
mesh_fixed_cpu[:,0:2] = np.concatenate([x_fixed, y_fixed], 1)
mesh_fixed = Variable(torch.from_numpy(mesh_fixed_cpu.astype("float32")).cuda())


# mesh_fixed.register_hook(save_grad('Mesh'))

def get_grad(input, label, name, c=False, is_z=True, need_sample=False):
    D.zero_grad()
    if (is_z):
        sample = G(input)
    else:
        input.requires_grad = True
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
    if (need_sample):
        return d_result, sample
    else:
        return d_result


for it in range(100000):
    # Sample data
    z = Variable(torch.randn(mb_size, Z_dim)).cuda()
    X = data.batch_next()  # with label
    X = Variable(torch.from_numpy(X.astype('float32'))).cuda()
    #	c = Variable(torch.from_numpy(mutil.label_num2vec(c.astype('int')).astype('float32'))).cuda()

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

    if it % 5000 == 0:
        #	print(zc_fixed_cpu)
        lr = lr * 0.8
        for param_group in G_solver.param_groups:
            param_group['lr'] = param_group['lr'] * 0.9
        for param_group in D_solver.param_groups:
            param_group['lr'] = param_group['lr'] * 0.9

    G.zero_grad()

    # Print and plot every now and then
    if it % 2000 == 0:
        fig, ax = plt.subplots()

        print('Iter-{}; D_accuracy_real/fake: {}/{}; G_accuracy: {}'.format(it, np.round(np.exp(-D_loss_real.data.tolist()[0]), 5),
            np.round(1 - np.exp(-D_loss_fake.data.tolist()[0]), 5), np.round(np.exp(-G_loss.data.tolist()[0]), 5)))
        X = X.cpu().data.numpy()
        G_sample = G(z_draw)
        G_sample_cpu = G_sample.cpu().data.numpy()
        mutil.draw_stat(G_sample_cpu, data_draw, '{}/haha_{}.png'.format(out_dir, str(cnt).zfill(3)))

        d_mesh = (get_grad(mesh_fixed, 1, 'mesh', is_z=False)).cpu().data.numpy()

        gd_mesh_cpu = -grads['mesh'].cpu().data.numpy()[:,0:2]

        gd_mesh_cpu_x, gd_mesh_cpu_y = np.expand_dims(gd_mesh_cpu[:, 0], 1).reshape(grid_num, grid_num), np.expand_dims(
            gd_mesh_cpu[:, 1], 1).reshape(grid_num, grid_num)
        d_mesh = d_mesh.reshape(grid_num, grid_num)
        x_fixed, y_fixed = x_fixed.reshape(grid_num, grid_num), y_fixed.reshape(grid_num, grid_num)
        ax.quiver(x_fixed[::3, ::3], y_fixed[::3, ::3], gd_mesh_cpu_x[::3, ::3], gd_mesh_cpu_y[::3, ::3],
                  d_mesh[::3, ::3], units='xy')

        ax.set(aspect=1, title="3d_6mode_{}".format(it))

        plt.scatter(X[:, 0], X[:, 1], s=1, edgecolors='blue', color='blue')
        plt.scatter(G_sample_cpu[:, 0], G_sample_cpu[:, 1], s=0.2, color='red', edgecolors='red', alpha = 0.1)
        plt.show()
        plt.ylim((down_line, top_line))
        plt.xlim((left_line, right_line))
        plt.savefig('{}/hehe_{}.png'.format(out_dir, str(cnt).zfill(3)), bbox_inches='tight', dpi=400)
        plt.close()
        cnt += 1

        # test_command = os.system("convert -quality 100 -delay 20 {}/*.png {}/video.mp4".format(out_dir, out_dir))
        torch.save(G.state_dict(), "{}/G.model".format(out_dir))
        torch.save(D.state_dict(), "{}/D.model".format(out_dir))
