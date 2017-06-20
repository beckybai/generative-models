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
import mnist_model as model
import data_prepare

out_dir = './out/mnist_10_{}'.format(datetime.now())
out_dir = out_dir.replace(" ", "_")
print(out_dir)

if not os.path.exists(out_dir):
    os.makedirs(out_dir)
    shutil.copyfile(sys.argv[0], out_dir + '/training_script.py')
    shutil.copyfile("./toy_model.py", out_dir + "/toy_model.py")
    shutil.copyfile("./data_prepare.py", out_dir + "/data_prepare.py")
    shutil.copyfile("./mnist_baseline.py", out_dir+ "/mnist_baseline.py")
    # shutil.copyfile("./gan_reproduce.py", out_dir+ "/gan_reproduce.py")
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
Z_dim = 100
X_dim = 2
h_dim = 128

# data = data_prepare.Straight_Line(90, start_points, end_points, type=1)
# data = data_prepare.Data_2D_Circle(mb_size,R = 2)
data = data_prepare.Mnist_10(mb_size)
# tmp_data = data.batch_next()
# mutil.save_picture_numpy(tmp_data,"./test.png")
z_draw = Variable(torch.randn(mb_size, Z_dim)).cuda()



# c_dim = mode_num * mode_num
cnt = 0

num = '0'
# else:
#     print("you have already creat one.")
#     exit(1)
grid_num = 100

top_line = 2
down_line =-2
td_interval = (top_line-down_line)/100.0

left_line = -2
right_line = 2
lr_interval = (right_line-left_line)/100.0



G = model.G_Net_conv(in_channel=Z_dim).cuda()
D = model.D_Net_conv(inchannel=1).cuda()

# G_fake = model.Direct_Net(X_dim+c_dim, 1, h_dim).cuda()
# G.apply(model.weights_init)
# D.apply(model.weights_init)

""" ===================== TRAINING ======================== """

lr = 1e-4
G_solver = optim.Adam(G.parameters(), lr=1e-4,betas=[0.5,0.999])
D_solver = optim.Adam(D.parameters(), lr=1e-4,betas=[0.5,0.999])

ones_label = Variable(torch.ones(mb_size)).cuda()
zeros_label = Variable(torch.zeros(mb_size)).cuda()

criterion = nn.BCELoss()


for it in range(100000):
    # Sample data

    z = Variable(torch.randn(mb_size, Z_dim,1,1)).cuda()

    X = data.batch_next(mb_size, shuffle=False)  # with label
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
    z = Variable(torch.randn(mb_size, Z_dim,1,1).cuda(), requires_grad=True)
    # print(c.cpu().data.numpy().shape)
    G_sample = G(z)
    # G_sample.register_hook(save_grad('G'))
    # G_sample.requires_grad= True
    D_fake = D(G_sample)
    G_loss = criterion(D_fake, ones_label)

    G_loss.backward()
    G_solver.step()
    # Housekeeping - reset gradient
    D.zero_grad()
    G.zero_grad()
    if it % 5000 == 0:
        for param_group in G_solver.param_groups:
            param_group['lr'] = param_group['lr'] * 0.8
        for param_group in D_solver.param_groups:
            param_group['lr'] = param_group['lr'] * 0.5

    # Print and plot every now and then
    if it % 200 == 0:
        fig, ax = plt.subplots()

        print('Iter-{}; D_accuracy_real/fake: {}/{}; G_accuracy: {}'.format(it, np.round(np.exp(-D_loss_real.data.tolist()[0]), 5),
            np.round(1 - np.exp(-D_loss_fake.data.tolist()[0]), 5), np.round(np.exp(-G_loss.data.tolist()[0]), 5)))
        X = X.cpu().data.numpy()
        G_sample = G(z)

        mutil.save_picture(G_sample,'{}/hehe_{}.png'.format(out_dir, str(cnt).zfill(3)),column=10)
        cnt += 1

        # test_command = os.system("convert -quality 100 -delay 20 {}/*.png {}/video.mp4".format(out_dir, out_dir))
        torch.save(G.state_dict(), "{}/G.model".format(out_dir))
        torch.save(D.state_dict(), "{}/D.model".format(out_dir))
