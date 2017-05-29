# add negative gridient

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

out_dir = './out/6_mode_gan_triplet_{}'.format(datetime.now())
out_dir = out_dir.replace(" ", "_")
print(out_dir)

if not os.path.exists(out_dir):
    os.makedirs(out_dir)
    shutil.copyfile(sys.argv[0], out_dir + '/training_script.py')

sys.stdout = mutil.Logger(out_dir)
gpu = 1
torch.cuda.set_device(gpu)
mb_size = 600  # mini-batch_size
mode_num = 6

distance = 10
R = np.array([2, 4, 6,8,10,12])
theta = np.array([[-15, 15], [75, 105], [165, 195],[0,30],[90,120],[180,205]])
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
D = model.D_Net(X_dim, 1, h_dim).cuda()
E = model.E_Net(X_dim, 10, h_dim).cuda()
# G_fake = model.Direct_Net(X_dim+c_dim, 1, h_dim).cuda()
path = "/home/bike/2027/generative-models/GAN/conditional_gan/toys/out/6_mode_gan_triplet_2017-05-27_06:12:43.974674/"
D_model = torch.load(path+'D.model')
G_model = torch.load(path+'G.model')
E_model = torch.load(path+'E.model')

G.load_state_dict(G_model)
D.load_state_dict(D_model)
E.load_state_dict(E_model)
# G.apply(model.weights_init)
# D.apply(model.weights_init)
# E.apply(model.weights_init)

""" ===================== TRAINING ======================== """

lr = 1e-5
G_solver = optim.Adam(G.parameters(), lr=lr)
D_solver = optim.Adam(D.parameters(), lr=lr)
E_solver = optim.Adam(E.parameters(), lr=lr*3)


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



x_limit = 16
y_limit = 16
unit = x_limit / (float(grid_num))*2

y_fixed, x_fixed = np.mgrid[-x_limit:x_limit:unit, -y_limit:y_limit:unit]

x_fixed, y_fixed = x_fixed.reshape(grid_num * grid_num, 1), y_fixed.reshape(grid_num * grid_num, 1)
mesh_fixed_cpu = np.concatenate([x_fixed, y_fixed], 1)
mesh_fixed = Variable(torch.from_numpy(mesh_fixed_cpu.astype("float32")).cuda())


# mesh_fixed.register_hook(save_grad('Mesh'))

def get_grad(input, label, name, c=False, is_z=True, need_sample = False):

    D.zero_grad()
    if (is_z):
        sample = G(torch.cat([input,c],1))
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

def get_grad_E(input1,input2,input3, c1,c2,name,is_z = True,sample_seq=0):
    E.zero_grad()
    G.zero_grad()
    if not is_z:
        input1.requires_grad=True
        input2.requires_grad=True # may be not necessary
        input3.requires_grad=True # may be not necessary
        mesh_label = Variable(torch.ones([grid_num*grid_num,1])).cuda()

    input2.register_hook(save_grad(name))
    e_anch = E(input1)
    e_real = E(input2)
    e_fake = E(input3)
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
    G_sample = G(torch.cat([z,c],1))
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
    E_solver.zero_grad()

    X, c = data.batch_next(has_label=True)  # with label
    X = Variable(torch.from_numpy(X.astype('float32'))).cuda()
    # c = np.zeros(mb_size)
    c = Variable(torch.from_numpy(
        mutil.label_num2vec(c.astype('int')).astype('float32'))).cuda()
    E_anch = E(X)


    X2, c2 = data.batch_next(has_label=True,negative=False)
    c2 = Variable(torch.from_numpy(
        mutil.label_num2vec(c2.astype('int')).astype('float32'))).cuda()
    assert (torch.sum(c==c2)==mb_size)
    X2 = Variable(torch.from_numpy(X2.astype("float32"))).cuda()
    # c2 = Variable(c2).cuda()
    # c2 = Variable(torch.from_numpy(
    #     mutil.label_num2vec(c2.astype('int')).astype('float32'))).cuda()
    E_real = E(X2)

    X3, c3 = data.batch_next(has_label=True,negative=True)  # with label
    X3 = Variable(torch.from_numpy(X3.astype('float32'))).cuda()
    # c = np.zeros(mb_size)
    c3 = Variable(torch.from_numpy(
        mutil.label_num2vec(c3.astype('int'), max_label=mode_num).astype('float32'))).cuda()
    assert (torch.sum(c==c3)==0)
    E_fake = E(X3)
    E_loss_t = criterion_t(E_anch,E_real,E_fake)
    # E_loss_t_a = criterion(E_anch,ones_label)
    # E_loss_t_r = criterion(E_real,ones_label)
    # E_loss_t_f = criterion(E_fake,zeros_label)
    # E_loss_t = E_loss_t_a + E_loss_t_r + E_loss_t_f
    E_loss_t.backward()
    E_solver.step()

    # Housekeeping - reset gradient
    # D.zero_grad()
    G_solver.zero_grad()
    E_solver.zero_grad()

    # For E ( generated data )
    z = Variable(torch.randn(mb_size, Z_dim)).cuda()
    x_g = torch.cat([z, c.float()], 1)
    G_sample = G(x_g)
    E_real = E(G_sample)
    # X3, c3 = celebA.batch_next(mb_size, label= c.data, shuffle=False, Negative = True)
    x_g2 = torch.cat([z, c3], 1)
    G_sample2 = G(x_g2)
    E_fake = E(G_sample2)

    E_anch = E(X)
    E_loss_g = criterion_t(E_anch,E_real,E_fake)

    E_loss_g.backward()
    G_solver.step()

    if it % 5000 == 0:
        #	print(zc_fixed_cpu)
        lr = lr * 0.9
        for param_group in G_solver.param_groups:
            param_group['lr'] = lr
        for param_group in D_solver.param_groups:
            param_group['lr'] = lr
        for param_group in E_solver.param_groups:
            param_group['lr'] = lr*3

    # Print and plot every now and then
    if it % 1000 == 0:
        fig, ax = plt.subplots()

        print('Iter-{}; D_loss_real/fake: {}/{}; G_loss: {}; E_loss:{}/{}'.format(it, D_loss_real.data.tolist(),
                                                                    D_loss_fake.data.tolist(), G_loss.data.tolist(),
                                                                                  E_loss_t.data.tolist(),E_loss_g.data.tolist()))

        # data prepare
        # the total data.
        X_cpu = X.cpu().data.numpy()
        d_g_sample_cpu, G_sample = get_grad(z_fixed, 1, 'G', c=c_fixed,is_z=True,need_sample=True)
        G_sample_cpu = G_sample.cpu().data.numpy()
        d_mesh = (get_grad(mesh_fixed, 1, 'mesh', is_z=False)).cpu().data.numpy()
        gd_mesh_cpu = -grads['mesh'].cpu().data.numpy()
        gd_mesh_cpu_x, gd_mesh_cpu_y = np.expand_dims(gd_mesh_cpu[:, 0], 1).reshape(grid_num, grid_num), np.expand_dims(
        	gd_mesh_cpu[:, 1], 1).reshape(grid_num, grid_num)
        d_mesh = d_mesh.reshape(grid_num, grid_num)
        x_fixed, y_fixed = x_fixed.reshape(grid_num, grid_num), y_fixed.reshape(grid_num, grid_num)
        ax.quiver(x_fixed[::3, ::3], y_fixed[::3, ::3], gd_mesh_cpu_x[::3, ::3], gd_mesh_cpu_y[::3, ::3],
                  d_mesh[::3, ::3], units='xy')
        ax.set(aspect=1, title='6 mode_Triplet_{}.png'.format(it))
        # ax.set(aspect=1, title='3 mode_Triplet_Eloss_{}_Dloss_{}.png'.format(np.round(float(E_loss_t.cpu().data.numpy()),2),np.round(float(D_loss_fake.cpu().data.numpy()),2)))
        plt.scatter(X_cpu[:, 0], X_cpu[:, 1], s=1, edgecolors='blue', color='blue')
        plt.scatter(G_sample_cpu[:, 0], G_sample_cpu[:, 1], s=1, color='red', edgecolors='red')
        plt.show()
        plt.ylim((-y_limit, y_limit))
        plt.xlim((-x_limit, x_limit))
        if it%5000==0:
            plt.savefig('{}/hehe_{}.pdf'.format(out_dir, str(cnt).zfill(3)), bbox_inches='tight')
        plt.savefig('{}/hehe_{}.png'.format(out_dir, str(cnt).zfill(3)), bbox_inches='tight')
        gd_cpu = -grads['G'].cpu().data.numpy()
        ax.quiver(G_sample_cpu[:, 0], G_sample_cpu[:, 1], gd_cpu[:, 0], gd_cpu[:, 1], d_g_sample_cpu.cpu().data.numpy(),
                  units='xy')
        if it%5000==0:
            plt.savefig('{}/haha_{}.pdf'.format(out_dir, str(cnt).zfill(3)), bbox_inches='tight')
            torch.save(G.state_dict(), "{}/G.model".format(out_dir))
            torch.save(D.state_dict(), "{}/D.model".format(out_dir))
            torch.save(E.state_dict(), "{}/E.model".format(out_dir))
        plt.savefig('{}/haha_{}.png'.format(out_dir, str(cnt).zfill(3)), bbox_inches='tight')

        plt.close()
        cnt += 1

        test_command = os.system("convert -quality 100 -delay 20 {}/hehe_*.png {}/video_hehe.mp4".format(out_dir, out_dir))
        test_command = os.system("convert -quality 100 -delay 20 {}/haha_*.png {}/video_haha.mp4".format(out_dir, out_dir))



        # Copy from another file
        tot_pic_num = 30 # three condition: c- anti c1 - anti  c2
        row_num = 6
        fig = plt.figure()
        # fig = plt.figure(figsize=(row_num, tot_pic_num//row_num))
        gs = gridspec.GridSpec(row_num, tot_pic_num//row_num, wspace=40.0, hspace=40.0)
        gs.update(wspace=0.5, hspace=1)

        xc = [None]*mode_num
        cc = [None]*mode_num
        zz = [None]*mode_num
        mc = [None]*mode_num
        xc_copy = [None]*mode_num

        for i in range(mode_num):
            _,cc[i]= data.batch_next(has_label=True,selected_label=i+1)
            xc[i] = data_mesh.batch_next(has_label=False,selected_label=i+1)
            xc_copy[i] = data_mesh.batch_next(has_label=False,selected_label=i+1)
            xc[i],cc[i] = n2v(xc[i]), n2v(mutil.label_num2vec(cc[i].astype('int'),max_label=mode_num-1))
            xc_copy[i] = n2v(xc_copy[i])
            zz[i] = Variable(torch.randn(mb_size, Z_dim)).cuda()
            mc[i] = n2v(mutil.label_num2vec((np.ones([grid_num*grid_num,1])*i).astype('int'),max_label=mode_num-1))

            # def get_grad_E(input1, input2, input3, c1, c2, name, need_sample=False):

        gg = [[None]*mode_num]*mode_num

        for i in range(mode_num):
            for j in range(mode_num):
                # gg[i,j] = G(torch.cat([zz[i],cc[j]],1))
                gg[i][j]=G(torch.cat([zz[i],cc[j]],1))

        for j in range(mode_num):
            for i in range(mode_num-1):
                get_grad_E(xc[j].detach(),mesh_fixed.detach(),xc[(j+i+1)%mode_num].detach(),mc[j],mc[(j+i+1)%mode_num],"mesh_{}_{}".format(str(j),str((j+i+1)%mode_num)),is_z = False)
            # get_grad_E(xc[j].detach(),mesh_fixed.detach(),xc[(j+2)%3].detach(),mc[j],mc[(j+2)%3],"mesh_{}_{}".format(str(j),str((j+2)%3)),is_z = False)
                get_grad_E(xc[j].detach(),xc_copy[j].detach(),mesh_fixed.detach(),mc[j],mc[(j+i+1)%mode_num],"mesh_false_{}_{}".format(str(j),str((j+i+1)%mode_num)),is_z = False)
            # get_grad_E(xc[j].detach(),xc_copy[j].detach(),mesh_fixed.detach(),mc[j],mc[(j+2)%3],"mesh_false_{}_{}".format(str(j),str((j+2)%3)),is_z = False)



        pic_num = 0

        for i in range(mode_num):
            for j in range(mode_num-1): # just for cycling, no specific meaning for j.
                ax = plt.subplot(gs[pic_num])
                # plt.axis('off')
                # ax.set_aspect('equal')
                ax.set(title ='mesh_{}_{}'.format(i,(i+j+1)%mode_num))
                ax.set(adjustable='box-forced', aspect='equal')
                gd_mesh_cpu = -grads['mesh_{}_{}'.format(i,(i+j+1)%mode_num)].cpu().data.numpy()
                gd_mesh_cpu_x, gd_mesh_cpu_y = np.expand_dims(gd_mesh_cpu[:, 0], 1).reshape(grid_num,
                                                                                            grid_num), np.expand_dims(
                    gd_mesh_cpu[:, 1], 1).reshape(grid_num, grid_num)
                d_mesh = d_mesh.reshape(grid_num, grid_num)
                x_fixed, y_fixed = x_fixed.reshape(grid_num, grid_num), y_fixed.reshape(grid_num, grid_num)
                ax.quiver(x_fixed[::3, ::3], y_fixed[::3, ::3], gd_mesh_cpu_x[::3, ::3], gd_mesh_cpu_y[::3, ::3], units='xy')
                plt.scatter(X_cpu[:, 0], X_cpu[:, 1], s=1, edgecolors='blue', color='blue')
                plt.scatter(G_sample_cpu[:, 0], G_sample_cpu[:, 1], s=1, color='red', edgecolors='red')
                plt.ylim((-y_limit, y_limit))
                plt.xlim((-x_limit, x_limit))
                pic_num+=1
                # we have X X1,X2
        plt.savefig('{}/huhu_{}.png'.format(out_dir, str(cnt).zfill(3)), bbox_inches='tight')
        if it%5000==0:
            plt.savefig('{}/huhu_{}.pdf'.format(out_dir, str(cnt).zfill(3)), bbox_inches='tight')
        plt.close()
        test_command = os.system("convert -quality 100 -delay 20 {}/huhu_*.png {}/video_huhu.mp4".format(out_dir, out_dir))

        pic_num = 0
        for i in range(mode_num):
            for j in range(mode_num-1): # just for cycling, no specific meaning for j.
                ax = plt.subplot(gs[pic_num])
                # plt.axis('off')
                # ax.set_aspect('equal')
                ax.set(title ='mesh_false_{}_{}'.format(i,(i+j+1)%mode_num))
                ax.set(adjustable='box-forced', aspect='equal')

                gd_mesh_cpu = -grads['mesh_false_{}_{}'.format(i,(i+j+1)%mode_num)].cpu().data.numpy()
                gd_mesh_cpu_x, gd_mesh_cpu_y = np.expand_dims(gd_mesh_cpu[:, 0], 1).reshape(grid_num,
                                                                                            grid_num), np.expand_dims(
                    gd_mesh_cpu[:, 1], 1).reshape(grid_num, grid_num)
                d_mesh = d_mesh.reshape(grid_num, grid_num)
                x_fixed, y_fixed = x_fixed.reshape(grid_num, grid_num), y_fixed.reshape(grid_num, grid_num)
                ax.quiver(x_fixed[::3, ::3], y_fixed[::3, ::3], gd_mesh_cpu_x[::3, ::3], gd_mesh_cpu_y[::3, ::3], units='xy')
                plt.scatter(X_cpu[:, 0], X_cpu[:, 1], s=1, edgecolors='blue', color='blue')
                plt.scatter(G_sample_cpu[:, 0], G_sample_cpu[:, 1], s=1, color='red', edgecolors='red')
                plt.ylim((-y_limit, y_limit))
                plt.xlim((-x_limit, x_limit))
                pic_num+=1
                # we have X X1,X2
        plt.savefig('{}/hoho_{}.png'.format(out_dir, str(cnt).zfill(3)), bbox_inches='tight')
        if it%5000==0:
            plt.savefig('{}/hoho_{}.pdf'.format(out_dir, str(cnt).zfill(3)), bbox_inches='tight')
        plt.close()
        test_command = os.system("convert -quality 100 -delay 20 {}/huhu_*.png {}/video_huhu.mp4".format(out_dir, out_dir))

            # plt.imshow(np.rollaxis(sample,0,3), interpolation='nearest')
