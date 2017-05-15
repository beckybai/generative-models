import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import datasets, transforms
import torch.utils.data
import torch.optim as optim
import owntool
import model
import mutil
from datetime import datetime
from tensorflow.examples.tutorials.mnist import input_data
import data_convert

import os
import sys,shutil

# chagne today : 2017/5/9
ngpu = 1
if __name__ == '__main__':
    hidden_d = 128 # the dimension of the feature map in the first layer

main_gpu = 7

column = 10
torch.cuda.set_device(main_gpu)

mnist = input_data.read_data_sets('../../MNIST_data', one_hot=False)
mm = data_convert.owndata()

seed_num = 1
torch.cuda.manual_seed(seed_num)
data_dir = '/home/bike/data/mnist/'
out_dir = '/home/bike/2027/generative-models/GAN/conditional_gan/new_f3_no_label_{}/'.format(datetime.now())

if not os.path.exists(out_dir):
    os.makedirs(out_dir)
    shutil.copyfile(sys.argv[0], out_dir + '/new_mnist_feature3_no_label.py')

sys.stdout = mutil.Logger(out_dir)

batch_size = 50
feature_size = 32

kwargs = {'num_workers': 1, 'pin_memory': True}

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    # elif classname.find('BatchNorm') != -1:
    #     m.weight.data.normal_(1.0, 0.02)
    #     m.bias.data.fill_(0)


# layer_names = list(model.keys())

# write the origin model again.

# D0 = model.E_Net_small()
encoder_model = torch.load("/home/bike/2027/generative-models/tools/D_mnist_little3.model")

# 24*24
enet = model.E_Net_small().cuda()
enet.load_state_dict(encoder_model)

add_in_feature = feature_size+ hidden_d # Add one dimension data for the input_feature data.
G = model.G_Net_FM_3(ngpu,add_in_feature,main_gpu=main_gpu).cuda()
# g_model = torch.load("/home/bike/2027/generative-models/GAN/conditional_gan/mnist_pretrain_DG/G_60000.model")
# G.load_state_dict(g_model)

in_channel = 1
E = model.Ev_Net_conv(ngpu,in_channel,main_gpu=main_gpu).cuda()
# E.apply(weights_init)
# G.apply(weights_init)

d_in_demension = 1
D = model.D_Net_conv(ngpu,d_in_demension,main_gpu=main_gpu).cuda()
# d_model = torch.load("/home/bike/2027/generative-models/GAN/conditional_gan/mnist_pretrain_DG/D_60000.model")
# D.load_state_dict(d_model)

G_solver = optim.Adam(G.parameters(),lr = 1e-4)
D_solver = optim.Adam(D.parameters(), lr = 1e-4)
E_solver = optim.Adam(E.parameters(), lr = 1e-4)

half_label = Variable(torch.ones(batch_size)*0.5).cuda()

check_points = 500
num_epoches = 100000
criterion = nn.BCELoss()
criterion_t = nn.TripletMarginLoss(p=1)
criterion_mse = nn.MSELoss()

for epoch in (range(1,num_epoches)):
# D PART ./ data_old/ d_f3 / error
    data, label = mm.batch_next(batch_size,label = [8],  shuffle=False)
    D_solver.zero_grad()
    G_solver.zero_grad()
    data_old = Variable(torch.from_numpy(data.astype('float32'))).cuda()
    data_old.data.resize_(batch_size, 1, 28, 28)
    _,_,_,f3 = enet(data_old.detach())
    d_f3 = f3.cuda()
    g_sampler = Variable((torch.randn([batch_size,hidden_d,1,1]))).cuda()
    d_f3.data.resize_(batch_size,feature_size,1,1)
    d_f3 = d_f3.detach()
    d_f3_output = G(torch.cat([d_f3,g_sampler],1)).detach()
    d_false_decision = D(d_f3_output)
    d_false_error = criterion(d_false_decision,Variable(torch.zeros(batch_size)).cuda())

    d_real_decision = D(data_old)
    d_real_error = criterion(d_real_decision,Variable(torch.ones(batch_size)).cuda())

    # d_real_false_error = criterion(d_real_decision/2 + d_false_decision/2, half_label)
    d_real_error.backward()
    d_false_error.backward()
    # error = d_real_error + d_false_errord_false_error
    # error.backward()
    D_solver.step()

# G Part / data_g / g_f3 / dg_error
    D.zero_grad()
    G.zero_grad()
    data_g, label_g = mm.batch_next(batch_size,label = [8], shuffle=False)
    data_g = Variable(torch.from_numpy(data_g.astype('float32'))).cuda()
    # data_g.data.resize_(batch_size, 1, 28, 28)
    _,_,_,f3 = enet(data_g)
    g_f3 = f3.cuda()
    g_f3.data.resize_(batch_size,feature_size,1,1)
    g_f3 = g_f3.detach()

    g_sampler = Variable((torch.randn([batch_size,hidden_d,1,1]))).cuda()
    g_f3_output = G(torch.cat([g_f3,g_sampler],1))
    # G for fake data
    # c_v = Variable(model.set_label_f3(f3)).cuda()
    dg_decision = D(g_f3_output)
    dg_error = criterion(dg_decision,Variable(torch.ones([batch_size,1])).cuda())
    dg_error.backward()
    G_solver.step()
# Evaluator: Triplet Net
    # E part
    # For E, triplet part
    D_solver.zero_grad()
    G_solver.zero_grad()
    # E_solver.zero_grad()

#     X, c = mm.batch_next(batch_size, shuffle=False)
#     X = Variable(torch.from_numpy(X.astype('floatxccc'))).cuda()
#     E_anch = E(X)
#
#     X2, c2 = mm.batch_next(batch_size, shuffle=False)
#     X2 = Variable(torch.from_numpy(X2.astype('float32'))).cuda()
#     E_real = E(X2)
#
#     new_label_base = np.array(range(10))
#     new_label = (new_label_base + (np.random.rand(10) * 9 + 1).astype('int'))%10
#     X3, c3 = mm.batch_next(batch_size, label=new_label, shuffle=False)
#     X3 = Variable(torch.from_numpy(X3.astype('float32'))).cuda()
#     E_fake = E(X3)
#
#     E_loss = criterion_t(E_anch, E_real, E_fake)
#     E_loss.backward()
#     E_solver.step()
#
# # Evaluator: G
#     D.zero_grad()
#     E.zero_grad()
#
#     # For E ( generated data )
#     _,_,_,f3 = enet(X)
#     e_f3 = f3.cuda()
#     e_f3.data.resize_(batch_size,feature_size,1,1)
#     e_f3 = e_f3.detach()
#     # e_f3 = F.sigmoid(e_f3)
#     g_sampler = Variable((torch.randn([batch_size,hidden_d,1,1]))).cuda()
#     G_sample = G(torch.cat([e_f3,g_sampler],1))
#     E_real = E(G_sample)
#     E_anch = E(X)
#     # genereted (another kind of data)
#     # data_ol == data_other_label
#     new_label_base = np.array(range(10))
#     new_label = (new_label_base + (np.random.rand(10) * 9 + 1).astype('int'))%10
#     data_ol, label_ol = mm.batch_next(batch_size, label = new_label, shuffle=False)
#     data_ol = Variable(torch.from_numpy(data_ol.astype('float32'))).cuda()
#     # data_ol.data.resize_(batch_size, 1, 28, 28)
#     _,_,_,f3_fake = enet(data_ol)
#     g_f3_fake = f3_fake.cuda()
#     # g_f3_fake = F.sigmoid(g_f3_fake)
#     g_f3_fake.data.resize_(batch_size,feature_size,1,1)
#     g_f3_fake = g_f3_fake.detach()
#     g_sampler = Variable(torch.randn([batch_size,hidden_d,1,1])).cuda()
#     G_sample_fake = G(torch.cat([g_f3_fake,g_sampler],1))
#     E_fake = E(G_sample_fake)
#     # E_anch = E(X)
#     E_loss = criterion_t(E_anch, E_real, E_fake)
#     E_loss.backward()
#     G_solver.step()

    # Housekeeping - reset gradient
    # print("heelo")
    if(epoch%check_points==0):
        # observe the weight of two pictures
        zeroinput = Variable(torch.zeros([batch_size, hidden_d,1,1])).cuda()
        g_zero_output = G(torch.cat([g_f3, zeroinput], 1))

        g_sampler2 = Variable((torch.rand([batch_size,  hidden_d,1,1]))).cuda()
        g_f3_output1 = G(torch.cat([g_f3, g_sampler2], 1))



        # owntool.save_picture(data, out_dir + "recon_epoch{}_data.png".format(epoch ),column=column)
        owntool.save_picture(data_g, out_dir + "recon_epoch{}_data_old.png".format(epoch ),column=column)
        owntool.save_picture(g_zero_output, out_dir + "recon_epoch{}_zero.png".format(epoch),column=column)
        owntool.save_picture(g_f3_output1, out_dir + "recon_epoch{}_real.png".format(epoch),column=column)

        print("%s: D: %s/%s G: %s (Real: %s, Fake: %s) "% (epoch,
                                                       owntool.extract(d_real_error)[0],
                                                         owntool.extract(d_false_error)[0],
                                                         owntool.extract(dg_error)[0],
                                                         owntool.stats(owntool.extract(d_real_decision)),
                                                         owntool.stats(owntool.extract(d_false_decision)),
                                                         # owntool.stats(owntool.extract(E_loss)[0])
                                                         )
              )
    if(epoch%(check_points*10)==0):
        np.set_printoptions(precision=2,suppress=True)
        # print("real")
        # print( np.array(E_real.data.tolist()))
        # print("anch")
        # print( np.array(E_anch.data.tolist()))
        # print("fake")
        # print( np.array(E_fake.data.tolist()))
        torch.save(G.state_dict(), '{}/G_{}.model'.format(out_dir, str(epoch)))
        torch.save(D.state_dict(), '{}/D_{}.model'.format(out_dir, str(epoch)))