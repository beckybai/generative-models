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

ngpu = 1
if __name__ == '__main__':
    hidden_d = 240 # the dimension of the feature map in the first layer

main_gpu = 6
column = 10
torch.cuda.set_device(main_gpu)

mnist = input_data.read_data_sets('../../MNIST_data', one_hot=False)
mm = data_convert.owndata()

seed_num = 1
torch.cuda.manual_seed(seed_num)
data_dir = '/home/bike/data/mnist/'
out_dir = '/home/bike/2027/generative-models/GAN/conditional_gan/f3_no_label_{}/'.format(datetime.now())

if not os.path.exists(out_dir):
    os.makedirs(out_dir)
    shutil.copyfile(sys.argv[0], out_dir + '/feature3_no_label.py')

sys.stdout = mutil.Logger(out_dir)

batch_size = 50

kwargs = {'num_workers': 1, 'pin_memory': True}
train_loader =  torch.utils.data.DataLoader(
    datasets.MNIST(data_dir, train = True,download=False,
                   transform= transforms.Compose([transforms.ToTensor(),transforms.Normalize((0,),(1,))])),
    batch_size = batch_size, shuffle=True, **kwargs)

# layer_names = list(model.keys())

# write the origin model again.

D0 = model.E_Net_2()
encoder_model = torch.load("./Dmodel")

# 24*24
enet = model.E_Net_2().cuda()
enet.load_state_dict(encoder_model)

add_in_feature = 240+ hidden_d # Add one dimension data for the input_feature data.
G = model.G_Net_FM_3(ngpu,add_in_feature,main_gpu=main_gpu).cuda()
# g_model = torch.load("./fm21/G_95000.model")
# G.load_state_dict(g_model)

in_channel = 1
E = model.Ev_Net_conv(ngpu,in_channel,main_gpu=main_gpu).cuda()

d_in_demension = 1
D = model.D_Net_conv(ngpu,d_in_demension,main_gpu=main_gpu).cuda()
G_solver = optim.Adam(G.parameters(),lr = 1e-4)
D_solver = optim.Adam(D.parameters(), lr = 1e-4)
E_solver = optim.Adam(E.parameters(), lr=1e-4)

half_label = Variable(torch.ones(batch_size)*0.5).cuda()

check_points = 500
num_epoches = 100000
criterion = nn.BCELoss()
criterion_t = nn.TripletMarginLoss(p=1)
criterion_mse = nn.MSELoss()

for epoch in (range(1,num_epoches)):
    # print("give me a clue")
    # data, label = mnist.train.next_batch(batch_size)
    data, label = mm.batch_next(batch_size, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], shuffle=True)
    D.zero_grad()
    G.zero_grad()

    data_old = Variable(torch.from_numpy(data)).cuda()
    data_old.data.resize_(batch_size, 1, 28, 28)

    # data, label = Variable(data), Variable(label)
    _,_,_,f3 = enet(data_old.detach())
    d_f3 = f3.cuda()
    g_sampler = Variable((torch.randn([batch_size,hidden_d,1,1])-1)*2.5).cuda()
    d_f3.data.resize_(batch_size,240,1,1)
    d_f3 = d_f3.detach()
    d_f3_output = G(torch.cat([d_f3,g_sampler],1)).detach()
    # c_v = Variable((model.set_label_f3(f3))).cuda() # the f3 is a batch_size * 240 vector
    # d_false_decision = D(torch.cat([g_f3_output.detach(),c_v],1))
    d_false_decision = D(d_f3_output)
    d_false_error = criterion(d_false_decision,Variable(torch.zeros(batch_size)).cuda())

    # data, label = mm.batch_next(batch_size, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], shuffle=True)
    # data = Variable(torch.from_numpy(data)).cuda()
    # data.data.resize_(batch_size, 1, 28, 28)
    d_real_decision = D(data_old)
    d_real_error = criterion(d_real_decision,Variable(torch.ones(batch_size)).cuda())
    d_real_false_error = criterion(d_real_decision/2 + d_false_decision/2, half_label)
    error = d_real_error + d_false_error + d_real_false_error
    error.backward()
    D_solver.step()

    # G Part
    D.zero_grad()
    G.zero_grad()
    data_g, label_g = mm.batch_next(batch_size,  shuffle=True)
    data_g = Variable(torch.from_numpy(data_g)).cuda()
    data_g.data.resize_(batch_size, 1, 28, 28)
    _,_,_,f3 = enet(data_g)
    # f3 = torch.mean(f3,1)
    # print(np.shape(owntool.extract(f3)))
    g_f3 = f3.cuda()
    g_f3.data.resize_(batch_size,240,1,1)
    g_f3 = g_f3.detach()

    g_sampler = Variable((torch.randn([batch_size,hidden_d,1,1])-1)*2.5).cuda()
    g_f3_output = G(torch.cat([g_f3,g_sampler],1))
    # G for fake data
    # c_v = Variable(model.set_label_f3(f3)).cuda()
    dg_decision = D(g_f3_output)
    dg_error = criterion(dg_decision,Variable(torch.ones([batch_size,1])).cuda())
    dg_error.backward()
    G_solver.step()

    D.zero_grad()
    G.zero_grad()
    E.zero_grad()

# Evaluator: Triplet Net
    # E part
    # For E, triplet part
    #     z = Variable(torch.randn(mb_size, Z_dim)).cuda()
    X, c = mm.batch_next(batch_size, shuffle=False)
    X = Variable(torch.from_numpy(X.astype('float32'))).cuda()
    # X.data.resize_(mb_size, 1, 32, 28)
    E_anch = E(X)

    X2, c2 = mm.batch_next(batch_size, shuffle=False)
    X2 = Variable(torch.from_numpy(X2.astype('float32'))).cuda()
    # X2.data.resize_(mb_size, 1, 28, 28)
    E_real = E(X2)

    new_label_base = np.array(range(10))
    new_label = (new_label_base + (np.random.rand(10) * 9 + 1).astype('int'))%10
    # random_label = model.set_label_ve_ma((c.astype('int') + (np.random.rand(batch_size) * 9 + 1).astype('int')) % 10)
    # random_label = Variable(torch.from_numpy(random_label.astype('float32'))).cuda()  # label for g c
    X3, c3 = mm.batch_next(batch_size, label=new_label, shuffle=False)
    X3 = Variable(torch.from_numpy(X3.astype('float32'))).cuda()
    # X3.data.resize_(mb_size, 1, 28, 28)
    E_fake = E(X3)

    E_loss = criterion_t(E_anch, E_real, E_fake)
    E_loss.backward()
    E_solver.step()

    # Housekeeping - reset gradient
    D.zero_grad()
    G.zero_grad()
    E.zero_grad()

    # For E ( generated data )
    _,_,_,f3 = enet(X)
    e_f3 = f3.cuda()
    e_f3.data.resize_(batch_size,240,1,1)
    e_f3 = e_f3.detach()
    g_sampler = Variable((torch.randn([batch_size,hidden_d,1,1])-1)*2.5).cuda()
    G_sample = G(torch.cat([e_f3,g_sampler],1))
    E_real = E(G_sample)

    # genereted (another kind of data)
    # data_ol == data_other_label
    new_label_base = np.array(range(10))
    new_label = (new_label_base + (np.random.rand(10) * 9 + 1).astype('int'))%10
    data_ol, label_ol = mm.batch_next(batch_size, label = new_label, shuffle=False)
    data_ol = Variable(torch.from_numpy(data_ol.astype('float32'))).cuda()
    data_ol.data.resize_(batch_size, 1, 28, 28)
    _,_,_,f3_fake = enet(data_ol)
    g_f3_fake = f3_fake.cuda()
    g_f3_fake.data.resize_(batch_size,240,1,1)
    g_f3_fake = g_f3_fake.detach()
    # g_sampler = Variable((torch.randn([batch_size,hidden_d,1,1])-1)*2.5).cuda()
    G_sample_fake = G(torch.cat([g_f3_fake,g_sampler],1))
    E_fake = E(G_sample_fake)
    E_anch = E(X)
    E_loss = criterion_t(E_anch, E_real, E_fake)
    E_loss.backward()
    # E_solver.step()
    G_solver.step()

    # Housekeeping - reset gradient
    D.zero_grad()
    G.zero_grad()
    E.zero_grad()

    # print("heelo")
    if(epoch%check_points==0):
        # observe the weight of two pictures
        zeroinput = Variable(torch.zeros([batch_size, hidden_d,1,1])).cuda()
        g_zero_output = G(torch.cat([g_f3, zeroinput], 1))

        g_sampler2 = Variable((torch.randn([batch_size,  hidden_d,1,1])-1)*2.5).cuda()
        # g_f3_output1 = G(torch.cat([g_f3, g_sampler2], 1))

        g_sampler3 = Variable(torch.randn([batch_size,hidden_d,1,1])).cuda()

        # zeroinput2 = Variable(torch.zeros([batch_size, 10, hidden_d, hidden_d])).cuda()
        # g_f3_output2 = G(torch.cat([zeroinput2, g_sampler3], 1))

        # owntool.save_picture(data, out_dir + "recon_epoch{}_data.png".format(epoch ),column=column)
        owntool.save_picture(data_g, out_dir + "recon_epoch{}_data_old.png".format(epoch ),column=column)
        owntool.save_picture(g_zero_output, out_dir + "recon_epoch{}_zero.png".format(epoch),column=column)
        owntool.save_picture(g_f3_output, out_dir + "recon_epoch{}_real.png".format(epoch),column=column)
        # owntool.save_picture(g_f3_output2, out_dir + "recon_f3_epoch{}_noise.png".format(epoch))

        # owntool.save_picture(data,out_dir + "/recon_o_epoch{}_{}.png".format(epoch,batch_index))
        # owntool.save_picture(g_f3_output,out_dir + "/recon_g_f3_epoch{}_{}.png".format(epoch,batch_index))

        print("%s: D: %s/%s G: %s (Real: %s, Fake: %s) E: %s"% (epoch,
                                                       owntool.extract(d_real_error)[0],
                                                         owntool.extract(d_false_error)[0],
                                                         owntool.extract(dg_error)[0],
                                                         owntool.stats(owntool.extract(d_real_decision)),
                                                         owntool.stats(owntool.extract(d_false_decision)),
                                                        owntool.stats(owntool.extract(E_loss)[0])
                                                         )
              )
    if(epoch%(check_points*10)==0):

        np.set_printoptions(precision=2,suppress=True)
        print("real")
        print( np.array(E_real.data.tolist()))
        print("anch")
        print( np.array(E_anch.data.tolist()))
        print("fake")
        print( np.array(E_fake.data.tolist()))
        torch.save(G.state_dict(), '{}/G_{}.model'.format(out_dir, str(epoch)))
        torch.save(D.state_dict(), '{}/D_{}.model'.format(out_dir, str(epoch)))