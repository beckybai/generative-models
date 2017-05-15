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
ngpu = 2
if __name__ == '__main__':
    hidden_d = 128 # the dimension of the feature map in the first layer

main_gpu = 2

column = 10
torch.cuda.set_device(main_gpu)

# mnist = input_data.read_data_sets('../../MNIST_data', one_hot=False)
mm = data_convert.cifar10()

seed_num = 1
torch.cuda.manual_seed(seed_num)
data_dir = '/home/bike/data/mnist/'
out_dir = '/home/bike/2027/generative-models/GAN/conditional_gan/cifar10_fc_triplet_net_{}/'.format(datetime.now())

if not os.path.exists(out_dir):
    os.makedirs(out_dir)
    shutil.copyfile(sys.argv[0], out_dir + '/cifar10_fc_triplet_net_first_try.py')

sys.stdout = mutil.Logger(out_dir)

batch_size = 100
feature_size = 64

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
encoder_model = torch.load("/home/bike/2027/cifar_pretrain/pytorch-cifar/cifar10_vgg_small_138.model")
# D_model = torch.load("/home/bike/2027/generative-models/GAN/conditional_gan/cifar10_fc_triplet_1w_bad/D_10000.model")
# G_model = torch.load("/home/bike/2027/generative-models/GAN/conditional_gan/cifar10_fc_triplet_1w_bad/G_10000.model")


# 24*24
enet = model.E_Net_VGG_cifar10_64().cuda()
enet = torch.nn.DataParallel(enet, device_ids=range(main_gpu, main_gpu+ngpu))
enet.load_state_dict(encoder_model)

add_in_feature = feature_size+ hidden_d # Add one dimension data for the input_feature data.
G = model.G_Net_FM_Cifar(ngpu,add_in_feature,main_gpu=main_gpu).cuda()
print(G)
# G.load_state_dict(G_model)
in_channel = 3
E = model.D_Net_Cifar10(ngpu,in_channel,output_features = 100,main_gpu=main_gpu).cuda()
print(E)
# E.apply(weights_init)
# G.apply(weights_init)

d_in_demension = 3
D = model.D_Net_Cifar10(ngpu,d_in_demension,main_gpu=main_gpu).cuda()
# D.load_state_dict(D_model)

print(D)
# d_model = torch.load("/home/bike/2027/generative-models/GAN/conditional_gan/mnist_pretrain_DG/D_60000.model")
# D.load_state_dict(d_model)

G_solver = optim.Adam(G.parameters(),lr = 1e-4)
D_solver = optim.Adam(D.parameters(), lr = 1e-4)
E_solver = optim.Adam(E.parameters(), lr = 1e-4)

half_label = Variable(torch.ones(batch_size)*0.5).cuda()

check_points = 250
num_epoches = 100000
criterion = nn.BCELoss()
criterion_t = nn.TripletMarginLoss(p=1)
criterion_mse = nn.MSELoss()

for epoch in (range(0,num_epoches)):
# D PART ./ data_old/ d_fc / error
    data, label = mm.batch_next(batch_size, shuffle=True)
    D.zero_grad()
    G.zero_grad()
    data_old = Variable(torch.from_numpy(data)).cuda()
    data_old.data.resize_(batch_size, 3, 32, 32)
    _,fc_feature= enet(data_old.detach())
    d_fc = fc_feature.cuda()
    g_sampler = Variable((torch.randn([batch_size,hidden_d,1,1]))).cuda()
    d_fc.data.resize_(batch_size,feature_size,1,1)
    d_fc = d_fc.detach()
    d_fc_output = G(torch.cat([d_fc,g_sampler],1)).detach()

    d_real_decision = D(data_old)

    d_false_decision = D(d_fc_output)
    d_false_error = criterion(d_false_decision,Variable(torch.zeros(batch_size)).cuda())


    d_real_error = criterion(d_real_decision,Variable(torch.ones(batch_size)).cuda())

    # d_real_false_error = criterion(d_real_decision/2 + d_false_decision/2, half_label)
    d_real_error.backward()
    d_false_error.backward()
    # error = d_real_error + d_false_errord_false_error
    # error.backward()
    D_solver.step()

# G Part / data_g / g_fc / dg_error
    D.zero_grad()
    G.zero_grad()
    data_g, label_g = mm.batch_next(batch_size, shuffle=True)
    data_g = Variable(torch.from_numpy(data_g)).cuda()
    # data_g.data.resize_(batch_size, 1, 28, 28)
    _,fc_feature= enet(data_g)
    g_fc = fc_feature.cuda()
    g_fc.data.resize_(batch_size,feature_size,1,1)
    g_fc = g_fc.detach()

    g_sampler = Variable((torch.randn([batch_size,hidden_d,1,1]))).cuda()
    g_fc_output = G(torch.cat([g_fc,g_sampler],1))
    # G for fake data
    # c_v = Variable(model.set_label_fc(fc)).cuda()
    dg_decision = D(g_fc_output)
    dg_error = criterion(dg_decision,Variable(torch.ones([batch_size,1])).cuda())
    dg_error.backward()
    G_solver.step()
# Evaluator: Triplet Net
    # E part
    # For E, triplet part
    D.zero_grad()
    G.zero_grad()
    E.zero_grad()

    X, c = mm.batch_next(batch_size, shuffle=False)
    X = Variable(torch.from_numpy(X.astype('float32'))).cuda()
    E_anch = E(X)

    X2, c2 = mm.batch_next(batch_size, shuffle=False)
    X2 = Variable(torch.from_numpy(X2.astype('float32'))).cuda()
    E_real = E(X2)

    new_label_base = np.array(range(10))
    new_label = (new_label_base + (np.random.rand(10) * 9 + 1).astype('int'))%10
    X3, c3 = mm.batch_next(batch_size, label=new_label, shuffle=False)
    X3 = Variable(torch.from_numpy(X3.astype('float32'))).cuda()
    E_fake = E(X3)

    E_loss = criterion_t(E_anch, E_real, E_fake)
    E_loss.backward()
    E_solver.step()

# Evaluator: G
    D.zero_grad()
    E.zero_grad()

    # For E ( generated data )
    _,fc_feature= enet(X)
    e_fc = fc_feature.cuda()
    e_fc.data.resize_(batch_size,feature_size,1,1)
    e_fc = e_fc.detach()
    # e_fc = F.sigmoid(e_fc)
    g_sampler = Variable((torch.randn([batch_size,hidden_d,1,1]))).cuda()
    G_sample = G(torch.cat([e_fc,g_sampler],1))
    E_real = E(G_sample)
    E_anch = E(X)
    # genereted (another kind of data)
    # data_ol == data_other_label
    new_label_base = np.array(range(10))
    new_label = (new_label_base + (np.random.rand(10) * 9 + 1).astype('int'))%10
    data_ol, label_ol = mm.batch_next(batch_size, label = new_label, shuffle=False)
    data_ol = Variable(torch.from_numpy(data_ol.astype('float32'))).cuda()
    # data_ol.data.resize_(batch_size, 1, 28, 28)
    _,fc_feature = enet(data_ol)
    g_fc_fake = fc_feature.cuda()
    # g_fc_fake = F.sigmoid(g_fc_fake)
    g_fc_fake.data.resize_(batch_size,feature_size,1,1)
    g_fc_fake = g_fc_fake.detach()
    g_sampler = Variable(torch.randn([batch_size,hidden_d,1,1])).cuda()
    G_sample_fake = G(torch.cat([g_fc_fake,g_sampler],1))
    E_fake = E(G_sample_fake)
    # E_anch = E(X)
    E_loss = criterion_t(E_anch, E_real, E_fake)
    E_loss.backward()
    G_solver.step()

    # Housekeeping - reset gradient
    # print("heelo")
    if(epoch%check_points==0):
        # observe the weight of two pictures
        if (epoch % check_points == 0):
            # observe the weight of two pictures
            zeroinput = Variable(torch.zeros([batch_size, hidden_d, 1, 1])).cuda()
            g_zero_output = G(torch.cat([e_fc, zeroinput], 1))

            g_sampler2 = Variable((torch.rand([batch_size, hidden_d, 1, 1]))).cuda()
            g_fc_output1 = G(torch.cat([e_fc, g_sampler2], 1))

            # owntool.save_picture(data, out_dir + "recon_epoch{}_data.png".format(epoch ),column=column)
            owntool.save_color_picture_pixel(X.data.tolist(), out_dir + "recon_epoch{}_data_old.png".format(epoch),
                                             column=column, image_size=32)
            owntool.save_color_picture_pixel(g_zero_output.data.tolist(),
                                             out_dir + "recon_epoch{}_zero.png".format(epoch), column=column,
                                             image_size=32)
            owntool.save_color_picture_pixel(g_fc_output1.data.tolist(),
                                             out_dir + "recon_epoch{}_real.png".format(epoch), column=column,
                                             image_size=32)

            print("%s: D: %s/%s G: %s (Real: %s, Fake: %s) E: %s" % (epoch,
                                                                     owntool.extract(d_real_error)[0],
                                                                     owntool.extract(d_false_error)[0],
                                                                     owntool.extract(dg_error)[0],
                                                                     owntool.stats(owntool.extract(d_real_decision)),
                                                                     owntool.stats(owntool.extract(d_false_decision)),
                                                                     owntool.stats(owntool.extract(E_loss)[0])
                                                                     )
                  )
        if (epoch % (check_points * 10) == 0):
            np.set_printoptions(precision=2, suppress=True)
            print("real")
            # num = list(range(0,100,5))
            print(np.array(E_real.data.tolist()[0:100:5]))
            print("anch")
            print(np.array(E_anch.data.tolist()[0:100:5]))
            print("fake")
            print(np.array(E_fake.data.tolist()[0:100:5]))
            torch.save(G.state_dict(), '{}/G_{}.model'.format(out_dir, str(epoch)))
            torch.save(D.state_dict(), '{}/D_{}.model'.format(out_dir, str(epoch)))
            torch.save(E.state_dict(), '{}/E_{}.model'.format(out_dir, str(epoch)))