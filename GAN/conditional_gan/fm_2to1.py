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

ngpu = 2
if __name__ == '__main__':
    f1_dimension = 24# the dimension of the feature map in the first layer
    f2_dimension = 8
    
main_gpu = 0
torch.cuda.set_device(main_gpu)

mnist = input_data.read_data_sets('../../MNIST_data', one_hot=False)
mm = data_convert.owndata()

seed_num = 1
torch.cuda.manual_seed(seed_num)
data_dir = '/home/bike/data/mnist/'
out_dir = '/home/bike/2027/generative-models/GAN/conditional_gan/fm21_out_{}/'.format(datetime.now())

if not os.path.exists(out_dir):
    os.makedirs(out_dir)
    shutil.copyfile(sys.argv[0], out_dir + '/fm_2to1.py')

sys.stdout = mutil.Logger(out_dir)

batch_size = 50
ms = f1_dimension # 28 for mnist data

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

add_in_feature = 51 # Add one dimension data for the input_feature data.
output_feature = 10
gnet = model.G_Net_FM_21(ngpu,add_in_feature, output_feature,main_gpu).cuda()
gnet1p = model.G_Net_FM(ngpu, 1,main_gpu).cuda()
genrator_model = torch.load("./bksb/G_4.model")
gnet1p.load_state_dict(genrator_model)

d_in_demension = 11
dnet = model.D_Net_conv_24(ngpu,d_in_demension,main_gpu).cuda()

g_net_optimizer = optim.Adam(gnet.parameters(),lr = 1e-3)
d_net_optimizer = optim.Adam(dnet.parameters(), lr = 1e-4)


check_points = 500
num_epoches = 100000
criterion = nn.BCELoss()

for epoch in (range(1,num_epoches)):
    # print("give me a clue")
    data, label = mm.bat
    dnet.zero_grad()
    gnet.zero_grad()

    data_old = Variable(torch.from_numpy(data)).cuda()
    data_old.data.resize_(batch_size, 1, 28, 28)

    # data, label = Variable(data), Variable(label)
    c_v = Variable(torch.from_numpy(model.set_label_ve(label,ms).astype('float32'))).cuda()

    _,f1,f2 = enet(data_old.detach())
    # print(np.shape(owntool.extract(f1)))
    g_f1 = f1.cuda()
    g_f2 = f2.cuda()

    # D Part
    g_sampler = Variable(torch.randn([batch_size,1,f2_dimension,f2_dimension])).cuda()
    g_f2_output = gnet(torch.cat([g_f2.detach(),g_sampler],1)).detach()
    d_false_decision = dnet(torch.cat([g_f2_output,c_v],1))


    # print(g_sampler.data.tolist()[0][0][0][0])
    d_real_decision = dnet(torch.cat([g_f1,c_v],1))

    d_real_error = criterion(d_real_decision,Variable(torch.ones(batch_size)).cuda())
    d_false_error = criterion(d_false_decision,Variable(torch.zeros(batch_size)).cuda())

    error = d_real_error + d_false_error
    error.backward()
    d_net_optimizer.step()

    # G Part
    dnet.zero_grad()
    # gnet.zero_grad()
    # gnet_noise.zero_grad()

    g_sampler = Variable(torch.randn([batch_size,1, f2_dimension,f2_dimension])).cuda()
    g_f2_output = gnet(torch.cat([g_f2.detach(),g_sampler],1))
    dg_decision = dnet(torch.cat([g_f2_output,c_v],1))
    dg_error = criterion(dg_decision,Variable(torch.ones(batch_size)).cuda())
    dg_error.backward()
    g_net_optimizer.step()

    dnet.zero_grad()
    gnet.zero_grad()

    # print("heelo")
    if(epoch%check_points==0):
        # observe the weight of two pictures

        # save the generated samples to the folder
        noise_f1 =   Variable(torch.randn([batch_size,1, f1_dimension,f1_dimension])).cuda()

        zeroinput = Variable(torch.zeros([batch_size, 1, f2_dimension, f2_dimension])).cuda()
        g_zero_output = gnet(torch.cat([g_f2, zeroinput], 1))

        g_sampler2 = Variable(torch.randn([batch_size, 1, f2_dimension, f2_dimension])).cuda()
        g_f1_output1 = gnet(torch.cat([g_f2, g_sampler2], 1))

        g_sampler3 = Variable(torch.randn([batch_size, 1, f2_dimension, f2_dimension])).cuda()

        zeroinput2 = Variable(torch.zeros([batch_size, 50, f2_dimension, f2_dimension])).cuda()
        g_f1_output2 = gnet(torch.cat([zeroinput2, g_sampler3], 1))

        owntool.save_picture(data, out_dir + "recon_epoch{}_origin.png".format(epoch ))
        owntool.save_picture(gnet1p(torch.cat([g_zero_output,noise_f1],1)), out_dir + "recon_epoch{}_zero.png".format(epoch))
        owntool.save_picture(gnet1p(torch.cat([g_f1_output1, noise_f1],1)), out_dir + "recon_epoch{}_real.png".format(epoch))
        owntool.save_picture(gnet1p(torch.cat([g_f1_output2,noise_f1],1)), out_dir + "recon_f1_epoch{}_noise.png".format(epoch))

        # owntool.save_picture(data,out_dir + "/recon_o_epoch{}_{}.png".format(epoch,batch_index))
        # owntool.save_picture(g_f1_output,out_dir + "/recon_g_f1_epoch{}_{}.png".format(epoch,batch_index))

        print("%s: D: %s/%s G: %s (Real: %s, Fake: %s)"% (epoch,
                                                       owntool.extract(d_real_error)[0],
                                                         owntool.extract(d_false_error)[0],
                                                         owntool.extract(dg_error)[0],
                                                         owntool.stats(owntool.extract(d_real_decision)),
                                                         owntool.stats(owntool.extract(d_false_decision))
                                                         )
              )
    if(epoch%(check_points*10)==0):
        torch.save(gnet.state_dict(), '{}/G_{}.model'.format(out_dir, str(epoch)))
        torch.save(dnet.state_dict(), '{}/D_{}.model'.format(out_dir, str(epoch)))