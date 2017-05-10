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
    hidden_d = 16 # the dimension of the feature map in the first layer

main_gpu = 4
torch.cuda.set_device(main_gpu)

mnist = input_data.read_data_sets('../../MNIST_data', one_hot=False)
mm = data_convert.owndata()

seed_num = 1
torch.cuda.manual_seed(seed_num)
data_dir = '/home/bike/data/mnist/'
out_dir = '/home/bike/2027/generative-models/GAN/conditional_gan/fm_out_{}/'.format(datetime.now())

if not os.path.exists(out_dir):
    os.makedirs(out_dir)
    shutil.copyfile(sys.argv[0], out_dir + '/fm_conv_overlay_noise_label_feature2_is_not_real_pic.py')

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
gnet = model.G_Net_FM_3(ngpu,add_in_feature,main_gpu=main_gpu).cuda()
# g_model = torch.load("./fm21/G_95000.model")
# gnet.load_state_dict(g_model)
d_in_demension = 2
dnet = model.D_Net_conv(ngpu,d_in_demension,main_gpu=main_gpu).cuda()
# nosie_d = 10

g_net_optimizer = optim.Adam(gnet.parameters(),lr = 1e-4)
d_net_optimizer = optim.Adam(dnet.parameters(), lr = 1e-4)


check_points = 500
num_epoches = 100000
criterion = nn.BCELoss()

for epoch in (range(1,num_epoches)):
    # print("give me a clue")
    # data, label = mnist.train.next_batch(batch_size)
    data, label = mm.batch_next(batch_size, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], shuffle=True)
    dnet.zero_grad()
    gnet.zero_grad()

    data_old = Variable(torch.from_numpy(data)).cuda()
    data_old.data.resize_(batch_size, 1, 28, 28)

    # data, label = Variable(data), Variable(label)
    c_v = Variable(torch.from_numpy(model.set_label_ve(label).astype('float32'))).cuda()

    _,_,_,f3 = enet(data_old.detach())
    g_f3 = f3.cuda()
    g_sampler = Variable((torch.randn([batch_size,hidden_d,1,1])-1)*2.5).cuda()
    g_f3.data.resize_(batch_size,240,1,1)
    g_f3_output = gnet(torch.cat([g_f3,g_sampler],1))
    d_false_decision = dnet(torch.cat([g_f3_output.detach(),c_v],1))
    d_false_error = criterion(d_false_decision,Variable(torch.zeros(batch_size)).cuda())

    data, label = mm.batch_next(batch_size, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], shuffle=True)
    data = Variable(torch.from_numpy(data)).cuda()
    data.data.resize_(batch_size, 1, 28, 28)
    c_v = Variable(torch.from_numpy(model.set_label_ve(label).astype('float32'))).cuda()
    d_real_decision = dnet(torch.cat([data,c_v],1))
    d_real_error = criterion(d_real_decision,Variable(torch.ones(batch_size)).cuda())
    error = d_real_error + d_false_error
    error.backward()
    d_net_optimizer.step()

    # G Part
    dnet.zero_grad()
    gnet.zero_grad()
    data, label = mm.batch_next(batch_size, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], shuffle=True)
    data = Variable(torch.from_numpy(data)).cuda()
    data.data.resize_(batch_size, 1, 28, 28)
    _,_,_,f3 = enet(data)
    # f3 = torch.mean(f3,1)
    # print(np.shape(owntool.extract(f3)))
    g_f3 = f3.cuda()
    g_f3.data.resize_(batch_size,240,1,1)
    g_f3 = g_f3.detach()

    g_sampler = Variable((torch.randn([batch_size,hidden_d,1,1])-1)*2.5).cuda()
    g_f3_output = gnet(torch.cat([g_f3,g_sampler],1))
    # G for fake data
    c_v = Variable(torch.from_numpy(model.set_label_ve(label).astype('float32'))).cuda()
    dg_decision = dnet(torch.cat([g_f3_output,c_v],1))
    dg_error = criterion(dg_decision,Variable(torch.ones([batch_size,1])).cuda())
    dg_error.backward()
    g_net_optimizer.step()

    dnet.zero_grad()
    gnet.zero_grad()

    # print("heelo")
    if(epoch%check_points==0):
        # observe the weight of two pictures

        # save the generated samples to the folder

        zeroinput = Variable(torch.zeros([batch_size, hidden_d,1,1])).cuda()
        g_zero_output = gnet(torch.cat([g_f3, zeroinput], 1))

        g_sampler2 = Variable((torch.randn([batch_size,  hidden_d,1,1])-1)*2.5).cuda()
        g_f3_output1 = gnet(torch.cat([g_f3, g_sampler2], 1))

        g_sampler3 = Variable(torch.randn([batch_size,hidden_d,1,1])).cuda()

        # zeroinput2 = Variable(torch.zeros([batch_size, 10, hidden_d, hidden_d])).cuda()
        # g_f3_output2 = gnet(torch.cat([zeroinput2, g_sampler3], 1))

        owntool.save_picture(data, out_dir + "recon_epoch{}_data.png".format(epoch ))
        owntool.save_picture(data_old, out_dir + "recon_epoch{}_data_old.png".format(epoch ))
        owntool.save_picture(g_zero_output, out_dir + "recon_epoch{}_zero.png".format(epoch))
        owntool.save_picture(g_f3_output1, out_dir + "recon_epoch{}_real.png".format(epoch))
        # owntool.save_picture(g_f3_output2, out_dir + "recon_f3_epoch{}_noise.png".format(epoch))
        # owntool.save_picture(data,out_dir + "/recon_o_epoch{}_{}.png".format(epoch,batch_index))
        # owntool.save_picture(g_f3_output,out_dir + "/recon_g_f3_epoch{}_{}.png".format(epoch,batch_index))

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