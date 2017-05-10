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
import data_convert

import os
import sys,shutil

ngpu = 1
if __name__ == '__main__':
    hidden_d = 24 # the dimension of the feature map in the first layer
main_gpu = 2
torch.cuda.set_device(main_gpu)


seed_num = 1
torch.cuda.manual_seed(seed_num)
data_dir = '/home/bike/data/mnist/'
out_dir = '/home/bike/2027/generative-models/GAN/conditional_gan/fm_out_{}/'.format(datetime.now())

if not os.path.exists(out_dir):
    os.makedirs(out_dir)
    shutil.copyfile(sys.argv[0], out_dir + '/training_script.py')

sys.stdout = mutil.Logger(out_dir)

batch_size = 50

kwargs = {'num_workers': 1, 'pin_memory': True}
train_loader =  torch.utils.data.DataLoader(
    datasets.MNIST(data_dir, train = True,download=False,
                   transform= transforms.Compose([transforms.ToTensor(),transforms.Normalize((0,),(1,))])),
    batch_size = batch_size, shuffle=True, **kwargs)

mm = data_convert.owndata()
# layer_names = list(model.keys())

# write the origin model again.

# 24*24
encoder_model = torch.load("./Dmodel")
enet = model.E_Net_2().cuda()
enet.load_state_dict(encoder_model)
feature_d = 2
add_in_feature = feature_d + 1 # Add one dimension data for the input_feature data.
gnet = model.G_Net_FM(ngpu,add_in_feature,main_gpu).cuda()
d_in_demension = 1
dnet = model.D_Net_conv(ngpu,d_in_demension,main_gpu).cuda()

g_net_optimizer = optim.Adam(gnet.parameters(),lr = 1e-4)
d_net_optimizer = optim.Adam(dnet.parameters(), lr = 1e-5)

num_epoches = 30000
check_points = 500
criterion = nn.BCELoss()
noise_d = 1


for epoch in (range(1,num_epoches)):
# print("give me a clue")
    data, label = mm.batch_next(batch_size,[0,1,2,3,4,5,6,7,8,9],shuffle=True)
    # dnet.zero_grad()
    dnet.zero_grad()
    gnet.zero_grad()
    data = torch.from_numpy(data).cuda()# pick data
    data_old = Variable(data)
    _,f1,_ = enet(data_old.detach()) # pick feature map
    g_f1 = model.get_feature(f1,feature_d, batch_size)
    g_sampler = Variable((torch.randn([batch_size,noise_d,hidden_d,hidden_d])-0.2)*0.5).cuda()
    g_f1_output = gnet(torch.cat([g_f1,g_sampler],1)).detach() # generate data
    # c_v = model.set_condition(g_f1.data, 1,batch_size=batch_size)
    c_v = Variable(torch.from_numpy(model.set_label_ve(label).astype('int'))).cuda()  # d_false_decision = dnet(torch.cat([g_f1_output, c_v], 1))# get false decision
    d_false_decision = dnet(torch.cat([g_f1_output,c_v],1))# get false decision

    d_false_error = criterion(d_false_decision, Variable(torch.zeros(batch_size)).cuda())

    data, label = mm.batch_next(batch_size, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], shuffle=True)
    data = Variable(torch.from_numpy(data).cuda())
    _, f1, _ = enet(data.detach())
    g_f1 = model.get_feature(f1,feature_d,batch_size)
    # c_v = model.set_condition(g_f1.data, 1, batch_size = batch_size)
    c_v = Variable(torch.from_numpy(model.set_label_ve(label).astype('float32'))).cuda()  # d_false_decision = dnet(torch.cat([g_f1_output, c_v], 1))# get false decision  # d_real_decision = dnet(torch.cat([data, c_v], 1))# get right decision
    d_real_decision = dnet(torch.cat([data,c_v],1))# get right decision
    d_real_error = criterion(d_real_decision,Variable(torch.ones(batch_size)).cuda())
    # D for fake data
    error = d_real_error + d_false_error
    error.backward()
    d_net_optimizer.step()

    # G Part
    dnet.zero_grad()
    gnet.zero_grad()
    # gnet_noise.zero_grad()
    data, label = mm.batch_next(batch_size,[0,1,2,3,4,5,6,7,8,9],shuffle=True)
    data = torch.from_numpy(data).cuda()
    data = Variable(data)
    _, f1, _ = enet(data.detach())
    g_f1 = model.get_feature(f1,feature_d,batch_size = batch_size)
    g_sampler = Variable((torch.randn([batch_size,noise_d,hidden_d,hidden_d])-0.2)*0.5).cuda()
    g_f1_output = gnet(torch.cat([g_f1,g_sampler],1))
    # c_v = model.set_condition(g_f1.data,1, batch_size=batch_size)
    c_v = Variable(torch.from_numpy(model.set_label_ve(label).astype('float32'))).cuda()  # d_false_decision = dnet(torch.cat([g_f1_output, c_v], 1))# get false decision  # d_real_decision = dnet(torch.cat([data, c_v], 1))# get right decision

    dg_decision = dnet(torch.cat([g_f1_output,c_v],1))#
    dg_error = criterion(dg_decision,Variable(torch.ones(batch_size)).cuda())
    dg_error.backward()
    g_net_optimizer.step()

    dnet.zero_grad()
    gnet.zero_grad()
    # print("heelo")
    if(epoch%check_points==0):
        # observe the weight of two pictures
        # save the generated samples to the folder

        zeroinput = Variable(torch.zeros([batch_size, noise_d, hidden_d, hidden_d])).cuda()
        g_zero_output = gnet(torch.cat([g_f1, zeroinput], 1))

        g_sampler2 = Variable((torch.randn([batch_size, noise_d, hidden_d, hidden_d]) - 0.2) * 0.5).cuda()
        g_f1_output1 = gnet(torch.cat([g_f1, g_sampler2], 1))

        g_sampler3 = Variable((torch.randn([batch_size, feature_d, hidden_d, hidden_d]) - 0.2) * 0.5).cuda()

        zeroinput2 = Variable(torch.zeros([batch_size, noise_d, hidden_d, hidden_d])).cuda()
        g_f1_output2 = gnet(torch.cat([zeroinput2, g_sampler3], 1))

        owntool.save_picture(data, out_dir + "recon_epoch{}_origin.png".format(epoch))
        owntool.save_picture(g_zero_output, out_dir + "recon_epoch{}_zero.png".format(epoch))
        owntool.save_picture(g_f1_output1, out_dir + "recon_epoch{}_real.png".format(epoch))
        owntool.save_picture(g_f1_output2, out_dir + "recon_f1_epoch{}_noise.png".format(epoch))

        # owntool.save_picture(data,out_dir + "/recon_o_epoch{}_{}.png".format(epoch,batch_index))
        # owntool.save_picture(g_f1_output,out_dir + "/recon_g_f1_epoch{}_{}.png".format(epoch,batch_index))

        print("%s: D: %s/%s G: %s (Real: %s, Fake: %s)"% (batch_size,
                                                       owntool.extract(d_real_error)[0],
                                                         owntool.extract(d_false_error)[0],
                                                         owntool.extract(dg_error)[0],
                                                         owntool.stats(owntool.extract(d_real_decision)),
                                                         owntool.stats(owntool.extract(d_false_decision))
                                                         )
              )
        torch.save(gnet.state_dict(), '{}/G_{}.model'.format(out_dir, str(epoch)))
        torch.save(dnet.state_dict(), '{}/D_{}.model'.format(out_dir, str(epoch)))