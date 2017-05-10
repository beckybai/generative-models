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


import os
import sys,shutil

ngpu = 1
if __name__ == '__main__':
    hidden_d = 8 # the dimension of the feature map in the first layer

torch.cuda.set_device(5)


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

# layer_names = list(model.keys())

# write the origin model again.

D0 = model.E_Net_2()
encoder_model = torch.load("./Dmodel")

# 24*24


enet = model.E_Net_2().cuda()
enet.load_state_dict(encoder_model)

add_in_feature = 1 # Add one dimension data for the input_feature data.
gnet = model.G_Net_FM_2(ngpu,add_in_feature).cuda()
d_in_demension = 1
dnet = model.D_Net_conv(ngpu,d_in_demension).cuda()

g_net_optimizer = optim.Adam(gnet.parameters(),lr = 1e-4)
d_net_optimizer = optim.Adam(dnet.parameters(), lr = 1e-4)

num_epoches = 30
check_points = 20000
criterion = nn.BCELoss()

for epoch in (range(1,num_epoches)):
    # print("give me a clue")
    for batch_index, (data, label) in enumerate(train_loader):
        dnet.zero_grad()
        data, label = data.cuda(), label.cuda()
        data, label = Variable(data), Variable(label)
        _,f1 = enet(data.detach())
        # print(np.shape(owntool.extract(f1)))
        g_f1 = f1.cuda()

        # D Part
        g_sampler = Variable(torch.randn([batch_size,1,hidden_d,hidden_d])).cuda()
        # print(g_sampler.data.tolist()[0][0][0][0])
        g_f1_output = gnet(torch.cat([g_f1,g_sampler],1)).detach()

        d_real_decision = dnet(data)
        d_real_error = criterion(d_real_decision,Variable(torch.ones(batch_size)).cuda())
        # D for fake data
        d_false_decision = dnet(g_f1_output.detach())
        d_false_error = criterion(d_false_decision,Variable(torch.zeros(batch_size)).cuda())
        error = d_real_error + d_false_error
        error.backward()
        d_net_optimizer.step()

        # G Part
        gnet.zero_grad()
        # gnet_noise.zero_grad()

        g_sampler = Variable(torch.randn([batch_size,1, hidden_d,hidden_d])).cuda()
        # print(g_sampler.data.tolist()[0][0][0][0])
        g_f1_output = gnet(torch.cat([g_f1,g_sampler],1))

        # G for fake data
        dg_decision = dnet(g_f1_output)
        dg_error = criterion(dg_decision,Variable(torch.ones(batch_size)).cuda())
        dg_error.backward()
        g_net_optimizer.step()

        # print("heelo")
        if(batch_index*batch_size%check_points==0):
            # observe the weight of two pictures

            # save the generated samples to the folder

            zeroinput = Variable(torch.zeros([batch_size, 1, hidden_d, hidden_d])).cuda()
            g_zero_output = gnet(torch.cat([g_f1, zeroinput], 1))

            g_sampler2 = Variable(torch.randn([batch_size, 1, hidden_d, hidden_d])).cuda()
            g_f1_output1 = gnet(torch.cat([g_f1, g_sampler2], 1))

            g_sampler3 = Variable(torch.randn([batch_size, 1, hidden_d, hidden_d])).cuda()

            zeroinput2 = Variable(torch.zeros([batch_size, 50, hidden_d, hidden_d])).cuda()
            g_f1_output2 = gnet(torch.cat([zeroinput2, g_sampler3], 1))

            owntool.save_picture(data, out_dir + "recon_epoch{}_{}_origin.png".format(epoch, batch_index))
            owntool.save_picture(g_zero_output, out_dir + "recon_epoch{}_{}_zero.png".format(epoch, batch_index))
            owntool.save_picture(g_f1_output1, out_dir + "recon_epoch{}_{}_real.png".format(epoch, batch_index))
            owntool.save_picture(g_f1_output2, out_dir + "recon_f1_epoch{}_{}_noise.png".format(epoch, batch_index))

            # owntool.save_picture(data,out_dir + "/recon_o_epoch{}_{}.png".format(epoch,batch_index))
            # owntool.save_picture(g_f1_output,out_dir + "/recon_g_f1_epoch{}_{}.png".format(epoch,batch_index))

            print("%s: D: %s/%s G: %s (Real: %s, Fake: %s)"% (batch_index*batch_size,
                                                           owntool.extract(d_real_error)[0],
                                                             owntool.extract(d_false_error)[0],
                                                             owntool.extract(dg_error)[0],
                                                             owntool.stats(owntool.extract(d_real_decision)),
                                                             owntool.stats(owntool.extract(d_false_decision))
                                                             )
                  )
    if(epoch%2==0):
        torch.save(gnet.state_dict(), '{}/G_{}.model'.format(out_dir, str(epoch)))
        torch.save(dnet.state_dict(), '{}/D_{}.model'.format(out_dir, str(epoch)))