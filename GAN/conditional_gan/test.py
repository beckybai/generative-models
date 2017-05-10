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
    hidden_d = 24 # the dimension of the feature map in the first layer

torch.cuda.set_device(4)


seed_num = 1
torch.cuda.manual_seed(seed_num)
data_dir = '/home/bike/data/mnist/'
out_dir = '/home/bike/2027/generative-models/GAN/conditional_gan/testf/fm_out_{}/'.format(datetime.now())

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

D0 = model.E_Net()
encoder_model = torch.load("./Dmodel")
D0.load_state_dict(encoder_model)

# 24*24


enet = model.E_Net().cuda()
encoder_model = torch.load("./Dmodel")
enet.load_state_dict(encoder_model)
add_in_feature = 1 # Add one dimension data for the input_feature data.


gnet = model.G_Net_FM(ngpu,add_in_feature).cuda()
gnet_model = torch.load("./bksb/G_2.model")
gnet.load_state_dict(gnet_model)

d_in_demension = 1
# dnet = model.D_Net_conv(ngpu,d_in_demension).cuda()

g_net_optimizer = optim.Adam(gnet.parameters(),lr = 1e-4)
# d_net_optimizer = optim.Adam(dnet.parameters(), lr = 1e-5)

num_epoches = 10
check_points = 5000
criterion = nn.BCELoss()


for batch_index, (data, label) in enumerate(train_loader):
    # dnet.zero_grad()
    data, label = data.cuda(), label.cuda()
    data, label = Variable(data), Variable(label)
    _,f1 = enet(data.detach())
    # print(np.shape(owntool.extract(f1)))
    g_f1 = f1.cuda()

    # D Part
    # g_sampler = Variable(torch.randn([batch_size,1,hidden_d,hidden_d])).cuda()

    zeroinput = Variable(torch.zeros([batch_size,1,hidden_d,hidden_d])).cuda()
    g_f1_output = gnet(torch.cat([g_f1,zeroinput],1))

    g_sampler2 = Variable(torch.randn([batch_size,1,hidden_d,hidden_d])).cuda()
    g_f1_output1 = gnet(torch.cat([g_f1,g_sampler2],1))

    g_sampler3 = Variable(torch.randn([batch_size,1,hidden_d,hidden_d])).cuda()

    zeroinput2 = Variable(torch.zeros([batch_size,10,hidden_d,hidden_d])).cuda()
    g_f1_output2 = gnet(torch.cat([zeroinput2,g_sampler3],1))

    owntool.save_picture(data,out_dir + "recon_epoch{}_{}_origin.png".format(0,batch_index))
    owntool.save_picture(g_f1_output,out_dir + "recon_epoch{}_{}_zero.png".format(0,batch_index))
    owntool.save_picture(g_f1_output1,out_dir + "recon_epoch{}_{}_real.png".format(0,batch_index))
    owntool.save_picture(g_f1_output2,out_dir + "recon_f1_epoch{}_{}_noise.png".format(0,batch_index))


