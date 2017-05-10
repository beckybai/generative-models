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

main_gpu = 2
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
feature_size = 128

kwargs = {'num_workers': 1, 'pin_memory': True}
train_loader =  torch.utils.data.DataLoader(
    datasets.MNIST(data_dir, train = True,download=False,
                   transform= transforms.Compose([transforms.ToTensor(),transforms.Normalize((0,),(1,))])),
    batch_size = batch_size, shuffle=True, **kwargs)

# layer_names = list(model.keys())

# write the origin model again.

# D0 = model.E_Net_small()
encoder_model = torch.load("/home/bike/2027/generative-models/tools/D_mnist_little.model")

# 24*24
enet = model.E_Net_small().cuda()
enet.load_state_dict(encoder_model)

add_in_feature = feature_size+ hidden_d # Add one dimension data for the input_feature data.
G = model.G_Net_FM_3(ngpu,add_in_feature,main_gpu=main_gpu).cuda()
# g_model = torch.load("./fm21/G_95000.model")
# G.load_state_dict(g_model)

in_channel = 1
E = model.Ev_Net_conv(ngpu,in_channel,main_gpu=main_gpu).cuda()

d_in_demension = 1
D = model.D_Net_conv(ngpu,d_in_demension,main_gpu=main_gpu).cuda()
G_solver = optim.Adam(G.parameters(),lr = 1e-5)
D_solver = optim.Adam(D.parameters(), lr = 1e-4)
E_solver = optim.Adam(E.parameters(), lr=1e-4)

half_label = Variable(torch.ones(batch_size)*0.5).cuda()

check_points = 500
num_epoches = 100000
criterion = nn.BCELoss()
criterion_t = nn.TripletMarginLoss(p=1)
criterion_mse = nn.MSELoss()

for epoch in (range(1,num_epoches)):


# D PART ./ data_old/ d_f3 / error
    data, label = mm.batch_next(batch_size, label = [3,4,5,6,7,8,9,0,1,2],shuffle=False)
    D.zero_grad()
    G.zero_grad()
    data_old = Variable(torch.from_numpy(data.astype('float32'))).cuda()
    data_old.data.resize_(batch_size, 1, 28, 28)
    _,_,_,f3 = enet(data_old.detach())


    d_f3 = f3.cuda()
    g_sampler = Variable((torch.randn([batch_size,hidden_d,1,1]))).cuda()
    d_f3.data.resize_(batch_size,feature_size,1,1)
    d_f3 = d_f3.detach()