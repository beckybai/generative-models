from tempfile import TemporaryFile

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
import scipy
import scipy.misc
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import os
import sys,shutil

# chagne today : 2017/5/9
ngpu = 1
if __name__ == '__main__':
    hidden_d = 128 # the dimension of the feature map in the first layer

main_gpu = 3

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
g_model = torch.load("/home/bike/2027/generative-models/GAN/conditional_gan/mnist_pretrain_G/G_80000.model")
G.load_state_dict(g_model)

in_channel = 1
E = model.Ev_Net_conv(ngpu,in_channel,main_gpu=main_gpu).cuda()
# E.apply(weights_init)
# G.apply(weights_init)

d_in_demension = 1
# D = model.D_Net_conv(ngpu,d_in_demension,main_gpu=main_gpu).cuda()
# d_model = torch.load("/home/bike/2027/generative-models/GAN/conditional_gan/new_f3_no_label_2017-05-10 13:45:32.768405/D_20000.model")
# D.load_state_dict(d_model)

G_solver = optim.Adam(G.parameters(),lr = 1e-4)
# D_solver = optim.Adam(D.parameters(), lr = 1e-4)
E_solver = optim.Adam(E.parameters(), lr=1e-4)

half_label = Variable(torch.ones(batch_size)*0.5).cuda()

check_points = 500
num_epoches = 100000
criterion = nn.BCELoss()
criterion_t = nn.TripletMarginLoss(p=1)
criterion_mse = nn.MSELoss()

epoch = 1



data, label = mm.batch_next(10000, shuffle=False)
data = torch.from_numpy(data.astype('float32'))
# data = data.repeat(batch_size // 10, 1, 1, 1)
# D.zero_grad()
G.zero_grad()
data_old = Variable(data).cuda()
data_old.data.resize_(10000, 1, 28, 28)
_, _, _, f3 = enet(data_old.detach())
# owntool.save_picture(np.array(f3.data.tolist()), out_dir + "feature_36.png", column=1)
scipy.misc.imsave("./feature_36.png",np.array(f3.data.tolist()))
