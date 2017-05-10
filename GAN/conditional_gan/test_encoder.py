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

import os
import sys, shutil

ngpu = 1
if __name__ == '__main__':
    f1_dimension = 24  # the dimension of the feature map in the first layer
    f2_dimension = 8

torch.cuda.set_device(4)

mnist = input_data.read_data_sets('../../MNIST_data', one_hot=False)

seed_num = 1
torch.cuda.manual_seed(seed_num)
data_dir = '/home/bike/data/mnist/'
out_dir = '/home/bike/2027/generative-models/GAN/conditional_gan/test_feature{}/'.format(datetime.now())

if not os.path.exists(out_dir):
    os.makedirs(out_dir)
    shutil.copyfile(sys.argv[0], out_dir + '/fm_2to1.py')

sys.stdout = mutil.Logger(out_dir)

batch_size = 50
ms = f1_dimension  # 28 for mnist data

kwargs = {'num_workers': 1, 'pin_memory': True}
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST(data_dir, train=True, download=False,
                   transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0,), (1,))])),
    batch_size=batch_size, shuffle=True, **kwargs)

# layer_names = list(model.keys())

# write the origin model again.

D0 = model.E_Net_2()
encoder_model = torch.load("./Dmodel")

# 24*24


enet = model.E_Net_2().cuda()
enet.load_state_dict(encoder_model)



for epoch in (range(1)):
    # print("give me a clue")
    data, label = mnist.train.next_batch(batch_size)
    print(label)

    data = Variable(torch.from_numpy(data)).cuda()
    data.data.resize_(batch_size, 1, 28, 28)

    # data, label = Variable(data), Variable(label)
    c_v = Variable(torch.from_numpy(model.set_label_ve(label, ms).astype('float32'))).cuda()

    _, f1, f2 = enet(data.detach())
    f1 = f1.data.tolist()
    f2 = f2.data.tolist()
    # print(np.shape(owntool.extract(f1)))

    for i in range(batch_size):
        f1_tmp, f2_tmp = np.expand_dims(f1[i],1),np.expand_dims(f2[i],1)

        g_f1 = Variable(torch.from_numpy(f1_tmp)).cuda()
        g_f2 = Variable(torch.from_numpy(f2_tmp)).cuda()
        # print(i)
        owntool.save_picture(g_f1,'{}f1_{}.png'.format(out_dir,i),image_size =24)
        owntool.save_picture(g_f2,'{}f2_{}.png'.format(out_dir,i),image_size = 8)


    # visulize the origin data. the first and second conv layers.
