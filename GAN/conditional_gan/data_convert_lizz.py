# pylint: skip-file
import torch
from torchvision import datasets, transforms
import numpy as np
from random import randrange
import random
import owntool
from torch.utils.serialization import load_lua
from pytimer import Timer

class owndata():
    def __init__(self):
        data_dir = "../../MNIST_data"

        train_loader =  torch.utils.data.DataLoader(
            datasets.MNIST(data_dir, train = True,download=False,
                           transform= transforms.Compose([transforms.ToTensor(),   transforms.Normalize((0,), (1,))])),
            batch_size = 60000, shuffle=False)

        for batch_index,(data, label) in enumerate(train_loader):
            data_list = data.numpy()
            label_list = label.numpy()

        sorted_label = sorted(range(len(label_list)),key=lambda x:label_list[x])
        self.new_data_list = data_list[sorted_label]
        self.new_label_list = label_list[sorted_label]

    def batch_next(self,size,label=list(range(10)), shuffle=True):
        class_num = 6000
        ls = np.size(label)
        if not shuffle:
            assert(size%ls==0)
            unit =size//ls # given the total number of the picture and the classes we need, calculate the number of each class
            return_list = np.zeros([size,1,28,28])
            return_label = np.zeros([size])
            for i, il in enumerate(label):
                index =  [int( class_num * random.random()) for i in range(unit)]
                index = class_num*il*np.ones(np.shape(index)).astype(int) + index
                return_list[i*unit:(i+1)*unit] = self.new_data_list[index]
                return_label[i*unit:(i+1)*unit] = il
        else:
            index = [int(60000*random.random()) for i in range(size)]
            return_list = self.new_data_list[index]
            return_label = self.new_label_list[index]

        return [return_list, return_label]
        # return [torch.from_numpy(return_list.astype('float32')), torch.from_numpy(return_label.astype('float32'))]
class cifar100():
    def __init__(self):
        data_dir = "../../CIFAR100_data"

        train_loader = torch.utils.data.DataLoader(
            datasets.CIFAR100(data_dir, train=True, download=False,
                           transform=transforms.Compose(
                               [transforms.ToTensor(), transforms.Normalize((0,), (1,))])),
            batch_size=50000, shuffle=True)

        for batch_index, (data, label) in enumerate(train_loader):
            data_list = data.numpy()
            label_list = label.numpy()

        sorted_label = sorted(range(len(label_list)), key=lambda x: label_list[x])
        self.new_data_list = data_list[sorted_label]
        self.new_label_list = label_list[sorted_label]

    def batch_next(self, size, label=list(range(100)), shuffle=True):
        class_num = 500
        ls = np.size(label)
        if not shuffle:
            assert (size % ls == 0)
            unit = size // ls
            return_list = np.zeros([size, 3, 32, 32])
            return_label = np.zeros([size])
            for i, il in enumerate(label):
                index = [int(class_num * random.random()) for i in range(unit)]
                index = class_num * il * np.ones(np.shape(index)).astype(int) + index
                return_list[i * unit:(i + 1) * unit] = self.new_data_list[index]
                return_label[i * unit:(i + 1) * unit] = il
        else:
            index = [int(50000 * random.random()) for i in range(size)]
            return_list = self.new_data_list[index]
            return_label = self.new_label_list[index]

        return [return_list, return_label]
                # return [torch.from_numpy(return_list.astype('float32')), torch.from_numpy(return_label.astype('float32'))]
class cifar10():
    def __init__(self):
        data_dir = "../../CIFAR10_data"

        train_loader = torch.utils.data.DataLoader(
            datasets.CIFAR10(data_dir, train=True, download=True,
                              transform=transforms.Compose(
                                  [transforms.ToTensor(), transforms.Normalize((0,), (1,))])),
            batch_size=50000, shuffle=True)

        for batch_index, (data, label) in enumerate(train_loader):
            data_list = data.numpy()
            label_list = label.numpy()

        sorted_label = sorted(range(len(label_list)), key=lambda x: label_list[x])
        self.new_data_list = data_list[sorted_label]
        self.new_label_list = label_list[sorted_label]

    def batch_next(self, size, label=list(range(10)), shuffle=True):
        class_num = 5000
        ls = np.size(label)
        if not shuffle:
            assert (size % ls == 0)
            unit = size // ls
            return_list = np.zeros([size, 3, 32, 32])
            return_label = np.zeros([size])
            for i, il in enumerate(label):
                index = [int(class_num * random.random()) for i in range(unit)]
                index = class_num * il * np.ones(np.shape(index)).astype(int) + index
                return_list[i * unit:(i + 1) * unit] = self.new_data_list[index]
                return_label[i * unit:(i + 1) * unit] = il
        else:
            index = [int(50000 * random.random()) for i in range(size)]
            return_list = self.new_data_list[index]
            return_label = self.new_label_list[index]

        return [return_list, return_label]


# mm = cifar100()
# [pic, label] = mm.batch_next(100, shuffle = False)
# owntool.save_color_picture_pixel(pic,'./ouuu3.jpg')

# mm = owndata()
# print(' kb is stupid')

# for i in range(10):
#     print(mm.batch_next(10,[0,1,2,3,4,5,6,7,8,9],True))
# [a,b] = mm.batch_next(50,[0,1,2,3,4])

class celebA():
    def __init__(self):
        data_dir = '/home/bike/data/celebA/images.dmp'
        label_dir = '/home/bike/data/celebA/imLabels.dmp'
        self.total_size = 200000 # in fact we have 202599 in total // and 18 classes
        self.label_size = 18
        self.data_raw = (load_lua(data_dir))[0:self.total_size]
        # self.data_raw = a.tolist()
        b = load_lua(label_dir)[0:self.total_size]
        self.label_raw = (b+1)/2
        self.t = 1
        self.label_dec = self.vec2num(self.label_raw)
        self.label_dec_list = self.label_dec.tolist()
        self.label_dec_set = set(self.label_dec_list)
        self.label_dec_array = np.array(self.label_dec_list)

    def vec2num(self,bin_label):
        dec_label = 0
        k = 1
        for i in range(self.label_size):
            dec_label = dec_label + bin_label[:,self.label_size-i-1]*k
            k = k * 2
        return  dec_label

    def batch_next(self, batch_size, label, shuffle=True, Negative =False):
        # the size of the label == batch_size                 ( since we have so many labels)
        if(Negative):
            label_index = (self.vec2num(label)).tolist()
            pic_list = torch.FloatTensor(batch_size, 3, 64, 64)
            label_list = torch.FloatTensor(batch_size, 18)
            for i in range(batch_size):
                ii = np.where(self.label_dec_array == (label_index[i]))[0]
                ii_size = ii.shape[0]
                if (ii_size == 1):
                    flip_one = np.array((self.data_raw[np.asscalar(ii)]).tolist())
                    flip_one = flip_one[:, :, ::-1]
                    pic_list[i, :, :, :] = torch.from_numpy(flip_one.astype('float32'))
                else:
                    select_index = np.random.randint(0, self.total_size)
                    while(select_index in ii):
                        select_index = (np.random.randint(0, self.total_size))
                    pic_list[i, :, :, :] = self.data_raw[select_index]
                    label_list[i,:] = self.label_raw[select_index]
            return [pic_list, label_list]

        else:
            if(shuffle):
                index = torch.LongTensor([int(self.total_size*random.random()) for i in range(batch_size)])
                pic_list = self.data_raw[index]
                label_list = self.label_raw[index]
                return [pic_list, label_list]

            else:
                if not label.size()[0]==batch_size:
                    print("Please give me the right label...")
                    return 0
                else:
                    label_index = (self.vec2num(label)).tolist()
                    pic_list = torch.FloatTensor(batch_size,3,64,64)
                    label_list = torch.FloatTensor(batch_size,18)
                    for i in range(batch_size):
                        ii = np.where(self.label_dec_array == (label_index[i]))[0]
                        ii_size = ii.shape[0]
                        if(ii_size==1):
                            # print(ii)
                            flip_one = np.array((self.data_raw[np.asscalar(ii)]).tolist())
                            flip_one = flip_one[:,:,::-1]
                            pic_list[i,:,:,:] = torch.from_numpy(flip_one.astype('float32'))
                        else:
                            select_index = np.random.randint(0,ii_size)
                            pic_list[i,:,:,:] = self.data_raw[np.asscalar(ii[select_index])]
                    return [pic_list, label]
        # print("to do ...")

mm = celebA()
mb_size = 100
timer = Timer()
timer.start()
X, c =mm.batch_next(mb_size, label=1, shuffle=True)
timer.checkpoint('anchor')
mm.batch_next(mb_size, label=c, shuffle=False)
timer.checkpoint('positive')
mm.batch_next(mb_size, label=c, shuffle=False, Negative=True)
timer.checkpoint('negative')
timer.summary()
timer.reset()

def timeit():
    timer = Timer()
    timer.start()
    X, c =mm.batch_next(mb_size, label=1, shuffle=True)
    timer.checkpoint('anchor')
    mm.batch_next(mb_size, label=c, shuffle=False)
    timer.checkpoint('positive')
    mm.batch_next(mb_size, label=c, shuffle=False, Negative=True)
    timer.checkpoint('negative')
    timer.summary()
    timer.reset()
