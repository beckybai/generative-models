import torch
from torchvision import datasets, transforms
import numpy as np
from random import randrange
import random
import owntool
from torch.utils.serialization import load_lua
import os,sys
import shutil
import h5py


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

    def get_neigbor(self, vec):
        all_neighbor = (np.zeros([19,18])).astype('int')
        ve_size = 18
        all_neighbor[0,:]= vec.astype('int')
        # ve_size = np.shape(vec)[0]
        for i in range(ve_size):
            tmp_vec = vec.copy()
            if(tmp_vec[i]==0):
                tmp_vec[i] = 1
            else:
                tmp_vec[i] = 0
            all_neighbor[i+1] = tmp_vec

        return self.vec2num(all_neighbor)

    def batch_next(self, batch_size, label, shuffle=True, Negative =False, Neigbor=False):
        # the size of the label == batch_size                 ( since we have so many labels)
        if(Negative):
            label = np.array(label.tolist())
            label_index = (self.vec2num(label))
            pic_list = torch.FloatTensor(batch_size, 3, 64, 64)
            label_list = torch.FloatTensor(batch_size, 18)
            for i in range(batch_size):
                if(Neigbor):
                    ii = []
                    label_index_group = (self.get_neigbor(label[i]))
                    for j in range(label_index_group.shape[0]):
                        ii.extend(np.where(self.label_dec_array == (label_index_group[j]))[0])
                    ii = np.array(ii)
                else:
                    ii = np.where(self.label_dec_array==label_index[i])[0]
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
                    label_list = np.array(label.tolist())
                    label_index = (self.vec2num(label_list))
                    pic_list = torch.FloatTensor(batch_size,3,64,64)
                    # label_list = torch.FloatTensor(batch_size,18)
                    for i in range(batch_size):
                        if (Neigbor):
                            ii = []
                            label_index_group = (self.get_neigbor(label_list[i]))
                            for j in range(label_index_group.shape[0]):
                                ii.extend(np.where(self.label_dec_array == (label_index_group[j]))[0])
                            ii = np.array(ii)
                        else:
                            ii = np.where(self.label_dec_array == label_index[i])[0]
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

#------------------Testing Code for celebA---------------------
# celebA = celebA()
# mb_size = 100
#
# X, c = celebA.batch_next(mb_size, label = 1,shuffle=True)
# out_dir = './test3/'
# out_dir.replace(" ","_")
# if not os.path.exists(out_dir):
#     os.makedirs(out_dir)
#     shutil.copyfile(sys.argv[0], out_dir + '/shuideguo.py')
#
#
# owntool.save_color_picture_pixel(X.tolist(), out_dir+'./X.jpg', image_size=64, column=10, mb_size=100)
#
# X2, c2 = celebA.batch_next(mb_size, label= c, shuffle=False, Neigbor=False)
# owntool.save_color_picture_pixel(X2.tolist(), out_dir+'./X2.jpg', image_size=64, column=10, mb_size=100)
#
# X3, c3 = celebA.batch_next(mb_size, label=c, shuffle=False, Negative = True, Neigbor=False)
# owntool.save_color_picture_pixel(X3.tolist(), out_dir+'./X3.jpg', image_size=64, column=10, mb_size=100)
class celebA_identity():
    def __init__(self):
        data_dir = '/home/bike/data/celebA/images.dmp'
        label_dir = '/home/bike/data/celebA/id.h5'
        file = h5py.File(label_dir, 'r')
        self.label_raw_np = np.array(file['id'])  # numpy format
        self.total_size = 200000 # in fact we have 202599 in total // and 18 classes
        self.label_size = 10177 # statistics from the origin data
        self.label_size = 14
        self.data_raw = (load_lua(data_dir))[0:self.total_size]
        # self.data_raw = a.tolist()
        # b = load_lua(label_dir)[0:self.total_size]
        self.label_raw_np = self.label_raw_np[0:self.total_size]
        self.t = 1
        self.label_vec = self.num2vec(self.label_raw_np)
        self.label_dec_list = self.label_raw_np.tolist()
        self.label_dec_set = set(self.label_dec_list) # python: label set
        self.label_dec_array = np.array(self.label_dec_set) # np: label set
        self.label_raw = torch.from_numpy(self.label_vec) # torch tensor: one-hot vector

    def num2vec(self,dec_label):
        total_size = np.size(dec_label)
        total_vec = np.zeros([total_size,14])

        for i in range(total_size):
            zero_list = np.zeros([1,14])
            num_str = list( map(int, list("{0:b}".format(dec_label[i]))))
            num_str_size = np.size(num_str)
            zero_list[0,14-num_str_size:14] = num_str
            total_vec[i,:] = zero_list

        return total_vec


    def vec2num(self,bin_label):
        dec_label = 0
        k = 1
        for i in range(self.label_size):
            dec_label = dec_label + bin_label[:,self.label_size-i-1]*k
            k = k * 2
        return  dec_label

    def get_neigbor(self, vec):
        all_neighbor = (np.zeros([self.label_size+1,self.label_size])).astype('int')
        ve_size = self.label_size
        all_neighbor[0,:]= vec.astype('int')
        # ve_size = np.shape(vec)[0]
        for i in range(ve_size):
            tmp_vec = vec.copy()
            if(tmp_vec[i]==0):
                tmp_vec[i] = 1
            else:
                tmp_vec[i] = 0
            all_neighbor[i+1] = tmp_vec

        return self.vec2num(all_neighbor)

    def batch_next(self, batch_size, label, shuffle=True, Negative =False, Neigbor=False):
        # the size of the label == batch_size                 ( since we have so many labels)
        if(Negative):
            label = np.array(label.tolist())
            label_index = (self.vec2num(label))
            pic_list = torch.FloatTensor(batch_size, 3, 64, 64)
            label_list = torch.FloatTensor(batch_size, self.label_size)
            for i in range(batch_size):
                if(Neigbor):
                    ii = []
                    label_index_group = (self.get_neigbor(label[i]))
                    for j in range(label_index_group.shape[0]):
                        ii.extend(np.where(self.label_raw_np == (label_index_group[j]))[0])
                    ii = np.array(ii)
                else:
                    ii = np.where(self.label_raw_np==label_index[i])[0]
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
                # right
                index = torch.LongTensor([int(self.total_size*random.random()) for i in range(batch_size)])
                pic_list = self.data_raw[index]
                label_list = self.label_raw[index]
                return [pic_list, label_list]

            else:
                if not label.size()[0]==batch_size:
                    print("Please give me the right label...")
                    return 0
                else:
                    label_list = np.array(label.tolist())
                    label_index = (self.vec2num(label_list))
                    pic_list = torch.FloatTensor(batch_size,3,64,64)
                    # label_list = torch.FloatTensor(batch_size,18)
                    for i in range(batch_size):
                        if (Neigbor):
                            ii = []
                            label_index_group = (self.get_neigbor(label_list[i]))
                            for j in range(label_index_group.shape[0]):
                                ii.extend(np.where(self.label_raw_np == (label_index_group[j]))[0])
                            ii = np.array(ii)
                        else:
                            ii = np.where(self.label_raw_np == label_index[i])[0]
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