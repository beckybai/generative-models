import torch
from torchvision import datasets, transforms
import numpy as np
from random import randrange
import random
import owntool


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

mm = owndata()
print(' kb is stupid')

# for i in range(10):
#     print(mm.batch_next(10,[0,1,2,3,4,5,6,7,8,9],True))
# [a,b] = mm.batch_next(50,[0,1,2,3,4])