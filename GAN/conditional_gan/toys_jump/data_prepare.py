import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import torch
import random
from torchvision import datasets, transforms
import mutil

















class Straight_Line():
    def __init__(self,batch_size, start_points, end_points, type):
        self.batch_size = batch_size
        self.start_points = start_points
        self.end_points = end_points
        self.pattern_num = self.start_points.shape[0]
        self.dimension = self.start_points.shape[1]
        self.single_pattern_num = self.batch_size // self.pattern_num

        if(type==1):
            self.is_numpy = 1
            self.is_torch = 0
        else:
            self.is_numpy = 0
            self.is_torch = 1

    def sample_a_line(self, num, size):
        if(self.is_numpy):
            line_sample = np.zeros([size, self.dimension])
            for i in range(self.dimension):
                line_sample[0:size, i] = np.random.rand(size)*(self.end_points[num,i]-self.start_points[num,i])+ self.start_points[num,i]
            return line_sample

    def batch_next(self):
        sample_list = np.zeros([self.batch_size, self.dimension])
        if self.batch_size % self.pattern_num == 0:
            for i in range(0, self.batch_size, self.single_pattern_num):
                sample_list[i:i+self.single_pattern_num, 0:self.dimension ] = self.sample_a_line(i//self.single_pattern_num, self.single_pattern_num)
        return sample_list


# class Circle_Twinkle():
#     def __init__(self, ):



# testing code
<<<<<<< HEAD
start_points = np.array([[0,0],[0,1],[0,2]])
end_points = np.array([[1,0],[1,1],[1,2]])
data = Straight_Line(90, start_points, end_points, type=1)
tmp_data = data.batch_next()

# print(data.batch_next())

# class Data_2D_Circle():
#     def __init__(self, batch_size, mode_num, distance, noise_variance = 1):
#         self.mode_num = mode_num
#         self.batch_size = batch_size
#         self.distance = distance
#         self.noise_variance = noise_variance
#         assert batch_size % (mode_num * mode_num) == 0
=======
# start_points = np.array([[0,0],[0,1],[0,2]])
# end_points = np.array([[1,0],[1,1],[1,2]])
# data = Straight_Line(90, start_points, end_points, type=1)
# tmp_data = data.batch_next()
# plt.scatter(tmp_data[:,0], tmp_data[:,1])
# plt.show()

class Data_2D_Circle():
    def __init__(self, batch_size, R, mode_num = 8,noise_variance = 0.02):
        self.mode_num = mode_num
        self.batch_size = batch_size
        self.R = R
        # self.distance = distance
        self.noise_variance = noise_variance
        assert batch_size % mode_num == 0
        self.mode_size = batch_size // mode_num
>>>>>>> ae63e762f33b386ac4deff32850a45e5519af39c
#         self.mode_size = batch_size // (mode_num * mode_num)
#
    def draw_circle(self):
        unit = 2 * np.pi / self.mode_num
        mode_matrix = np.zeros([self.mode_num, 2])
        for i in range(self.mode_num):
            mode_matrix[i, :] = [self.R * np.cos(unit * i), self.R * np.sin(unit * i)]
        return mode_matrix
#
    def batch_next(self):
        # sample_list = np.zeros([self.batch_size,2])
        pattern_list = self.draw_circle()
        sample_list = pattern_list.repeat(self.mode_size,axis=0)
        random_bias_x = np.random.normal(0, self.noise_variance, size=[self.batch_size, 1])
        random_bias_y = np.random.normal(0, self.noise_variance, size=[self.batch_size, 1])
        sample_list = sample_list + np.concatenate((random_bias_x, random_bias_y),axis=1)

        return sample_list

# data = Data_2D_Circle(80,R = 2)
# tmp_data = data.batch_next()
# plt.scatter(tmp_data[:,0], tmp_data[:,1])
# plt.show()

class Mnist_10():
    def __init__(self, batch_size):
        self.batch_size = batch_size
        data_dir = "/home/sensetime/data/MNIST_data"

        train_loader = torch.utils.data.DataLoader(
            datasets.MNIST(data_dir, train=True, download=False,
                           transform=transforms.Compose(
                               [transforms.ToTensor(), transforms.Normalize((0,), (1,))])),
            batch_size=60000, shuffle=False)

        for batch_index, (data, label) in enumerate(train_loader):
            data_list = data.numpy()
            label_list = label.numpy()

        sorted_label = sorted(range(len(label_list)), key=lambda x: label_list[x])
        self.new_data_list = data_list[sorted_label]
        self.new_label_list = label_list[sorted_label]

    def _batch_next(self, size, label=list(range(10)), shuffle=True):
        class_num = 5000
        class_num_stone = np.array([0,5924,12666,18624,24755,30597,
                                    36018,41936,48201,54052])
        ls = np.size(label)
        if not shuffle:
            assert (size % ls == 0)
            unit = size // ls  # given the total number of the picture and the classes we need, calculate the number of each class
            return_list = np.zeros([size, 1, 28, 28])
            return_label = np.zeros([size])
            for i, il in enumerate(label):
                index = [int(class_num * random.random()) for _ in range(unit)]
                index = class_num_stone[i].astype('int')+index
                # index = class_num * il * np.ones(np.shape(index)).astype(int) + index
                return_list[i * unit:(i + 1) * unit] = self.new_data_list[index]
                return_label[i * unit:(i + 1) * unit] = il
        else:
            index = [int(60000 * random.random()) for i in range(size)]
            return_list = self.new_data_list[index]
            return_label = self.new_label_list[index]

        # return [return_list, return_label]
        return return_list

    def batch_next(self):
        tmp_data = self._batch_next(10,shuffle=False)
        assert self.batch_size % 10 ==0
        repeat_num = self.batch_size // 10
        tmp_data = tmp_data.repeat(repeat_num, axis = 0)
        return tmp_data

# testing code
# data = Mnist_10(100)
# tmp_data = data.batch_next()
# mutil.save_picture_numpy(tmp_data,"./test.png")


#         mode_matrix = np.zeros([self.batch_size, 2])
#         label_matrix = np.zeros([self.batch_size, 1])
#         pattern_matrix = self.draw_circle(self.distance, self.mode_num * self.mode_num)
#
#         if not shuffle:
#             label = 0
#             for i in range(self.mode_num):
#                 for j in range(self.mode_num):
#                     random_bias_x = np.random.normal(0, self.noise_variance, size=[self.mode_size, 1])
#                     random_bias_y = np.random.normal(0, self.noise_variance, size=[self.mode_size, 1])
#
#                     mode_matrix[
#                     range((i * self.mode_num + j) * self.mode_size, (i * self.mode_num + j + 1) * self.mode_size, 1),
#                     0:2] = np.array(
#
#                         [pattern_matrix[i * self.mode_num + j, 0] + random_bias_x,
#                          pattern_matrix[i * self.mode_num + j, 1]
#                          + random_bias_y]).transpose().reshape(
#                         self.mode_size, 2)
#                     label_matrix[
#                         range((i * self.mode_num + j) * self.mode_size, (i * self.mode_num + j + 1) * self.mode_size,
#                               1)] = label
#                     label += 1
#
#             if (need_label):
#                 return mode_matrix, label_matrix
#             else:
#                 return mode_matrix
#
#         # if shuffle:
#         #     # asssert: the type of the label is numpy, and the label is a given one.
#         #     if old_label.shape[0] != self.batch_size:
#         #         print("give me a right label, please")
#         #     else:
#         #         old_label = np.nonzero(old_label)[1]
#         #         # print((old_label))
#         #         noise_label = np.array([np.random.randint(1, np.max(old_label)) for i in range(0, self.batch_size)])
#         #         label_matrix = (old_label + noise_label) % np.max(old_label + 1)
#         #     # print(noise_label)
#         #
#         #     for i in range(0, self.batch_size):
#         #         mode_matrix[i, 0:2] = pattern_matrix[label_matrix[i].astype('int')]
#         #
#         #     return mode_matrix, label_matrix
