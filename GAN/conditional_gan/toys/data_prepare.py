import numpy as np
import torch
import torch.nn as nn
import torch.functional as F
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt



# 1. add sample during the training praocess. observe the process
# 2. add fixed modes one time.
# 3. sampling from a fixed dataset.
# 4. the pattern is fixed. ( a circle , a line, the grid)`

class Data_2D():
    def __init__(self, batch_size, mode_num, distance):
        self.mode_num = mode_num
        self.batch_size = batch_size
        self.distance = distance
        assert batch_size % (mode_num * mode_num) == 0
        self.mode_size = batch_size // (mode_num * mode_num)

    def batch_next(self):
        mode_matrix = np.zeros([self.batch_size, 2])
        for i in range(self.mode_num):
            for j in range(self.mode_num):
                random_bias_x = np.random.normal(0, 0.1, size=[self.mode_size, 1])
                random_bias_y = np.random.normal(0, 0.1, size=[self.mode_size, 1])

                mode_matrix[
                range((i * self.mode_num + j) * self.mode_size, (i * self.mode_num + j + 1) * self.mode_size, 1),
                0:2] = np.array(
                    [i * self.distance + random_bias_x, j * self.distance + random_bias_y]).transpose().reshape(
                    self.mode_size, 2)
        return mode_matrix


class Data_2D_Circle():
    def __init__(self, batch_size, mode_num, distance, noise_variance = 1):
        self.mode_num = mode_num
        self.batch_size = batch_size
        self.distance = distance
        self.noise_variance = noise_variance
        assert batch_size % (mode_num * mode_num) == 0
        self.mode_size = batch_size // (mode_num * mode_num)

    def draw_circle(self, R, part):
        unit = 2 * np.pi / part
        mode_matrix = np.zeros([part, 2])
        for i in range(part):
            mode_matrix[i, :] = [R * np.cos(unit * i), R * np.sin(unit * i)]

        return mode_matrix

    def batch_next(self, old_label, need_label=False, shuffle=False, ):
        mode_matrix = np.zeros([self.batch_size, 2])
        label_matrix = np.zeros([self.batch_size, 1])
        pattern_matrix = self.draw_circle(self.distance, self.mode_num * self.mode_num)

        if not shuffle:
            label = 0
            for i in range(self.mode_num):
                for j in range(self.mode_num):
                    random_bias_x = np.random.normal(0, self.noise_variance, size=[self.mode_size, 1])
                    random_bias_y = np.random.normal(0, self.noise_variance, size=[self.mode_size, 1])

                    mode_matrix[
                    range((i * self.mode_num + j) * self.mode_size, (i * self.mode_num + j + 1) * self.mode_size, 1),
                    0:2] = np.array(

                        [pattern_matrix[i * self.mode_num + j, 0] + random_bias_x,
                         pattern_matrix[i * self.mode_num + j, 1]
                         + random_bias_y]).transpose().reshape(
                        self.mode_size, 2)
                    label_matrix[
                        range((i * self.mode_num + j) * self.mode_size, (i * self.mode_num + j + 1) * self.mode_size,
                              1)] = label
                    label += 1

            if (need_label):
                return mode_matrix, label_matrix
            else:
                return mode_matrix

        if shuffle:
            # asssert: the type of the label is numpy, and the label is a given one.
            if old_label.shape[0] != self.batch_size:
                print("give me a right label, please")
            else:
                old_label = np.nonzero(old_label)[1]
                # print((old_label))
                noise_label = np.array([np.random.randint(1, np.max(old_label)) for i in range(0, self.batch_size)])
                label_matrix = (old_label + noise_label) % np.max(old_label + 1)
            # print(noise_label)

            for i in range(0, self.batch_size):
                mode_matrix[i, 0:2] = pattern_matrix[label_matrix[i].astype('int')]

            return mode_matrix, label_matrix

        #
# test = Data_2D_Circle(320,4,10)
# result = test.batch_next()
# print(result)
# plt.scatter(result[:,0],result[:,1],alpha=0.5)
# plt.s=
# test = Net_2D(320,4)
# result = test.batch_next()
# # print(result)
# plt.scatter(result[:,0],result[:,1],alpha=0.5)
# plt.show()


#
# batch_next(old_label=False, need_label=False, shuffle=False,negative = False):
class Data_2D_Curve():
    def __init__(self, batch_size, theta, R):
        self.batch_size = batch_size
        self.theta = theta
        self.R = R

    def sample_curve(self, size, theta, R=1):
        sample_data = np.zeros([size, 2])
        if (theta.size == 1):
            # prepare a guassian like data
            theta_pi = theta /  float(180) * np.pi
            sample_data_x = R * np.cos(theta_pi)
            sample_data_y = R * np.sin(theta_pi)
            sample_data_single = [sample_data_x, sample_data_y]
            sample_data = sample_data_single.repeat([size, 1])
        elif (theta.size == 2):
            theta_pi = theta / float(180) * np.pi
            sample_theta = np.random.rand(size, 1) * (theta_pi[1] - theta_pi[0]) + theta_pi[0]
            sample_data_x = R * np.cos(sample_theta)
            sample_data_y = R * np.sin(sample_theta)
            sample_data = np.concatenate([sample_data_x, sample_data_y], 1)
        else:
            print("please input a number or a interval")
            return -1
        return sample_data

    def sample(self,size, theta, R,negative = False ,noise="guassian",selected_label= False):
        theta_size = theta.size//2
        assert size%theta_size==0
        unit = size //theta_size
        assert R.size== theta_size
        data_pure = np.zeros([size,2])

        if(selected_label):
            label =np.repeat(selected_label-1,unit,0)
        else:
            label_list = np.arange(theta_size)
            label = np.repeat(label_list,unit,0)


        if(negative):
            noise_label = [np.random.randint(1,theta_size) for i in range(size)]
            label = (label+noise_label)% theta_size
        else:
            pass

        if(noise == 'r'):
            R_noise =np.random.normal(0,0.01,[size])
        else:
            R_noise = np.zeros([size])

# this part can influence the final result.
        for i in range(size):
            if(selected_label):
                data_pure[i,0:2] = self.sample_curve(1,theta,R+R_noise[i])
            else:
                data_pure[i,0:2] = self.sample_curve(1,theta[label[i]],R[label[i]]+R_noise[i])

        if(noise=='guassian'):
            data_noise = np.random.normal(0,1,size=[unit,2])
            data_pure = data_pure+ data_noise
        elif(noise == 'x'):
            data_noise = np.random.normal(0,1,size=[unit,1])
            data_pure[:,0]  = data_pure[:,0] + data_noise
        elif(noise == 'y'):
            data_noise = np.random.normal(0,1,size=[unit,1])
            data_pure[:,1] = data_pure[:,1] + data_noise
        elif(noise == 'r'):
            pass
            # R = R + np.random.normal(0,1,[size])
        else:
            print("what are you talking about?????")
            return -1
        return data_pure, label

    def batch_next(self,has_label = False,negative = False, selected_label=None):
        mode_matrix = np.zeros([self.batch_size, 2])
        label_matrix = np.zeros([self.batch_size, 1])
        # pattern_matrix = self.sample_curve(self.distance, self.mode_num )

        # a given structure
        if(selected_label):
            mode_matrix, label_matrix = self.sample(self.batch_size,negative=negative, theta=self.theta[selected_label-1], R = self.R[selected_label-1],noise='r',selected_label=selected_label)
            # pass
        else:
            mode_matrix, label_matrix = self.sample(self.batch_size,negative=negative,theta=self.theta, R=self.R, noise='r')

        if(has_label):
            return mode_matrix, label_matrix
        else:
            return mode_matrix

# test = Data_2D_Curve(300,3,10)
# result, label = test.batch_next(negative=True)
# print(result)
# print(label)
# plt.scatter(result[:,0],result[:,1],alpha=0.5)

# print(result)
# # plt.scatter(result[:,0],result[:,1],alpha=0.5)
# plt.show()
# plt.savefig('./hehe_0-01.png')