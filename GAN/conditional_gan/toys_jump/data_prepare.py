import numpy as np
import matplotlib as mpl
# mpl.use('Agg')
import matplotlib.pyplot as plt


















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

# testing code
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
#         self.mode_size = batch_size // (mode_num * mode_num)
#
#     def draw_circle(self, R, part):
#         unit = 2 * np.pi / part
#         mode_matrix = np.zeros([part, 2])
#         for i in range(part):
#             mode_matrix[i, :] = [R * np.cos(unit * i), R * np.sin(unit * i)]
#
#         return mode_matrix
#
#     def batch_next(self, need_label=False, shuffle=False, ):
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
