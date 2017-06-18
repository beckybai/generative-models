import numpy as np
import matplotlib as mpl
mpl.use('Agg')


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

    def batch_next(self, need_label=False, shuffle=False, ):
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

        # if shuffle:
        #     # asssert: the type of the label is numpy, and the label is a given one.
        #     if old_label.shape[0] != self.batch_size:
        #         print("give me a right label, please")
        #     else:
        #         old_label = np.nonzero(old_label)[1]
        #         # print((old_label))
        #         noise_label = np.array([np.random.randint(1, np.max(old_label)) for i in range(0, self.batch_size)])
        #         label_matrix = (old_label + noise_label) % np.max(old_label + 1)
        #     # print(noise_label)
        #
        #     for i in range(0, self.batch_size):
        #         mode_matrix[i, 0:2] = pattern_matrix[label_matrix[i].astype('int')]
        #
        #     return mode_matrix, label_matrix