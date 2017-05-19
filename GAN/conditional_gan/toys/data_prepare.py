import numpy as np
import torch
import torch.nn as nn
import torch.functional as F
import numpy as np
import matplotlib.pyplot as plt

# 1. add sample during the training praocess. observe the process
# 2. add fixed modes one time.
# 3. sampling from a fixed dataset.
# 4. the pattern is fixed. ( a circle , a line, the grid)`

class Data_2D():
	def __init__(self, batch_size, mode_num,distance ):
		self.mode_num = mode_num
		self.batch_size = batch_size
		self.distance = distance
		assert batch_size % (mode_num * mode_num) == 0
		self.mode_size = batch_size // (mode_num * mode_num)



	def batch_next(self):
		mode_matrix  = np.zeros([self.batch_size,2])
		for i in range(self.mode_num):
			for j in range(self.mode_num):
				random_bias_x = np.random.normal(0,0.1,size=[self.mode_size,1])
				random_bias_y = np.random.normal(0,0.1,size = [self.mode_size,1])



				mode_matrix[range((i*self.mode_num+j)*self.mode_size,(i*self.mode_num+j+1)*self.mode_size,1),0:2]=np.array([i*self.distance+random_bias_x, j*self.distance +random_bias_y]).transpose().reshape(self.mode_size,2)
		return mode_matrix

class Data_2D_Circle():
	def __init__(self, batch_size, mode_num, distance):
		self.mode_num = mode_num
		self.batch_size = batch_size
		self.distance = distance
		assert batch_size % (mode_num * mode_num) == 0
		self.mode_size = batch_size // (mode_num * mode_num)

	def draw_circle(self,R,part):
		unit = np.pi/part
		mode_matrix = np.zeros([part,2])
		for i in range(part):
			mode_matrix[i,:]=[R*np.sin(unit)]


	def batch_next(self):
		mode_matrix = np.zeros([self.batch_size, 2])
		for i in range(self.mode_num):
			for j in range(self.mode_num):
				random_bias_x = np.random.normal(0, 0.1, size=[self.mode_size, 1])
				random_bias_y = np.random.normal(0, 0.1, size=[self.mode_size, 1])

				mode_matrix[
				range((i * self.mode_num + j) * self.mode_size, (i * self.mode_num + j + 1) * self.mode_size, 1),
				0:2] = np.array(

					[pattern_matrix[i*self.mode_num+j,0]  + random_bias_x, pattern_matrix[i*self.mode_num+j,1]
					  + random_bias_y]).transpose().reshape(
					self.mode_size, 2)
		return mode_matrix

		#


#test = Data_2D_Circle(320,4,10)
#result = test.batch_next()
# print(result)
#plt.scatter(result[:,0],result[:,1],alpha=0.5)
#plt.s=
# test = Net_2D(320,4)
# result = test.batch_next()
# # print(result)
# plt.scatter(result[:,0],result[:,1],alpha=0.5)
# plt.show()

