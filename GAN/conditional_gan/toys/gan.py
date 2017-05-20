import torch
import torch.autograd as autograd
import torch.optim as optim
import numpy as np
import matplotlib as mpl
from datetime import datetime

mpl.use('Agg')

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F
import shutil, sys
import mutil
import toy_model as model
import data_prepare

out_dir = './out/gan_{}'.format(datetime.now())
out_dir = out_dir.replace(" ", "_")
print(out_dir)

if not os.path.exists(out_dir):
	os.makedirs(out_dir)
	shutil.copyfile(sys.argv[0], out_dir + '/training_script.py')

sys.stdout = mutil.Logger(out_dir)
gpu = 0
torch.cuda.set_device(gpu)
mb_size = 100  # mini-batch_size
mode_num = 2

distance = 10
data = data_prepare.Data_2D_Circle(mb_size, mode_num, distance)

Z_dim = 2
X_dim = 2
h_dim = 128
cnt = 0

num = '0'

# else:
#     print("you have already creat one.")
#     exit(1)

G = model.G_Net(Z_dim, X_dim, h_dim).cuda()
D = model.D_Net(X_dim, 1, h_dim).cuda()

G.apply(model.weights_init)
D.apply(model.weights_init)

""" ===================== TRAINING ======================== """

G_solver = optim.Adam(G.parameters(), lr=1e-4)
D_solver = optim.Adam(D.parameters(), lr=1e-4)

ones_label = Variable(torch.ones(mb_size)).cuda()
zeros_label = Variable(torch.zeros(mb_size)).cuda()

criterion = nn.BCELoss()

grads = {}


def save_grad(name):
	def hook(grad):
		grads[name] = grad

	return hook


z_fixed = Variable(torch.randn(mb_size, Z_dim)).cuda()
y_fixed,x_fixed = np.mgrid[0:12:100j,-10:13:100j]
mesh_fixed_cpu = np.array([x_fixed, y_fixed])
mesh_fixed = Variable(torch.from_numpy(mesh_fixed_cpu).cuda())
# mesh_fixed.register_hook(save_grad('Mesh'))


def get_grad(input,label,name):
	sample = G(input)
	sample.regsitor_hook(save_grad(name))
	d_result = D(sample)
	loss_real = criterion(d_result,ones_label*label)
	loss_real.backward()
	return d_result


for it in range(10000):
	# Sample data
	z = Variable(torch.randn(mb_size, Z_dim)).cuda()
	X = data.batch_next()
	X = Variable(torch.from_numpy(X.astype('float32'))).cuda()

	D_solver.zero_grad()
	# Dicriminator forward-loss-backward-update
	G_sample = G(z)
	D_real = D(X)
	D_fake = D(G_sample)

	D_loss_real = criterion(D_real, ones_label)
	D_loss_fake = criterion(D_fake, zeros_label)
	D_loss = D_loss_real + D_loss_fake

	D_loss.backward()
	D_solver.step()

	# Housekeeping - reset gradient
	D.zero_grad()
	G.zero_grad()

	# Generator forward-loss-backward-update
	z = Variable(torch.randn(mb_size, Z_dim).cuda(), requires_grad=True)
	G_sample = G(z)
	G_sample.register_hook(save_grad('G'))
	# G_sample.requires_grad= True
	D_fake = D(G_sample)
	G_loss = criterion(D_fake, ones_label)

	G_loss.backward()
	G_solver.step()
	# print(grads['G'])

	# Housekeeping - reset gradient
	D.zero_grad()
	G.zero_grad()

	# Print and plot every now and then
	if it % 30 == 0:
		fig, ax = plt.subplots()

		print('Iter-{}; D_loss_real/fake: {}/{}; G_loss: {}'.format(it, D_loss_real.data.tolist(),
		                                                            D_loss_fake.data.tolist(), G_loss.data.tolist()))
		X = X.cpu().data.numpy()
		G_sample = G_sample.cpu().data.numpy()
		get_grad(z_fixed,0,'fixed_false')
		get_grad(z_fixed,1,'fixed_truth')
		d_mesh = (get_grad(mesh_fixed,1,'mesh')).cpu().data.numpy()

		G_sample_cpu = G_sample.cpu().data.numpy()
		gd_cpu = grads['G'].cpu().data.numpy()
		ax.quiver(G_sample_cpu[:, 0], G_sample_cpu[:, 1], gd_cpu[:, 0], gd_cpu[:, 1])


		gd_mesh_cpu = grads['mesh'].data.numpy()
		ax.quiver(x_fixed,y_fixed,gd_mesh_cpu[:,0],gd_mesh_cpu[:,1],d_mesh)

		gd_fixed_cpu = grads['fixed_truth'].data.numpy()
		z_fixed_cpu = z_fixed.cpu().data.numpy()
		ax.quiver(z_fixed_cpu[:,0],z_fixed_cpu[:,1],gd_fixed_cpu[:,0],gd_fixed_cpu[:,1])

		ax.set(aspect=1, title='Quiver Plot')

		plt.scatter(X[:, 0], X[:, 1], s=1, edgecolors='blue', color='blue')
		plt.scatter(G_sample[:, 0], G_sample[:, 1], s=1, color='red', edgecolors='red')
		plt.show()
		plt.savefig('{}/hehe_{}.png'.format(out_dir, str(cnt).zfill(3)), bbox_inches='tight')
		plt.close()
		cnt += 1

		test_command = os.system("convert -quality 100 -delay 20 {}/*.png {}/video.mp4".format(out_dir, out_dir))

		torch.save(G.state_dict(), "{}/G.model".format(out_dir))
		torch.save(D.state_dict(), "{}/D.model".format(out_dir))