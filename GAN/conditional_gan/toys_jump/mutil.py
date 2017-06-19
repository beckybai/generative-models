import os
import shutil
import numpy as np
import torch
import sys

import scipy
import scipy.misc
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


class Logger(object):
    def __init__(self,path):
        self.terminal = sys.stdout
        self.log = open("{}/logfile.log".format(path), "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        #this flush method is needed for python 3 compatibility.
        #this handles the flush command by doing nothing.
        #you might want to specify some extra behavior here.
        pass

def label_num2vec(num_vec, max_label = 2):
    batch_size = num_vec.shape[0]
    if(np.max(num_vec)==0) or np.max(num_vec)==np.min(num_vec):
        pass
    else:
        max_label = np.max(num_vec)
#    print(num_vec)
    label_mat = np.zeros([batch_size,max_label+1])
    for i in range(0,batch_size):
#        print(num_vec[i])
    	label_mat[i,num_vec[i]]=1
    return label_mat



# save the pictures by the pixels.
def save_picture(pic, output_dir,column = 10,image_size=28):
    batch_size = np.shape(pic.data.tolist())[0]
    # column = 6
    row = batch_size // column
    rec_data = np.reshape(pic.data.tolist(), (batch_size, image_size, image_size))
    rec_data = np.concatenate(rec_data,0)
    # scipy.misc.imsave(output_dir, rec_data)
    row_colum = []
    for i in range(row):
        rows = []
        for j in range(column):
            rows.append(rec_data[(column*i+j)*image_size:(column*i+ (j+1))*image_size])
        rows_whole = np.concatenate(rows,1)
        rows_white = np.concatenate([np.array(rows_whole),np.ones([2,image_size*column])])
        row_colum.append(rows_white)
        # add white squra
    scipy.misc.imsave(output_dir, np.concatenate( row_colum , 0 ))

def extract(v):
    return v.data.storage().tolist()

def stats(d):
    return [np.mean(d), np.std(d)]


# for cifar 10
def save_color_pictrue(pic, output_dir, column = 10, image_size = 32):
    row_num = 1
    fig = plt.figure(figsize=(row_num, 10))
    gs = gridspec.GridSpec(10, row_num)
    gs.update(wspace=0.05, hspace=0.05)

    for i, sample in enumerate(pic):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        plt.imshow(np.rollaxis(sample,0,3), interpolation='nearest')

    plt.savefig(output_dir, bbox_inches='tight')

    plt.close(fig)

# we have 100 classes in total : )
def save_color_picture_pixel(pic, output_dir, column = 10, image_size = 32,mb_size = 100):
    reconx = np.transpose(np.reshape(pic, (mb_size, 3, image_size, image_size)), (0, 2, 3, 1))
    reconx = [reconx[i] for i in range(mb_size)]
    rows = []
    for i in range(column):
        rows.append(np.concatenate(reconx[i::column], 1))
    reconx = np.concatenate(rows, 0)
    scipy.misc.imsave(output_dir, reconx)