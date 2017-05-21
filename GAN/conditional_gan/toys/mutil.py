import os
import shutil
import numpy as np
import torch
import sys

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

def label_num2vec(num_vec):
    batch_size = num_vec.shape[0]
    max_label = np.max(num_vec)
    label_mat = np.zeros(batch_size,max_label+1)
    label_mat[:,num_vec]=1
    return label_mat