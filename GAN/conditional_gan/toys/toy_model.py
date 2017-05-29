import torch
import torch.autograd as autograd
import torch.optim as optim
import numpy as np
import torch.nn as nn
import torch.functional as F


class G_Net(nn.Module):
    def __init__(self, input_dimension, output_dimension, hidden_dimension=256):
        # self.c = c
        super(G_Net, self).__init__()
        self.pipeline1 = nn.Sequential(
            nn.Linear(input_dimension, hidden_dimension, bias=True),
            nn.ELU(),
            nn.Linear(hidden_dimension, hidden_dimension * 2, bias=True),
            nn.ELU(),
            nn.Linear(hidden_dimension*2, hidden_dimension*4, bias=True),
            nn.ELU(),
            nn.Linear(hidden_dimension * 4, output_dimension, bias=True),
            #	        nn.ELU()
        )

    def forward(self, x):
        output = self.pipeline1(x)
        return output


class Direct_Net(nn.Module):
    def __init__(self, input_dimension, output_dimension, hidden_dimension=256):
        # self.c = c
        self.c = input_dimension

    def forward(self, x):
        output = x
        return output

class D_Net(nn.Module):
    def __init__(self, input_dimension, output_dimension, hidden_dimension=256):
        super(D_Net, self).__init__()
        self.main = nn.Sequential(
            nn.Linear(input_dimension, hidden_dimension, bias=True),
            nn.ELU(),
            nn.Linear(hidden_dimension, hidden_dimension * 2, bias=True),
            nn.ELU(),
            nn.Linear(hidden_dimension * 2, hidden_dimension*4, bias=True),
            nn.ELU(),
            nn.Linear(hidden_dimension*4, output_dimension, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        output = self.main(x)
        return output

class D_Net_w(nn.Module):
    def __init__(self, input_dimension, output_dimension, hidden_dimension=256):
        super(D_Net_w, self).__init__()
        self.main = nn.Sequential(
            nn.Linear(input_dimension, hidden_dimension, bias=True),
            nn.ELU(),
            nn.Linear(hidden_dimension, hidden_dimension * 2, bias=True),
            nn.ELU(),
            nn.Linear(hidden_dimension * 2, hidden_dimension*4, bias=True),
            nn.ELU(),
            nn.Linear(hidden_dimension*4, output_dimension, bias=True)
            # nn.Sigmoid()
        )

    def forward(self, x):
        output = self.main(x)
        return output

class E_Net(nn.Module):
    def __init__(self, input_dimension, output_dimension, hidden_dimension=256):
        super(E_Net, self).__init__()
        self.main = nn.Sequential(
            nn.Linear(input_dimension, hidden_dimension, bias=True),
            nn.ELU(),
            nn.Linear(hidden_dimension, hidden_dimension * 2, bias=True),
            nn.ELU(),
            nn.Linear(hidden_dimension * 2, output_dimension, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        output = self.main(x)
        return output

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        m.weight.data = m.weight.data * 1 / np.sqrt(m.in_features / 2)

        # elif classname.find('BatchNorm') != -1:
        #     m.weight.data.normal_(0, 1)
        #     m.bias.data.fill_(0)
