import torch
import torch.nn as nn
from torch.nn.functional import relu
from models.torch_utils import GaussianNoise, Flatten

class CNNfull_2layer(nn.Module):
    def __init__(self, output_units, bias=False, noise=0.05):
        super(CNNfull_2layer,self).__init__()
        self.name = 'CNNfull_2layer'
        modules = []
        modules.append(nn.Conv2d(40,8,kernel_size=15, bias=bias))
        modules.append(GaussianNoise(std=noise))
        modules.append(nn.ReLU())
        modules.append(nn.Conv2d(8,output_units, kernel_size=(36,36), bias=bias))
        modules.append(Flatten())
        modules.append(nn.Softplus())
        self.sequential = nn.Sequential(*modules)
        
    def forward(self, x):
        x = self.sequential(x)
        #assert x.size(1) == output_units, "Not fully-connected conv layer for the last layer."
        return x
