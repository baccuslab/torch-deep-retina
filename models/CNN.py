import torch
import torch.nn as nn
from torch.nn.functional import relu
from models.torch_utils import GaussianNoise, Flatten

class CNN(nn.Module):
    def __init__(self, output_units, bias=False, noise=0.05):
        super(CNN,self).__init__()
        self.name = 'McNiruNet'
        modules = []
        modules.append(nn.Conv2d(40,8,kernel_size=15, bias=bias))
        modules.append(GaussianNoise(std=noise))
        modules.append(nn.ReLU())
        modules.append(nn.Conv2d(8,8,kernel_size=11, bias=bias))
        modules.append(GaussianNoise(std=noise))
        modules.append(nn.ReLU())
        modules.append(Flatten())
        modules.append(nn.Linear(8*26*26,output_units, bias=bias))
        modules.append(nn.Softplus())
        self.sequential = nn.Sequential(*modules)
        
    def forward(self, x):
        return self.sequential(x)
