import torch
import torch.nn as nn
from torch.nn.functional import relu
from models.torch_utils import GaussianNoise, Flatten

class CNN_2layer(nn.Module):
    def __init__(self, output_units, bias=False, noise=0.05):
        super(CNN_2layer,self).__init__()
        self.name = 'CNN_2layer'
        modules = []
        modules.append(nn.Conv2d(40,8,kernel_size=15, bias=bias))
        modules.append(GaussianNoise(std=noise))
        modules.append(nn.ReLU())
        modules.append(Flatten())
        modules.append(nn.Linear(8*36*36, output_units, bias=bias))
        modules.append(nn.Softplus())
        self.sequential = nn.Sequential(*modules)
        
    def forward(self, x):
        return self.sequential(x)
