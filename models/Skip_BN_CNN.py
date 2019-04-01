import torch
import torch.nn as nn
from torch.distributions import normal
from models.torch_utils import GaussianNoise, ScaleShift, Flatten, Reshape, SkipConnection1

class SkipBNCNN(nn.Module):
    def __init__(self, output_units=5, noise=.05, bias=True):
        super(SkipBNCNN,self).__init__()
        self.name = 'SkipNet'
        modules = []
        #in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True
        modules.append(SkipConnection1(40,8,15,bias=bias))
        modules.append(nn.Conv2d(48,8,kernel_size=15, bias=bias))
        modules.append(Flatten())
        modules.append(nn.BatchNorm1d(8*36*36, eps=1e-3, momentum=.99))
        modules.append(GaussianNoise(std=noise))
        modules.append(nn.ReLU())
        modules.append(Reshape((-1,8,36,36)))
        modules.append(nn.Conv2d(8,8,kernel_size=11, bias=bias))
        modules.append(Flatten())
        modules.append(nn.BatchNorm1d(8*26*26, eps=1e-3, momentum=.99))
        modules.append(GaussianNoise(std=noise))
        modules.append(nn.ReLU())
        modules.append(nn.Linear(8*26*26,output_units, bias=bias))
        modules.append(nn.BatchNorm1d(output_units))
        modules.append(nn.Softplus())
        self.sequential = nn.Sequential(*modules)
        
    def forward(self, x):
        return self.sequential(x)

