import torch
import torch.nn as nn
from torch.distributions import normal
from models.torch_utils import GaussianNoise, ScaleShift, Flatten, Reshape

class BNCNN2D(nn.Module):
    def __init__(self, output_units=5, noise=.05, bias=True, chans=None):
        super(BNCNN2D,self).__init__()
        self.name = 'McNiruNet'
        modules = []
        if chans is None:
            chans = [8,8]
        modules.append(nn.Conv2d(40,chans[0],kernel_size=15, bias=bias))
        modules.append(nn.BatchNorm2d(chans[0], eps=1e-3, momentum=.99))
        modules.append(GaussianNoise(std=noise))
        modules.append(nn.ReLU())

        modules.append(nn.Conv2d(chans[0],chans[1],kernel_size=11, bias=bias))
        modules.append(nn.BatchNorm2d(chans[1], eps=1e-3, momentum=.99))
        modules.append(GaussianNoise(std=noise))
        modules.append(nn.ReLU())

        modules.append(Flatten())
        modules.append(nn.Linear(chans[1]*26*26,output_units, bias=bias))
        modules.append(nn.BatchNorm1d(output_units))
        modules.append(nn.Softplus())
        self.sequential = nn.Sequential(*modules)
        
    def forward(self, x):
        return self.sequential(x)

