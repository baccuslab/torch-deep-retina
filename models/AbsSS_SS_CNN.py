import torch
import torch.nn as nn
from models.torch_utils import GaussianNoise, AbsScaleShift, Flatten

class AbsSSSSCNN(nn.Module):
    def __init__(self, output_units, scale=True, shift=False, bias=True, noise=0.05, adapt_gauss=False):
        super(AbsSSSSCNN,self).__init__()
        self.name = 'McNiruNet'
        modules = []
        modules.append(nn.Conv2d(40,8,kernel_size=15, bias=bias))
        modules.append(AbsScaleShift((8,36,36)))
        modules.append(GaussianNoise(std=noise, adapt=adapt_gauss))
        modules.append(nn.ReLU())
        modules.append(nn.Conv2d(8,8,kernel_size=11, bias=bias))
        modules.append(AbsScaleShift((8,26,26)))
        modules.append(GaussianNoise(std=noise, adapt=adapt_gauss))
        modules.append(nn.ReLU())
        modules.append(Flatten())
        modules.append(nn.Linear(8*26*26,output_units, bias=bias))
        modules.append(AbsScaleShift(output_units))
        modules.append(nn.Softplus())
        self.sequential = nn.Sequential(*modules)
        
    def forward(self, x):
        return self.sequential(x)

