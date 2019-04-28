import torch
import torch.nn as nn
from torch.distributions import normal
from models.torch_utils import GaussianNoise, ScaleShift, Flatten, Reshape, AbsBatchNorm1d

class AbsBNBNCNN(nn.Module):
    def __init__(self, output_units=5, noise=.05, bias=True, adapt_gauss=False, chans=[8,8]):
        super(AbsBNBNCNN,self).__init__()
        self.name = 'McNiruNet'
        modules = []
        self.chans = chans
        modules.append(nn.Conv2d(40,chans[0],kernel_size=15, bias=bias))
        modules.append(Flatten())
        modules.append(AbsBatchNorm1d(chans[0]*36*36, eps=1e-3, momentum=.99))
        modules.append(GaussianNoise(std=noise, adapt=adapt_gauss))
        modules.append(nn.ReLU())
        modules.append(Reshape((-1,chans[0],36,36)))
        modules.append(nn.Conv2d(chans[0],chans[1],kernel_size=11, bias=bias))
        modules.append(Flatten())
        modules.append(AbsBatchNorm1d(chans[1]*26*26, eps=1e-3, momentum=.99))
        modules.append(GaussianNoise(std=noise, adapt=adapt_gauss))
        modules.append(nn.ReLU())
        modules.append(nn.Linear(chans[1]*26*26,output_units, bias=bias))
        modules.append(AbsBatchNorm1d(output_units))
        modules.append(nn.Softplus())
        self.sequential = nn.Sequential(*modules)
        
    def forward(self, x):
        return self.sequential(x)

