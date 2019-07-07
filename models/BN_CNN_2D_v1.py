import torch
import torch.nn as nn
from torch.distributions import normal
from models.torch_utils import GaussianNoise, ScaleShift, Flatten, Reshape

# Juyoung's BNCNN 2D model version 1 (0705 2019)
# No bias for conv layer, bias for FC layer
# No batchnorm layer after the final layer
# Back to Defalt params fo bachnorm2d layer
# Different from original McNiruNet

class BNCNN2D_v1(nn.Module):
    def __init__(self, output_units=5, noise=.1, bias=False, chans=None):
        super(BNCNN2D_v1,self).__init__()
        self.name = 'JuNet'
        modules = []
        if chans is None:
            chans = [8,8]
        modules.append(nn.Conv2d(40,chans[0],kernel_size=15, bias=bias))
        modules.append(nn.BatchNorm2d(chans[0], eps=1e-5, momentum=.1))
        modules.append(GaussianNoise(std=noise))
        modules.append(nn.ReLU())

        modules.append(nn.Conv2d(chans[0],chans[1],kernel_size=11, bias=bias))
        modules.append(nn.BatchNorm2d(chans[1], eps=1e-5, momentum=.1))
        modules.append(GaussianNoise(std=noise))
        modules.append(nn.ReLU())

        modules.append(Flatten())
        modules.append(nn.Linear(chans[1]*26*26,output_units, bias=True))
        modules.append(nn.Softplus())
        self.sequential = nn.Sequential(*modules)
        
    def forward(self, x):
        return self.sequential(x)

