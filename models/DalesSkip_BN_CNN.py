import torch
import torch.nn as nn
from torch.distributions import normal
from models.torch_utils import AbsConv2d, DaleActivations, GaussianNoise, ScaleShift, Flatten, Reshape, DalesSkipConnection, diminish_weight_magnitude, AbsLinear

class DalesSkipBNCNN(nn.Module):
    def __init__(self, output_units=5, noise=.05, bias=True, x_shape=(40,50,50), skip_depth=2, neg_p=.5):
        super(DalesSkipBNCNN,self).__init__()
        self.name = 'SkipNet'
        modules = []
        #in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True
        modules.append(DalesSkipConnection(40,skip_depth,15,x_shape=x_shape,bias=bias, noise=noise, neg_p=1))
        modules.append(AbsConv2d(40+skip_depth,8,kernel_size=15, bias=bias))
        diminish_weight_magnitude(modules[-1].parameters())
        modules.append(Flatten())
        modules.append(nn.BatchNorm1d(8*36*36, eps=1e-3))
        modules.append(GaussianNoise(std=noise))
        modules.append(nn.ReLU())
        modules.append(Reshape((-1,8,36,36)))
        modules.append(DaleActivations(8, neg_p))
        modules.append(AbsConv2d(8,8,kernel_size=11, bias=bias))
        diminish_weight_magnitude(modules[-1].parameters())
        modules.append(Flatten())
        modules.append(nn.BatchNorm1d(8*26*26, eps=1e-3))
        modules.append(GaussianNoise(std=noise))
        modules.append(nn.ReLU())
        modules.append(Reshape((-1,8,26,26)))
        modules.append(DaleActivations(8, neg_p))
        modules.append(Flatten())
        modules.append(AbsLinear(8*26*26,output_units, bias=bias))
        diminish_weight_magnitude(modules[-1].parameters())
        modules.append(nn.BatchNorm1d(output_units))
        modules.append(nn.Softplus())
        self.sequential = nn.Sequential(*modules)
        
    def forward(self, x):
        return self.sequential(x)

