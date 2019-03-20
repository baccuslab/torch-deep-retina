import torch
import torch.nn as nn
from models.torch_utils import GaussianNoise, ScaleShift, AbsConv2d, AbsLinear, DaleActivations, Flatten, Reshape
import numpy as np

class DalesSSCNN(nn.Module):
    def __init__(self, output_units=5, bias=True, noise=0.1, neg_p=0.5, scale=True, shift=True):
        super(DalesSSCNN,self).__init__()
        self.name = 'DaleNet'
        modules = []
        modules.append(AbsConv2d(40,8,kernel_size=15, bias=bias))
        self.diminish_weight_magnitude(modules[-1].parameters())
        modules.append(ScaleShift((8,36,36), scale=scale, shift=shift))
        modules.append(GaussianNoise(std=noise))
        modules.append(nn.ReLU())
        modules.append(DaleActivations(8, neg_p))
        modules.append(AbsConv2d(8,8,kernel_size=11, bias=bias))
        self.diminish_weight_magnitude(modules[-1].parameters())
        modules.append(GaussianNoise(std=noise))
        modules.append(ScaleShift((8,26,26), scale=scale, shift=shift))
        modules.append(nn.ReLU())
        modules.append(DaleActivations(8, neg_p))
        modules.append(Flatten())
        modules.append(AbsLinear(8*26*26,output_units, bias=bias))
        self.diminish_weight_magnitude(modules[-1].parameters())
        modules.append(ScaleShift(output_units, scale=scale, shift=shift))
        modules.append(nn.Softplus())
        self.sequential = nn.Sequential(*modules)
        
    def forward(self, x):
        return self.sequential(x)
    
    def diminish_weight_magnitude(self, params):
        for param in params:
            divisor = float(np.prod(param.data.shape))/2
            if param.data is not None and divisor >= 0:
                param.data = param.data/divisor

