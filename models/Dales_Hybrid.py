import torch
import torch.nn as nn
<<<<<<< HEAD
from models.torch_utils import GaussianNoise, ScaleShift, AbsConv2d, AbsLinear, DaleActivations, Flatten, Reshape
=======
from models.torch_utils import GaussianNoise, ScaleShift, AbsConv2d, AbsLinear, DaleActivations
>>>>>>> 9615d40cb6f684e0da52181988b9dad935127b52
import numpy as np

class DalesHybrid(nn.Module):
    def __init__(self, output_units=5, bias=True, noise=0.1, neg_p=0.5):
        super(DalesHybrid,self).__init__()
        self.name = 'DaleNet'
        modules = []
        modules.append(nn.Conv2d(40,8,kernel_size=15, bias=bias))
        self.diminish_weight_magnitude(modules[-1].parameters())
        modules.append(Flatten())
        modules.append(nn.BatchNorm1d(8*36*36, eps=1e-3, momentum=.99))
        modules.append(GaussianNoise(std=noise))
        modules.append(nn.ReLU())
        modules.append(Reshape((-1,8,36,36)))
        modules.append(DaleActivations(8, neg_p))
        modules.append(AbsConv2d(8,8,kernel_size=11, bias=bias))
        self.diminish_weight_magnitude(modules[-1].parameters())
        modules.append(Flatten())
        modules.append(GaussianNoise(std=noise))
        modules.append(nn.BatchNorm1d(8*26*26, eps=1e-3, momentum=.99))
        modules.append(nn.ReLU())
        modules.append(Reshape((-1,8,26,26)))
        modules.append(DaleActivations(8, neg_p))
        modules.append(Flatten())
        modules.append(AbsLinear(8*26*26,output_units, bias=bias))
        self.diminish_weight_magnitude(modules[-1].parameters())
        modules.append(ScaleShift(output_units))
        modules.append(nn.Softplus())
        self.sequential = nn.Sequential(*modules)
        
    def forward(self, x):
        return self.sequential(x)
    
    def diminish_weight_magnitude(self, params):
        for param in params:
            divisor = float(np.prod(param.data.shape))/2
            if param.data is not None and divisor >= 0:
                param.data = param.data/divisor

