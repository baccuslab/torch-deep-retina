import torch
import torch.nn as nn
from models.torch_utils import GaussianNoise, ScaleShift

class SSCNN(nn.Module):
    def __init__(self, scale=True, shift=False, bias=True, noise=0.05):
        super(SSCNN,self).__init__()
        self.name = 'McNiruNet'
        modules = []
        modules.append(nn.Conv2d(40,8,kernel_size=15, bias=bias))
        modules.append(ScaleShift((8,36,36)))
        modules.append(GaussianNoise(std=noise))
        modules.append(nn.ReLU())
        modules.append(nn.Conv2d(8,8,kernel_size=11, bias=bias))
        modules.append(GaussianNoise(std=noise))
        modules.append(ScaleShift((8,26,26)))
        modules.append(nn.ReLU())
        modules.append(nn.Linear(8*26*26,output_units, bias=bias))
        modules.append(ScaleShift(output_units))
        modules.append(nn.Softplus())
        self.sequential = nn.Sequential(*modules)
        
    def forward(self, x):
        return self.sequential(x)

