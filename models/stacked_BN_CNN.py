import torch
import torch.nn as nn
from torch.distributions import normal
from models.torch_utils import GaussianNoise, ScaleShift, Flatten, Reshape, StackedConv2d

class StackedBNCNN(nn.Module):
    def __init__(self, output_units=5, noise=.05, bias=True, adapt_gauss=False):
        super(StackedBNCNN,self).__init__()
        self.name = 'StackedNet'
        module_list = []
        module_list.append(StackedConv2d(40,8,kernel_size=15))
        module_list.append(Flatten())
        module_list.append(nn.BatchNorm1d(8*36*36, eps=1e-3, momentum=.99))
        module_list.append(GaussianNoise(std=noise, adapt=adapt_gauss))
        module_list.append(nn.ReLU())
        module_list.append(Reshape((-1,8,36,36)))
        module_list.append(StackedConv2d(8,8,kernel_size=11))
        module_list.append(Flatten())
        module_list.append(nn.BatchNorm1d(8*26*26, eps=1e-3, momentum=.99))
        module_list.append(GaussianNoise(std=noise, adapt=adapt_gauss))
        module_list.append(nn.ReLU())
        module_list.append(nn.Linear(8*26*26,output_units, bias=False))
        module_list.append(nn.BatchNorm1d(output_units, eps=1e-3, momentum=.99))
        module_list.append(nn.Softplus())
        self.sequential = nn.Sequential(*module_list)

    def forward(self, x):
        return self.sequential(x)
    
