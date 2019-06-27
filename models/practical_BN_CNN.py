import torch
import torch.nn as nn
from torch.distributions import normal
from models.torch_utils import Flatten, Reshape

class PracticalBNCNN(nn.Module):
    def __init__(self, output_units=5, noise=.1, bias=True):
        super(PracticalBNCNN,self).__init__()
        modules = []
        modules.append(nn.Conv2d(40,8,kernel_size=15, bias=bias))
        modules.append(nn.Dropout(p=noise/2))
        modules.append(nn.ReLU())
        modules.append(Flatten())
        modules.append(nn.BatchNorm1d(8*36*36, eps=1e-3))
        modules.append(Reshape((-1,8,36,36)))
        modules.append(nn.Conv2d(8,8,kernel_size=11, bias=bias))
        modules.append(nn.Dropout(p=noise))
        modules.append(nn.ReLU())
        modules.append(Flatten())
        modules.append(nn.BatchNorm1d(8*26*26, eps=1e-3))
        modules.append(nn.Linear(8*26*26,output_units, bias=bias))
        modules.append(nn.BatchNorm1d(output_units))
        modules.append(nn.Softplus())
        self.sequential = nn.Sequential(*modules)
        
    def forward(self, x):
        return self.sequential(x)

#class PracticalBNCNN(nn.Module):
#    def __init__(self, output_units=5, noise=.1, bias=True):
#        super(PracticalBNCNN,self).__init__()
#        modules = []
#        n_chan0 = 8
#        n_chan1 = 8
#        modules.append(nn.Conv2d(40,n_chan0,kernel_size=15, bias=bias))
#        modules.append(nn.Dropout(p=noise/2))
#        modules.append(nn.ReLU())
#        modules.append(nn.BatchNorm2d(n_chan0))
#        modules.append(nn.Conv2d(n_chan0,n_chan1,kernel_size=11, bias=bias))
#        modules.append(nn.Dropout(p=noise))
#        modules.append(nn.ReLU())
#        modules.append(nn.BatchNorm2d(n_chan1))
#        modules.append(Flatten())
#        modules.append(nn.Linear(n_chan1*26*26,output_units, bias=bias))
#        modules.append(nn.BatchNorm1d(output_units))
#        modules.append(nn.Softplus())
#        self.sequential = nn.Sequential(*modules)
#        
#    def forward(self, x):
#        return self.sequential(x)