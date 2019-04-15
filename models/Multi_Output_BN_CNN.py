import torch
import torch.nn as nn
from torch.distributions import normal
from models.torch_utils import GaussianNoise, ScaleShift, Flatten, Reshape

class MultiOutputBNCNN(nn.Module):
    def __init__(self, output_units=[5], noise=.05, bias=True):
        super(MultiOutputBNCNN,self).__init__()
        self.name = 'McNiruNet'

        # Create base of model
        modules = []
        modules.append(nn.Conv2d(40,8,kernel_size=15, bias=bias))
        modules.append(Flatten())
        modules.append(nn.BatchNorm1d(8*36*36, eps=1e-3, momentum=.99))
        modules.append(GaussianNoise(std=noise))
        modules.append(nn.ReLU())
        modules.append(Reshape((-1,8,36,36)))
        modules.append(nn.Conv2d(8,8,kernel_size=11, bias=bias))
        modules.append(Flatten())
        modules.append(nn.BatchNorm1d(8*26*26, eps=1e-3, momentum=.99))
        modules.append(GaussianNoise(std=noise))
        modules.append(nn.ReLU())
        self.features = nn.Sequential(*modules)

        # Create output layers
        self.output_layers = nn.ModuleList([])
        for n_units in range(output_units):
            modules = []
            modules.append(nn.Linear(8*26*26, n_units, bias=bias))
            modules.append(nn.BatchNorm1d(n_units))
            modules.append(nn.Softplus())
            self.output_layers.append(nn.Sequential(*modules))
        
    def forward(self, x, output_idx):
        feats = self.features(x)
        return self.output_layers[output_idx](feats)

