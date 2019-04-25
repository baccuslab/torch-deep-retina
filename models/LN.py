import torch
import torch.nn as nn
import sys
sys.path.append('../')
from models.torch_utils import GaussianNoise

class LN(nn.Module):
    def __init__(self,output_units=1,noise_std=0.05):
        super(LN,self).__init__()
        self.name = 'LN'
        self.linear = nn.Linear(40*50*50,output_units)
        self.gaussian1 = GaussianNoise(std=noise_std)

    def forward(self,x):
        x = x.view(-1,40*50*50)
       
        x = nn.functional.softplus(self.linear(x))
        return x