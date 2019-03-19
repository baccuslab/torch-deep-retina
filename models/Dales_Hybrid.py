import torch
import torch.nn as nn
from models.torch_utils import GaussianNoise, ScaleShift, AbsConv2d, AbsLinear, DaleActivations
import numpy as np

class DalesHybrid(nn.Module):
    def __init__(self, output_units=5, bias=True, noise=0.1, neg_p=0.5):
        super(DalesHybrid,self).__init__()
        self.name = 'DaleNet'
        self.conv1 = nn.Conv2d(40,8,kernel_size=15, bias=bias)
        param = next(self.conv1.parameters())
        print("Before:", param.norm(2))
        self.diminish_weight_magnitude(self.conv1.parameters())
        print("After:", param.norm(2))
        self.batch1 = nn.BatchNorm1d(8*36*36, eps=1e-3, momentum=.99)
        self.gaussian1 = GaussianNoise(std=noise)
        self.relu1 = nn.ReLU()
        self.dale1 = DaleActivations(8, neg_p)
        self.conv2 = AbsConv2d(8,8,kernel_size=11, bias=bias)
        self.diminish_weight_magnitude(self.conv2.parameters())
        self.gaussian2 = GaussianNoise(std=noise)
        self.batch2 = nn.BatchNorm1d(8*26*26, eps=1e-3, momentum=.99)
        self.relu2 = nn.ReLU()
        self.dale2 = DaleActivations(8, neg_p)
        self.linear = AbsLinear(8*26*26,output_units, bias=bias)
        self.diminish_weight_magnitude(self.linear.parameters())
        self.ss = ScaleShift(output_units)
        self.softplus = nn.Softplus()
        self.losses = []
        self.actgrad1=[]
        self.actgrad2=[]
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.batch1(x.view(x.size(0), -1))
        x = self.gaussian1(x)
        x = self.relu1(x)
        x = x.view(-1, 8, 36, 36)
        x = self.dale1(x)
        x = self.conv2(x)
        x = self.batch2(x.view(x.size(0), -1))
        x = self.gaussian2(x)
        x = self.relu2(x)
        x = self.dale2(x.view(-1, 8, 26, 26))
        x = self.linear(x.view(x.size(0), -1))
        x = self.ss(x)
        x = self.softplus(x)
        return x
    
    def diminish_weight_magnitude(self, params):
        for param in params:
            divisor = float(np.prod(param.data.shape))/2
            if param.data is not None and divisor >= 0:
                param.data = param.data/divisor

    def record_grad1(self,grad):
        self.actgrad1=grad.clone()
        
    def record_grad2(self,grad):
        self.actgrad2=grad.clone()

