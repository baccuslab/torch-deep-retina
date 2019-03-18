import torch
import torch.nn as nn
from torch.distributions import normal

class BNCNN(nn.Module):
    def __init__(self, n_output_units=5, noise_std=.1):
        super(BNCNN,self).__init__()
        self.name = 'McNiruNet'
        self.conv1 = nn.Conv2d(40,8,kernel_size=15)
        self.batch1 = nn.BatchNorm1d(8*36*36, eps=1e-3, momentum=.99)
        self.relu1 = nn.ReLU()
        self.gaussian1 = GaussianNoise(noise_std=noise_std)
        self.conv2 = nn.Conv2d(8,8,kernel_size=11)
        self.batch2 = nn.BatchNorm1d(8*26*26, eps=1e-3, momentum=.99)
        self.relu2 = nn.ReLU()
        self.gaussian2 = GaussianNoise(noise_std=noise_std)
        self.linear = nn.Linear(8*26*26,n_output_units, bias=False)
        self.batch3 = nn.BatchNorm1d(n_output_units, eps=1e-3, momentum=.99)
        self.losses = []
        self.actgrad1=[]
        self.actgrad2=[]
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.batch1(x.view(x.size(0), -1))
        x = self.relu1(self.gaussian1(x.view(-1, 8, 36, 36)))
        x = self.conv2(x)
        x = self.batch2(x.view(x.size(0), -1))
        x = self.relu2(self.gaussian2(x.view(-1, 8, 26, 26)))
        x = self.linear(x.view(-1, 8*26*26))
        x = self.batch3(x)
        x = nn.functional.softplus(x)
        return x
    
    def record_grad1(self,grad):
        self.actgrad1=grad.clone()
        
    def record_grad2(self,grad):
        self.actgrad2=grad.clone()

