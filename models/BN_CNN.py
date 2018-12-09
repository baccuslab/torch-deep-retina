import torch
import torch.nn as nn
from torch.distributions import normal
 class BN_CNN(nn.Module):
    def __init__(self):
        super(BN_CNN,self).__init__()
        self.name = 'BN_CNN'
        self.conv1 = nn.Conv2d(40,8,kernel_size=15)
        self.batch1 = nn.BatchNorm1d(8*36*36)
        self.conv2 = nn.Conv2d(8,8,kernel_size=11)
        self.batch2 = nn.BatchNorm1d(8*26*26)
        self.linear = nn.Linear(8*26*26,5, bias=False)
        self.batch3 = nn.BatchNorm1d(5)
        self.losses = []
        
    def gaussian(self, x, sigma):
         noise = normal.Normal(torch.zeros(x.size()), sigma*torch.ones(x.size()))
        return x + noise.sample().cuda()
        
    def forward(self,x):
        x = self.conv1(x)
        x = self.batch1(x.view(x.size(0), -1))
        x = nn.functional.relu(self.gaussian(x.view(-1, 8, 36, 36), 0.05))
        x = self.conv2(x)
        x = self.batch2(x.view(x.size(0), -1))
        x = nn.functional.relu(self.gaussian(x.view(-1, 8, 26, 26), 0.05))
        x = self.linear(x.view(-1, 8*26*26))
        x = self.batch3(x)
        x = nn.functional.softplus(x)
        return x