import torch
import torch.nn as nn
from torch.distributions import normal
from models.torch_utils import Flatten, Reshape

class PracticalBNCNN(nn.Module):
    def __init__(self, output_units=5, noise=.1, bias=True):
        super(PracticalBNCNN,self).__init__()
        #self.name = 'McNiruNet'
        #self.conv1 = nn.Conv2d(40, 8, kernel_size=(15, 15), stride=(1, 1))
        #self.relu1 = nn.ReLU()
        #self.batch1=nn.BatchNorm1d(10368, eps=0.001, momentum=0.99, affine=True, track_running_stats=True)
        #self.dropout1=nn.Dropout(p=0.15)
        #self.conv2=nn.Conv2d(8, 8, kernel_size=(11, 11), stride=(1, 1))
        #self.relu2=nn.ReLU()
        #self.batch2=nn.BatchNorm1d(5408, eps=0.001, momentum=0.99, affine=True, track_running_stats=True)
        #self.dropout2=nn.Dropout(p=0.3)
        #self.linear=nn.Linear(in_features=5408, out_features=5, bias=False)
        #self.batch3=nn.BatchNorm1d(5, eps=0.001, momentum=0.99, affine=True, track_running_stats=True)
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
    #def forward(self, x):
    #    x = self.conv1(x)        
    #    x = self.relu1(x)
    #    shape = x.shape
    #    x = x.view(x.shape[0], -1)
    #    x = self.batch1(x)
    #    x = self.dropout1(x)
    #    x = x.view(shape)
    #    x = self.conv2(x)
    #    x = self.relu2(x)
    #    x = x.view(x.shape[0], -1)
    #    x = self.batch2(x)
    #    x = self.dropout2(x)
    #    x = self.linear(x)
    #    x = self.batch3(x)
    #    return nn.functional.softplus(x)
