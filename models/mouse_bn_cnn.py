import torch
import torch.nn as nn
from torch.distributions import normal

class BNCNN(nn.Module):
    def __init__(self):
        super(BNCNN,self).__init__()
        self.name = 'McNiruNet'
        self.conv1 = nn.Conv2d(40,8,kernel_size=15)
        self.batch1 = nn.BatchNorm1d(8*36*36, eps=1e-3, momentum=.99)
        self.conv2 = nn.Conv2d(8,8,kernel_size=11)
        self.batch2 = nn.BatchNorm1d(8*26*26, eps=1e-3, momentum=.99)
        self.linear = nn.Linear(8*26*26,2, bias=False)
        self.batch3 = nn.BatchNorm1d(2, eps=1e-3, momentum=.99)
        self.losses = []
        self.actgrad1=[]
        self.actgrad2=[]
        
    def gaussian(self, x, sigma):
        noise = normal.Normal(torch.zeros(x.size()), sigma*torch.ones(x.size()))
        if x.is_cuda:
            return x + noise.sample().cuda()
        return x + noise.sample()
        
    def forward(self, x):
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
    
    def record_grad1(self,grad):
        self.actgrad1=grad.clone()
        
    def record_grad2(self,grad):
        self.actgrad2=grad.clone()

    def inspect(self, x):
        model_dict = {}
        model_dict['stimulus'] = x
        x = self.conv1(x);
        model_dict['conv2d_1'] = x
        x = x.view(x.size(0), -1)
        model_dict['flatten_1'] = x
        x = self.batch1(x)
        model_dict['batch_normalization_1'] = x
        x = self.gaussian(x.view(-1, 8, 36, 36), 0.05)
        model_dict['gaussian_1'] = x
        x = nn.functional.relu(x)
        model_dict['activation_1'] = x
        model_dict['activation_1'].register_hook(self.record_grad1)
        x = self.conv2(x);
        model_dict['conv2d_2'] = x
        x = x.view(x.size(0), -1)
        model_dict['flatten_2'] = x
        x = self.batch2(x)
        model_dict['batch_normalization_2'] = x
        x = self.gaussian(x.view(-1, 8, 26, 26), 0.05)
        model_dict['gaussian_2'] = x
        x = nn.functional.relu(x)
        model_dict['activation_2'] = x
        model_dict['activation_2'].register_hook(self.record_grad2)
        x = self.linear(x.view(-1, 8*26*26))
        model_dict['dense'] = x
        x = self.batch3(x)
        model_dict['batch_normalization_3'] = x
        x = nn.functional.softplus(x)
        model_dict['activation_3'] = x
        return model_dict

