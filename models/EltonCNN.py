import torch
import torch.nn as nn
import sys
sys.path.append('../')
from models.torch_utils import GaussianNoise

class EltonCNN(nn.Module):
    def __init__(self, n_output_units=5, history = 40, noise_std=0.05):
        super(EltonCNN,self).__init__()
        self.name = 'McNiruNet'
        self.conv1 = nn.Conv2d(history,8,kernel_size=6)
        #self.batch1 = nn.BatchNorm1d(8*36*36) # keras version uses bnorm with eps=1e-3, momentum=.99
        self.gaussian1 = GaussianNoise(std=noise_std)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(8,8,kernel_size=6)
        #self.batch2 = nn.BatchNorm1d(8*26*26) # keras version uses bnorm with eps=1e-3, momentum=.99
        self.gaussian2 = GaussianNoise(std=noise_std)
        self.relu2 = nn.ReLU()
        self.linear = nn.Linear(8*10*10,n_output_units, bias=False)
        #self.batch3 = nn.BatchNorm1d(n_output_units) # keras version uses bnorm with eps=1e-3, momentum=.99
        self.losses = []
        self.actgrad1=[]
        self.actgrad2=[]
        
    def forward(self, x):
        x = self.conv1(x)
        
       # x = self.batch1(x.view(x.size(0), -1))
        x = self.gaussian1(x.view(-1, 8, 15, 15))
        x = self.relu1(x)
        x = self.conv2(x)
       # x = self.batch2(x.view(x.size(0), -1))
        x = self.gaussian2(x.view(-1, 8, 10, 10))
        x = self.relu2(x)
        x = self.linear(x.view(-1, 8*10*10))
        #x = self.batch3(x)
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

