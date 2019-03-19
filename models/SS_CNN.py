import torch
import torch.nn as nn
from models.torch_utils import GaussianNoise, ScaleShift

class SSCNN(nn.Module):
    def __init__(self, scale=True, shift=False, bias=True, noise=0.05):
        super(SSCNN,self).__init__()
        self.name = 'McNiruNet'
        self.conv1 = nn.Conv2d(40,8,kernel_size=15, bias=bias)
        self.ss1 = ScaleShift((8,36,36), shift=shift, scale=scale)
        self.conv2 = nn.Conv2d(8,8,kernel_size=11, bias=bias)
        self.ss2 = ScaleShift((8,26,26), shift=shift, scale=scale)
        self.linear = nn.Linear(8*26*26,5, bias=bias)
        self.ss3 = ScaleShift(5, shift=shift, scale=scale)
        self.gaussian = GaussianNoise(std=noise)
        self.relu = nn.ReLU()
        self.softplus = nn.Softplus()
        self.losses = []
        self.actgrad1=[]
        self.actgrad2=[]
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.ss1(x)
        x = self.relu(self.gaussian(x))
        x = self.conv2(x)
        x = self.ss2(x)
        x = self.relu(self.gaussian(x))
        x = self.linear(x.view(-1, 8*26*26))
        x = self.ss3(x)
        x = self.softplus(x)
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


