import torch
import torch.nn as nn
from models.torch_utils import GaussianNoise, ScaleShift, AbsConv2d, AbsLinear, DaleActivations

class DalesSSCNN(nn.Module):
    def __init__(self, scale=True, shift=True, bias=True, gauss_std=0.05, neg_p=.5):
        super(DalesSSCNN,self).__init__()
        self.name = 'DaleNet'
        self.conv1 = AbsConv2d(40,8,kernel_size=15, bias=bias)
        self.relu1 = nn.ReLU()
        self.ss1 = ScaleShift((8,36,36), shift=shift, scale=scale)
        self.dale1 = DaleActivations(8, neg_p)
        self.conv2 = AbsConv2d(8,8,kernel_size=11, bias=bias)
        self.relu2 = nn.ReLU()
        self.ss2 = ScaleShift((8,26,26), shift=shift, scale=scale)
        self.dale2 = DaleActivations(8, neg_p)
        self.linear = AbsLinear(8*26*26,5, bias=bias)
        self.ss3 = ScaleShift(5, scale=scale, shift=shift)
        self.gaussian = GaussianNoise(std=gauss_std)
        self.softplus = nn.Softplus()
        self.losses = []
        self.actgrad1=[]
        self.actgrad2=[]
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.ss1(x)
        x = self.gaussian(x)
        x = self.relu1(x)
        x = self.dale1(x)
        x = self.conv2(x)
        x = self.ss2(x)
        x = self.gaussian(x)
        x = self.relu2(x)
        x = self.dale2(x)
        x = self.linear(x.view(x.size(0), -1))
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

