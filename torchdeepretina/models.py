import torch
import torch.nn as nn
from torchdeepretina.torch_utils import *
import numpy as np

def try_kwarg(kwargs, key, default):
    try:
        return kwargs[key]
    except:
        return default

class TDRModel(nn.Module):
    def __init__(self, n_units=5, noise=.05, bias=True, linear_bias=None, chans=[8,8], bn_moment=.01, 
                                 softplus=True, inference_exp=False, img_shape=(40,50,50), ksizes=(15,11), **kwargs):
        super().__init__()
        self.n_units = n_units
        self.chans = chans 
        self.softplus = softplus 
        self.infr_exp = inference_exp 
        self.bias = bias 
        self.img_shape = img_shape 
        self.ksizes = ksizes 
        self.linear_bias = linear_bias 
        self.noise = noise 
        self.bn_moment = bn_moment 
    
    def forward(self, x):
        return x

    def extra_repr(self):
        try:
            return 'n_units={}, noise={}, bias={}, linear_bias={}, chans={}, bn_moment={}, softplus={}, inference_exp={}, img_shape={}, ksizes={}'.format(self.n_units, self.noise, self.bias, self.linear_bias, self.chans, self.bn_moment, self.softplus, self.inference_exp, self.img_shape, self.ksizes)
        except:
            pass


class BNCNN(TDRModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = 'McNiruNet'
        modules = []
        self.shapes = []
        shape = self.img_shape[1:]
        modules.append(nn.Conv2d(self.img_shape[0],self.chans[0],kernel_size=self.ksizes[0], bias=self.bias))
        shape = update_shape(shape, self.ksizes[0])
        self.shapes.append(tuple(shape))
        modules.append(Flatten())
        modules.append(nn.BatchNorm1d(self.chans[0]*shape[0]*shape[1], eps=1e-3, momentum=self.bn_moment))
        modules.append(GaussianNoise(std=self.noise))
        modules.append(nn.ReLU())
        modules.append(Reshape((-1,self.chans[0],*shape)))
        modules.append(nn.Conv2d(self.chans[0],self.chans[1],kernel_size=self.ksizes[1], bias=self.bias))
        shape = update_shape(shape, self.ksizes[1])
        self.shapes.append(tuple(shape))
        modules.append(Flatten())
        modules.append(nn.BatchNorm1d(self.chans[1]*shape[0]*shape[1], eps=1e-3, momentum=self.bn_moment))
        modules.append(GaussianNoise(std=self.noise))
        modules.append(nn.ReLU())
        modules.append(nn.Linear(self.chans[1]*shape[0]*shape[1],self.n_units, bias=self.linear_bias))
        modules.append(nn.BatchNorm1d(self.n_units, eps=1e-3, momentum=self.bn_moment))
        if softplus:
            modules.append(nn.Softplus())
        else:
            modules.append(Exponential(train_off=True))
        self.sequential = nn.Sequential(*modules)
        
    def forward(self, x):
        if not self.training and self.infr_exp:
            return torch.exp(self.sequential(x))
        return self.sequential(x)

class RNNCNN(TDRModel):
    def __init__(self, rnn_chans=[2,2], **kwargs):
        super(**kwargs)
        self.rnns = nn.ModuleList([])
        self.shapes = []
        shape = self.img_shape[1:] # (H, W)

        # Block 1
        modules = []
        self.rnns.append(ConvRNNCell(self.img_shape[0], self.chans[0], rnn_chans[0], kernel_size=self.ksizes[0], bias=self.bias))
        shape = update_shape(shape, self.ksizes[0])
        self.shapes.append(tuple(shape))
        modules.append(Flatten())
        modules.append(nn.BatchNorm1d(self.chans[0]*shape[0]*shape[1], eps=1e-3, momentum=self.bn_moment))
        modules.append(GaussianNoise(std=self.noise))
        modules.append(nn.ReLU())
        modules.append(Reshape((-1,self.chans[0],*shape)))
        self.sequential1 = nn.Sequential(*modules)

        # Block 2
        modules = []
        self.rnns.append(ConvRNNCell(self.chans[0],self.chans[1], rnn_chans[1], kernel_size=self.ksizes[1], bias=self.bias))
        shape = update_shape(shape, self.ksizes[1])
        self.shapes.append(tuple(shape))
        modules.append(Flatten())
        modules.append(nn.BatchNorm1d(self.chans[1]*shape[0]*shape[1], eps=1e-3, momentum=self.bn_moment))
        modules.append(GaussianNoise(std=noise))
        modules.append(nn.ReLU())
        modules.append(nn.Linear(self.chans[1]*shape[0]*shape[1],self.n_units, bias=self.linear_bias))
        modules.append(nn.BatchNorm1d(self.n_units, eps=1e-3, momentum=bn_moment))
        if softplus:
            modules.append(nn.Softplus())
        else:
            modules.append(Exponential(train_off=True))
        self.sequential2 = nn.Sequential(*modules)
        
    def forward(self, x, hs):
        """
        x: torch FloatTensor (B, C, H, W)
            the inputs
        hs: list of torch FloatTensors len==2, (B, RNN_CHAN, H, W), (B, RNN_CHAN1, H1, W1)
            list of the rnn cell states
        """
        fx, h1 = self.rnns[0](x, hs[0])
        fx = self.sequential1(fx)
        fx, h2 = self.rnns[1](x, hs[1])
        fx = self.sequential2(fx)
        if not self.training and self.infr_exp:
            fx = torch.exp(fx)
        return fx, [h1, h2]

    def extra_repr(self):
        try:
            return 'adapt_gauss={}'.format(self.adapt_gauss)
        except:
            pass

class LinearStackedBNCNN(TDRModel):
    def __init__(self, bnorm=True, drop_p=0, **kwargs):
        super().__init__(**kwargs)
        self.name = 'StackedNet'
        self.bnorm = bnorm
        self.drop_p = drop_p
        shape = self.img_shape[1:] # (H, W)
        self.shapes = []

        modules = []
        modules.append(LinearStackedConv2d(self.img_shape[0],self.chans[0],kernel_size=self.ksizes[0], abs_bnorm=self.bnorm, 
                                                                                    bias=self.bias, drop_p=self.drop_p))
        shape = update_shape(shape, self.ksizes[0])
        self.shapes.append(tuple(shape))
        modules.append(Flatten())
        modules.append(AbsBatchNorm1d(self.chans[0]*shape[0]*shape[1], eps=1e-3, momentum=self.bn_moment))
        modules.append(GaussianNoise(std=self.noise))
        modules.append(nn.ReLU())
        modules.append(Reshape((-1,self.chans[0],shape[0], shape[1])))
        modules.append(LinearStackedConv2d(self.chans[0],self.chans[1],kernel_size=self.ksizes[1], abs_bnorm=self.bnorm, 
                                                                                bias=self.bias, drop_p=self.drop_p))
        shape = update_shape(shape, self.ksizes[1])
        self.shapes.append(tuple(shape))
        modules.append(Flatten())
        modules.append(AbsBatchNorm1d(self.chans[1]*shape[0]*shape[1], eps=1e-3, momentum=self.bn_moment))
        modules.append(GaussianNoise(std=self.noise))
        modules.append(nn.ReLU())
        modules.append(nn.Linear(self.chans[1]*shape[0]*shape[1], self.n_units, bias=self.linear_bias))
        modules.append(AbsBatchNorm1d(self.n_units, eps=1e-3, momentum=self.bn_moment))
        if self.softplus:
            modules.append(nn.Softplus())
        else:
            modules.append(Exponential(train_off=True))
        self.sequential = nn.Sequential(*modules)

    def forward(self, x):
        if not self.training and self.infr_exp:
            return torch.exp(self.sequential(x))
        return self.sequential(x)
