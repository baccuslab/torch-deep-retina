import torch
import torch.nn as nn
from torchdeepretina.torch_utils import *
import numpy as np

# Use for bncnnbnormmomentum folder
class BNCNN(nn.Module):
    def __init__(self, n_units=5, noise=.05, bias=True, linear_bias=None, adapt_gauss=False, chans=[8,8], bnorm_momentum=.01, softplus=True, inference_exp=False):
        super(BNCNN,self).__init__()
        self.name = 'McNiruNet'
        self.chans = chans
        self.softplus = softplus
        self.infr_exp = inference_exp
        self.n_units = n_units
        if linear_bias is None:
            linear_bias = bias
        modules = []
        modules.append(nn.Conv2d(40,chans[0],kernel_size=15, bias=bias))
        modules.append(Flatten())
        modules.append(nn.BatchNorm1d(chans[0]*36*36, eps=1e-3, momentum=bnorm_momentum))
        modules.append(GaussianNoise(std=noise, adapt=adapt_gauss))
        modules.append(nn.ReLU())
        modules.append(Reshape((-1,chans[0],36,36)))
        modules.append(nn.Conv2d(chans[0],chans[1],kernel_size=11, bias=bias))
        modules.append(Flatten())
        modules.append(nn.BatchNorm1d(chans[1]*26*26, eps=1e-3, momentum=bnorm_momentum))
        modules.append(GaussianNoise(std=noise, adapt=adapt_gauss))
        modules.append(nn.ReLU())
        modules.append(nn.Linear(chans[1]*26*26,n_units, bias=linear_bias))
        modules.append(nn.BatchNorm1d(n_units, eps=1e-3, momentum=bnorm_momentum))
        if softplus:
            modules.append(nn.Softplus())
        else:
            modules.append(Exponential(train_off=True))
        self.sequential = nn.Sequential(*modules)
        
    def forward(self, x):
        if not self.training and self.infr_exp:
            return torch.exp(self.sequential(x))
        return self.sequential(x)

    def extra_repr(self):
        try:
            return 'adapt_gauss={}'.format(self.adapt_gauss)
        except:
            pass

class LinearDecoupBNCNN(nn.Module):
    def __init__(self, n_units=5, noise=.05, bias=True, linear_bias=None, adapt_gauss=False, chans=[8,8], bnorm_momentum=.01, softplus=True, inference_exp=False):
        super(LinearDecoupBNCNN,self).__init__()
        self.name = 'McNiruNet'
        self.n_units = n_units
        self.chans = chans
        self.softplus = softplus
        self.infr_exp = inference_exp
        if linear_bias is None:
            linear_bias = bias
        modules = []
        modules.append(nn.Conv2d(40,chans[0],kernel_size=15, bias=bias))
        modules.append(Flatten())
        modules.append(nn.BatchNorm1d(chans[0]*36*36, eps=1e-3, momentum=bnorm_momentum))
        modules.append(GaussianNoise(std=noise, adapt=adapt_gauss))
        modules.append(nn.ReLU())
        modules.append(Reshape((-1,chans[0],36,36)))
        modules.append(nn.Conv2d(chans[0],chans[1],kernel_size=11, bias=bias))
        modules.append(Flatten())
        modules.append(nn.BatchNorm1d(chans[1]*26*26, eps=1e-3, momentum=bnorm_momentum))
        modules.append(GaussianNoise(std=noise, adapt=adapt_gauss))
        modules.append(nn.ReLU())
        modules.append(DecoupledLinear(chans[1]*26*26, n_units, bias=linear_bias))
        modules.append(nn.BatchNorm1d(n_units, eps=1e-3, momentum=bnorm_momentum))
        if softplus:
            modules.append(nn.Softplus())
        else:
            modules.append(Exponential(train_off=True))
        self.sequential = nn.Sequential(*modules)
        
    def forward(self, x):
        if not self.training and self.infr_exp:
            return torch.exp(self.sequential(x))
        return self.sequential(x)

    def extra_repr(self):
        try:
            return 'adapt_gauss={}'.format(self.adapt_gauss)
        except:
            pass

class NoFinalBNBNCNN(nn.Module):
    def __init__(self, n_units=5, noise=.05, bias=True, linear_bias=None, adapt_gauss=False, chans=[8,8], bnorm_momentum=.01, softplus=True, inference_exp=False):
        super(NoBNSoftplusBNCNN,self).__init__()
        self.name = 'McNiruNet'
        self.n_units = n_units
        self.chans = chans
        self.infr_exp = inference_exp
        if linear_bias is None:
            linear_bias = bias
        modules = []
        modules.append(nn.Conv2d(40,chans[0],kernel_size=15, bias=bias))
        modules.append(Flatten())
        modules.append(nn.BatchNorm1d(chans[0]*36*36, eps=1e-3, momentum=bnorm_momentum))
        modules.append(GaussianNoise(std=noise, adapt=adapt_gauss))
        modules.append(nn.ReLU())
        modules.append(Reshape((-1,chans[0],36,36)))
        modules.append(nn.Conv2d(chans[0],chans[1],kernel_size=11, bias=bias))
        modules.append(Flatten())
        modules.append(nn.BatchNorm1d(chans[1]*26*26, eps=1e-3, momentum=bnorm_momentum))
        modules.append(GaussianNoise(std=noise, adapt=adapt_gauss))
        modules.append(nn.ReLU())
        modules.append(nn.Linear(chans[1]*26*26,n_units, bias=linear_bias))
        if softplus:
            modules.append(nn.Softplus())
        else:
            modules.append(Exponential(train_off=True))
        self.sequential = nn.Sequential(*modules)
        
    def forward(self, x):
        if not self.training and self.infr_exp:
            return torch.exp(self.sequential(x))
        return self.sequential(x)

    def extra_repr(self):
        try:
            return 'adapt_gauss={}'.format(self.adapt_gauss)
        except:
            pass

class ScaledSoftplusBNCNN(nn.Module):
    def __init__(self, n_units=5, noise=.05, bias=True, linear_bias=None, adapt_gauss=False, chans=[8,8], bnorm_momentum=.01, inference_exp=False):
        super(ScaledSoftplusBNCNN,self).__init__()
        self.name = 'McNiruNet'
        self.n_units = n_units
        self.chans = chans
        self.infr_exp = inference_exp
        if linear_bias is None:
            linear_bias = bias
        modules = []
        modules.append(nn.Conv2d(40,chans[0],kernel_size=15, bias=bias))
        modules.append(Flatten())
        modules.append(nn.BatchNorm1d(chans[0]*36*36, eps=1e-3, momentum=bnorm_momentum))
        modules.append(GaussianNoise(std=noise, adapt=adapt_gauss))
        modules.append(nn.ReLU())
        modules.append(Reshape((-1,chans[0],36,36)))
        modules.append(nn.Conv2d(chans[0],chans[1],kernel_size=11, bias=bias))
        modules.append(Flatten())
        modules.append(nn.BatchNorm1d(chans[1]*26*26, eps=1e-3, momentum=bnorm_momentum))
        modules.append(GaussianNoise(std=noise, adapt=adapt_gauss))
        modules.append(nn.ReLU())
        modules.append(nn.Linear(chans[1]*26*26,n_units, bias=linear_bias))
        modules.append(nn.BatchNorm1d(n_units, eps=1e-3, momentum=bnorm_momentum))
        modules.append(ScaledSoftplus())
        self.sequential = nn.Sequential(*modules)
        
    def forward(self, x):
        if not self.training and self.infr_exp:
            return torch.exp(self.sequential(x))
        return self.sequential(x)

    def extra_repr(self):
        try:
            return 'adapt_gauss={}'.format(self.adapt_gauss)
        except:
            pass

class AbsBNBNCNN(nn.Module):
    def __init__(self, n_units=5, noise=.05, bias=True, linear_bias=None, adapt_gauss=False, chans=[8,8], bnorm_momentum=0.1, softplus=True, inference_exp=False):
        super(AbsBNBNCNN,self).__init__()
        self.name = 'McNiruNet'
        self.n_units = n_units
        modules = []
        self.chans = chans
        self.infr_exp = inference_exp
        if linear_bias is None:
            linear_bias = bias
        modules.append(nn.Conv2d(40,chans[0],kernel_size=15, bias=bias))
        modules.append(Flatten())
        modules.append(AbsBatchNorm1d(chans[0]*36*36, eps=1e-3, momentum=bnorm_momentum))
        modules.append(GaussianNoise(std=noise, adapt=adapt_gauss))
        modules.append(nn.ReLU())
        modules.append(Reshape((-1,chans[0],36,36)))
        modules.append(nn.Conv2d(chans[0],chans[1],kernel_size=11, bias=bias))
        modules.append(Flatten())
        modules.append(AbsBatchNorm1d(chans[1]*26*26, eps=1e-3, momentum=bnorm_momentum))
        modules.append(GaussianNoise(std=noise, adapt=adapt_gauss))
        modules.append(nn.ReLU())
        modules.append(nn.Linear(chans[1]*26*26,n_units, bias=linear_bias))
        modules.append(AbsBatchNorm1d(n_units, momentum=bnorm_momentum))
        if softplus:
            modules.append(nn.Softplus())
        else:
            modules.append(Exponential(train_off=True))
        self.sequential = nn.Sequential(*modules)
        
    def forward(self, x):
        if not self.training and self.infr_exp:
            return torch.exp(self.sequential(x))
        return self.sequential(x)

class AbsSSSSCNN(nn.Module):
    def __init__(self, n_units, scale=True, shift=False, bias=True, linear_bias=None,
                        noise=0.05, adapt_gauss=False, chans=[8,8], softplus=True, inference_exp=False):
        super(AbsSSSSCNN,self).__init__()
        self.name = 'McNiruNet'
        self.n_units = n_units
        self.chans=[8,8]
        self.infr_exp = inference_exp
        if linear_bias is None:
            linear_bias = bias
        modules = []
        modules.append(nn.Conv2d(40,chans[0],kernel_size=15, bias=bias))
        modules.append(AbsScaleShift((chans[0],36,36)))
        modules.append(GaussianNoise(std=noise, adapt=adapt_gauss))
        modules.append(nn.ReLU())
        modules.append(nn.Conv2d(chans[0],chans[1],kernel_size=11, bias=bias))
        modules.append(AbsScaleShift((chans[1],26,26)))
        modules.append(GaussianNoise(std=noise, adapt=adapt_gauss))
        modules.append(nn.ReLU())
        modules.append(Flatten())
        modules.append(nn.Linear(chans[1]*26*26,n_units, bias=linear_bias))
        modules.append(AbsScaleShift(n_units))
        if softplus:
            modules.append(nn.Softplus())
        else:
            modules.append(Exponential(train_off=True))
        self.sequential = nn.Sequential(*modules)
        
    def forward(self, x):
        if not self.training and self.infr_exp:
            return torch.exp(self.sequential(x))
        return self.sequential(x)

class BNCNN2D(nn.Module):
    def __init__(self, n_units=5, noise=.05, bias=True, linear_bias=None, adapt_gauss=False, chans=[8,8], bnorm_momentum=0.1, softplus=True, inference_exp=False):
        super(BNCNN2D,self).__init__()
        self.name = 'McNiruNet'
        self.n_units = n_units
        self.infr_exp = inference_exp
        if linear_bias is None:
            linear_bias = bias
        modules = []
        self.adapt_gauss = adapt_gauss
        self.chans = chans
        modules.append(nn.Conv2d(40,chans[0],kernel_size=15, bias=bias))
        modules.append(nn.BatchNorm2d(chans[0], eps=1e-3, momentum=bnorm_momentum))
        modules.append(GaussianNoise(std=noise, adapt=adapt_gauss))
        modules.append(nn.ReLU())

        modules.append(nn.Conv2d(chans[0],chans[1],kernel_size=11, bias=bias))
        modules.append(nn.BatchNorm2d(chans[1], eps=1e-3, momentum=bnorm_momentum))
        modules.append(GaussianNoise(std=noise, adapt=adapt_gauss))
        modules.append(nn.ReLU())

        modules.append(Flatten())
        modules.append(nn.Linear(chans[1]*26*26,n_units, bias=linear_bias))
        modules.append(nn.BatchNorm1d(n_units, momentum=bnorm_momentum))
        if softplus:
            modules.append(nn.Softplus())
        else:
            modules.append(Exponential(train_off=True))
        self.sequential = nn.Sequential(*modules)
        
    def forward(self, x):
        if not self.training and self.infr_exp:
            return torch.exp(self.sequential(x))
        return self.sequential(x)

class CNN(nn.Module):
    def __init__(self, n_units, bias=False, linear_bias=None, noise=0.05, adapt_gauss=False, chans=[8,8], softplus=True, inference_exp=False):
        super(CNN,self).__init__()
        self.name = 'McNiruNet'
        self.n_units = n_units
        self.chans = chans
        self.adapt_gauss = adapt_gauss
        self.infr_exp = inference_exp
        if linear_bias is None:
            linear_bias = bias
        modules = []
        modules.append(nn.Conv2d(40,chans[0],kernel_size=15, bias=bias))
        modules.append(GaussianNoise(std=noise, adapt=adapt_gauss))
        modules.append(nn.ReLU())
        modules.append(nn.Conv2d(chans[0],chans[1],kernel_size=11, bias=bias))
        modules.append(GaussianNoise(std=noise, adapt=adapt_gauss))
        modules.append(nn.ReLU())
        modules.append(Flatten())
        modules.append(nn.Linear(chans[1]*26*26,n_units, bias=linear_bias))
        if softplus:
            modules.append(nn.Softplus())
        else:
            modules.append(Exponential(train_off=True))
        self.sequential = nn.Sequential(*modules)
        
    def forward(self, x):
        if not self.training and self.infr_exp:
            return torch.exp(self.sequential(x))
        return self.sequential(x)

class DalesBNCNN(nn.Module):
    def __init__(self, n_units=5, bias=True, linear_bias=None, noise=0.1, adapt_gauss=False, chans=[8,8], neg_p=0.5, bnorm_momentum=0.1, abs_bias=True, softplus=True, inference_exp=False):
        super(DalesBNCNN,self).__init__()
        self.name = 'DaleNet'
        self.n_units = n_units
        self.chans = chans
        self.adapt_gauss = adapt_gauss
        self.abs_bias = abs_bias
        self.infr_exp = inference_exp
        if linear_bias is None:
            linear_bias = bias
        modules = []
        modules.append(AbsConv2d(40,chans[0],kernel_size=15, bias=bias, abs_bias=abs_bias))
        self.diminish_weight_magnitude(modules[-1].parameters())
        modules.append(Flatten())
        modules.append(nn.BatchNorm1d(chans[0]*36*36, eps=1e-3, momentum=bnorm_momentum))
        modules.append(GaussianNoise(std=noise, adapt=adapt_gauss))
        modules.append(nn.ReLU())
        modules.append(Reshape((-1,chans[0],36,36)))
        modules.append(DaleActivations(chans[0], neg_p))
        modules.append(AbsConv2d(chans[0],chans[1],kernel_size=11, bias=bias, abs_bias=abs_bias))
        self.diminish_weight_magnitude(modules[-1].parameters())
        modules.append(Flatten())
        modules.append(nn.BatchNorm1d(chans[1]*26*26, eps=1e-3, momentum=bnorm_momentum))
        modules.append(GaussianNoise(std=noise, adapt=adapt_gauss))
        modules.append(nn.ReLU())
        modules.append(Reshape((-1,chans[1],26,26)))
        modules.append(DaleActivations(chans[1], neg_p))
        modules.append(Flatten())
        modules.append(AbsLinear(chans[1]*26*26,n_units, bias=linear_bias, abs_bias=abs_bias))
        self.diminish_weight_magnitude(modules[-1].parameters())
        modules.append(nn.BatchNorm1d(n_units, momentum=bnorm_momentum))
        if softplus:
            modules.append(nn.Softplus())
        else:
            modules.append(Exponential(train_off=True))
        self.sequential = nn.Sequential(*modules)
        
    def forward(self, x):
        if not self.training and self.infr_exp:
            return torch.exp(self.sequential(x))
        return self.sequential(x)
    
    def diminish_weight_magnitude(self, params):
        for param in params:
            divisor = float(np.prod(param.data.shape))/2
            if param.data is not None and divisor >= 0:
                param.data = param.data/divisor

class AbsBNDalesBNCNN(nn.Module):
    def __init__(self, n_units=5, bias=True, linear_bias=None, noise=0.1, adapt_gauss=False, chans=[8,8], neg_p=0.5, bnorm_momentum=0.1, softplus=True, inference_exp=False):
        super(AbsBNDalesBNCNN,self).__init__()
        self.name = 'DaleNet'
        self.n_units = n_units
        self.chans = chans
        self.adapt_gauss = adapt_gauss
        self.infr_exp = inference_exp
        if linear_bias is None:
            linear_bias = bias
        modules = []
        modules.append(AbsConv2d(40,chans[0],kernel_size=15, bias=bias, abs_bias=False))
        self.diminish_weight_magnitude(modules[-1].parameters())
        modules.append(Flatten())
        modules.append(AbsBatchNorm1d(chans[0]*36*36, eps=1e-3, momentum=bnorm_momentum))
        modules.append(GaussianNoise(std=noise, adapt=adapt_gauss))
        modules.append(nn.ReLU())
        modules.append(Reshape((-1,chans[0],36,36)))
        modules.append(DaleActivations(chans[0], neg_p))
        modules.append(AbsConv2d(chans[0],chans[1],kernel_size=11, bias=bias, abs_bias=False))
        self.diminish_weight_magnitude(modules[-1].parameters())
        modules.append(Flatten())
        modules.append(AbsBatchNorm1d(chans[1]*26*26, eps=1e-3, momentum=bnorm_momentum))
        modules.append(GaussianNoise(std=noise, adapt=adapt_gauss))
        modules.append(nn.ReLU())
        modules.append(Reshape((-1,chans[1],26,26)))
        modules.append(DaleActivations(chans[1], neg_p))
        modules.append(Flatten())
        modules.append(AbsLinear(chans[1]*26*26,n_units, bias=linear_bias, abs_bias=False))
        self.diminish_weight_magnitude(modules[-1].parameters())
        modules.append(AbsBatchNorm1d(n_units, momentum=bnorm_momentum))
        if softplus:
            modules.append(nn.Softplus())
        else:
            modules.append(Exponential(train_off=True))
        self.sequential = nn.Sequential(*modules)
        
    def forward(self, x):
        if not self.training and self.infr_exp:
            return torch.exp(self.sequential(x))
        return self.sequential(x)
    
    def diminish_weight_magnitude(self, params):
        for param in params:
            divisor = float(np.prod(param.data.shape))/2
            if param.data is not None and divisor >= 0:
                param.data = param.data/divisor

class DalesSkipBNCNN(nn.Module):
    def __init__(self, n_units=5, noise=.05, bias=True, linear_bias=None,x_shape=(40,50,50), skip_depth=2, 
                        neg_p=.5, adapt_gauss=False, chans=[8,8], bnorm_momentum=0.1, softplus=True, inference_exp=False):
        super(DalesSkipBNCNN,self).__init__()
        self.name = 'SkipNet'
        self.n_units = n_units
        self.chans=chans
        self.adapt_gauss=adapt_gauss
        self.infr_exp = inference_exp
        if linear_bias is None:
            linear_bias = bias
        modules = []
        #in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True
        modules.append(DalesSkipConnection(40,skip_depth,15,x_shape=x_shape,bias=bias, noise=noise, neg_p=1))
        modules.append(AbsConv2d(40+skip_depth,chans[0],kernel_size=15, bias=bias, abs_bias=False))
        diminish_weight_magnitude(modules[-1].parameters())
        modules.append(Flatten())
        modules.append(nn.BatchNorm1d(chans[0]*36*36, eps=1e-3, momentum=bnorm_momentum))
        modules.append(GaussianNoise(std=noise, adapt=adapt_gauss))
        modules.append(nn.ReLU())
        modules.append(Reshape((-1,chans[0],36,36)))
        modules.append(DaleActivations(chans[0], neg_p))
        modules.append(AbsConv2d(chans[0],chans[1],kernel_size=11, bias=bias, abs_bias=False))
        diminish_weight_magnitude(modules[-1].parameters())
        modules.append(Flatten())
        modules.append(nn.BatchNorm1d(chans[1]*26*26, eps=1e-3, momentum=bnorm_momentum))
        modules.append(GaussianNoise(std=noise, adapt=adapt_gauss))
        modules.append(nn.ReLU())
        modules.append(Reshape((-1,chans[1],26,26)))
        modules.append(DaleActivations(chans[1], neg_p))
        modules.append(Flatten())
        modules.append(AbsLinear(chans[1]*26*26,n_units, bias=linear_bias, abs_bias=False))
        diminish_weight_magnitude(modules[-1].parameters())
        modules.append(nn.BatchNorm1d(n_units, momentum=bnorm_momentum))
        if softplus:
            modules.append(nn.Softplus())
        else:
            modules.append(Exponential(train_off=True))
        self.sequential = nn.Sequential(*modules)
        
    def forward(self, x):
        if not self.training and self.infr_exp:
            return torch.exp(self.sequential(x))
        return self.sequential(x)

class DalesDoubleSkipBNCNN(nn.Module):
    def __init__(self, n_units=5, noise=.05, bias=True, linear_bias=None,x_shape=(40,50,50), skip_depth=2, 
                      neg_p=.5, adapt_gauss=False, chans=[8,8], bnorm_momentum=0.1, softplus=True, inference_exp=False):
        super(DalesDoubleSkipBNCNN,self).__init__()
        self.name = 'SkipNet'
        self.n_units = n_units
        self.chans=chans
        self.adapt_gauss=adapt_gauss
        self.infr_exp = inference_exp
        if linear_bias is None:
            linear_bias = bias
        modules = []
        #in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True
        modules.append(DalesSkipConnection(40,skip_depth,15,x_shape=x_shape,bias=bias, noise=noise, neg_p=1, bnorm_momentum=bnorm_momentum))
        modules.append(AbsConv2d(40+skip_depth,chans[0],kernel_size=15, bias=bias, abs_bias=False))
        diminish_weight_magnitude(modules[-1].parameters())
        modules.append(Flatten())
        modules.append(nn.BatchNorm1d(chans[0]*36*36, eps=1e-3, momentum=bnorm_momentum))
        modules.append(GaussianNoise(std=noise, adapt=adapt_gauss))
        modules.append(nn.ReLU())
        modules.append(Reshape((-1,chans[0],36,36)))
        modules.append(DaleActivations(chans[0], neg_p))
        modules.append(DalesSkipConnection(chans[0],chans[1],kernel_size=11, x_shape=(chans[0],36,36), bias=bias, noise=noise,neg_p=.5, bnorm_momentum=bnorm_momentum))
        modules.append(Flatten())
        modules.append(AbsLinear((chans[1]+chans[0])*36*36,n_units, bias=linear_bias, abs_bias=False))
        diminish_weight_magnitude(modules[-1].parameters())
        modules.append(nn.BatchNorm1d(n_units, momentum=bnorm_momentum))
        if softplus:
            modules.append(nn.Softplus())
        else:
            modules.append(Exponential(train_off=True))
        self.sequential = nn.Sequential(*modules)
        
    def forward(self, x):
        if not self.training and self.infr_exp:
            return torch.exp(self.sequential(x))
        return self.sequential(x)

class DalesSSCNN(nn.Module):
    def __init__(self, n_units=5, bias=True, linear_bias=None,noise=0.1, neg_p=0.5, scale=True, shift=True,
                                         adapt_gauss=False, chans=[8,8], softplus=True, inference_exp=False):
        super(DalesSSCNN,self).__init__()
        self.name = 'DaleNet'
        self.n_units = n_units
        self.chans=chans
        self.adapt_gauss = adapt_gauss
        self.infr_exp = inference_exp
        if linear_bias is None:
            linear_bias = bias
        modules = []
        modules.append(AbsConv2d(40,chans[0],kernel_size=15, bias=bias, abs_bias=False))
        self.diminish_weight_magnitude(modules[-1].parameters())
        modules.append(ScaleShift((chans[0],36,36), scale=scale, shift=shift))
        modules.append(GaussianNoise(std=noise, adapt=adapt_gauss))
        modules.append(nn.ReLU())
        modules.append(DaleActivations(chans[0], neg_p))
        modules.append(AbsConv2d(chans[0],chans[1],kernel_size=11, bias=bias, abs_bias=False))
        self.diminish_weight_magnitude(modules[-1].parameters())
        modules.append(ScaleShift((chans[1],26,26), scale=scale, shift=shift))
        modules.append(GaussianNoise(std=noise, adapt=adapt_gauss))
        modules.append(nn.ReLU())
        modules.append(DaleActivations(chans[1], neg_p))
        modules.append(Flatten())
        modules.append(AbsLinear(chans[1]*26*26,n_units, bias=linear_bias, abs_bias=False))
        self.diminish_weight_magnitude(modules[-1].parameters())
        modules.append(ScaleShift(n_units, scale=scale, shift=shift))
        if softplus:
            modules.append(nn.Softplus())
        else:
            modules.append(Exponential(train_off=True))
        self.sequential = nn.Sequential(*modules)
        
    def forward(self, x):
        if not self.training and self.infr_exp:
            return torch.exp(self.sequential(x))
        return self.sequential(x)
    
    def diminish_weight_magnitude(self, params):
        for param in params:
            divisor = float(np.prod(param.data.shape))/2
            if param.data is not None and divisor >= 0:
                param.data = param.data/divisor

class Gauss1dBNCNN(nn.Module):
    def __init__(self, n_units=5, noise=.05, bias=True, linear_bias=None,adapt_gauss=False, chans=[8,8], bnorm_momentum=0.1, softplus=True, inference_exp=False):
        super(Gauss1dBNCNN,self).__init__()
        self.name = 'McNiruNet'
        self.n_units = n_units
        self.chans=chans
        self.infr_exp = inference_exp
        if linear_bias is None:
            linear_bias = bias
        modules = []
        modules.append(nn.Conv2d(40,chans[0],kernel_size=15, bias=bias))
        modules.append(Flatten())
        modules.append(nn.BatchNorm1d(chans[0]*36*36, eps=1e-3, momentum=bnorm_momentum))
        modules.append(GaussianNoise1d(chans[0]*36*36, noise=noise))
        modules.append(nn.ReLU())
        modules.append(Reshape((-1,chans[0],36,36)))
        modules.append(nn.Conv2d(chans[0],chans[1],kernel_size=11, bias=bias))
        modules.append(Flatten())
        modules.append(nn.BatchNorm1d(chans[1]*26*26, eps=1e-3, momentum=bnorm_momentum))
        modules.append(GaussianNoise1d(chans[1]*26*26, noise=noise))
        modules.append(nn.ReLU())
        modules.append(nn.Linear(chans[1]*26*26,n_units, bias=linear_bias))
        modules.append(nn.BatchNorm1d(n_units, momentum=bnorm_momentum))
        if softplus:
            modules.append(nn.Softplus())
        else:
            modules.append(Exponential(train_off=True))
        self.sequential = nn.Sequential(*modules)
        
    def forward(self, x):
        if not self.training and self.infr_exp:
            return torch.exp(self.sequential(x))
        return self.sequential(x)

class ParallelDataBNCNN(nn.Module):
    def __init__(self, n_units=[5], noise=.05, bias=True, linear_bias=None,adapt_gauss=False, chans=[8,8], bnorm_momentum=0.1, softplus=True, inference_exp=False):
        super(ParallelDataBNCNN,self).__init__()
        self.name = 'McNiruNet'
        self.n_units = n_units
        self.chans=chans
        self.infr_exp = inference_exp
        if linear_bias is None:
            linear_bias = bias
        # Create base of model
        modules = []
        modules.append(nn.Conv2d(40,chans[0],kernel_size=15, bias=bias))
        modules.append(Flatten())
        modules.append(nn.BatchNorm1d(chans[0]*36*36, eps=1e-3, momentum=bnorm_momentum))
        modules.append(GaussianNoise(std=noise, adapt=adapt_gauss))
        modules.append(nn.ReLU())
        modules.append(Reshape((-1,chans[0],36,36)))
        modules.append(nn.Conv2d(chans[0],chans[1],kernel_size=11, bias=bias))
        modules.append(Flatten())
        modules.append(nn.BatchNorm1d(chans[1]*26*26, eps=1e-3, momentum=bnorm_momentum))
        modules.append(GaussianNoise(std=noise, adapt=adapt_gauss))
        modules.append(nn.ReLU())
        self.features = nn.Sequential(*modules)

        # Create output layers
        self.output_layers = nn.ModuleList([])
        for n_units in n_units:
            modules = []
            modules.append(nn.Linear(chans[1]*26*26, n_units, bias=linear_bias))
            modules.append(nn.BatchNorm1d(n_units, momentum=bnorm_momentum))
            if softplus:
                modules.append(nn.Softplus())
            else:
                modules.append(Exponential(train_off=True))
            self.output_layers.append(nn.Sequential(*modules))
        
    def forward(self, x, output_idx):
        feats = self.features(x)
        if not self.training and self.infr_exp:
            return torch.exp(self.output_layers[output_idx](feats))
        return self.output_layers[output_idx](feats)

    def req_grad(self, mode):
        """
        If mode is false, gradients are not calculated. If mode is true gradients are calculated.

        mode : bool
        """
        for p in self.parameters():
            try:
                p.requires_grad = mode
            except:
                pass
 
class ParallelDataStackedBNCNN(nn.Module):
    def __init__(self, n_units=[5], noise=.05, bias=True, linear_bias=None, adapt_gauss=False, chans=[8,8], bnorm_momentum=0.1, softplus=True, inference_exp=False):
        super(ParallelDataStackedBNCNN,self).__init__()
        self.name = 'McNiruNet'
        self.n_units = n_units
        self.chans=chans
        self.infr_exp = inference_exp
        if linear_bias is None:
            linear_bias = bias
        # Create base of model
        modules = []
        modules.append(StackedConv2d(40,chans[0],kernel_size=15, bias=bias))
        modules.append(Flatten())
        modules.append(nn.BatchNorm1d(chans[0]*36*36, eps=1e-3, momentum=bnorm_momentum))
        modules.append(GaussianNoise(std=noise, adapt=adapt_gauss))
        modules.append(nn.ReLU())
        modules.append(Reshape((-1,chans[0],36,36)))
        modules.append(StackedConv2d(chans[0],chans[1],kernel_size=11, bias=bias))
        modules.append(Flatten())
        modules.append(nn.BatchNorm1d(chans[1]*26*26, eps=1e-3, momentum=bnorm_momentum))
        modules.append(GaussianNoise(std=noise, adapt=adapt_gauss))
        modules.append(nn.ReLU())
        self.features = nn.Sequential(*modules)

        # Create output layers
        self.output_layers = nn.ModuleList([])
        for n_units in n_units:
            modules = []
            modules.append(nn.Linear(chans[1]*26*26, n_units, bias=linear_bias))
            modules.append(nn.BatchNorm1d(n_units, momentum=bnorm_momentum))
            if softplus:
                modules.append(nn.Softplus())
            else:
                modules.append(Exponential(train_off=True))
            self.output_layers.append(nn.Sequential(*modules))
        
    def forward(self, x, output_idx):
        feats = self.features(x)
        if not self.training and self.infr_exp:
            return torch.exp(self.output_layers[output_idx](feats))
        return self.output_layers[output_idx](feats)

    def req_grad(self, mode):
        """
        If mode is false, gradients are not calculated. If mode is true gradients are calculated.

        mode : bool
        """
        for p in self.parameters():
            try:
                p.requires_grad = mode
            except:
                pass

class AbsBNPracticalBNCNN(nn.Module):
    def __init__(self, n_units=5, noise=.1, bias=True, linear_bias=None,chans=[8,8], bnorm_momentum=0.1, softplus=True, inference_exp=False):
        super(AbsBNPracticalBNCNN,self).__init__()
        self.chans=chans
        self.infr_exp = inference_exp
        self.n_units = n_units
        if linear_bias is None:
            linear_bias = bias
        modules = []
        modules.append(nn.Conv2d(40,chans[0],kernel_size=15, bias=bias))
        modules.append(nn.Dropout(p=noise/2))
        modules.append(nn.ReLU())
        modules.append(Flatten())
        modules.append(AbsBatchNorm1d(chans[0]*36*36, eps=1e-3, momentum=bnorm_momentum))
        modules.append(Reshape((-1,chans[0],36,36)))
        modules.append(nn.Conv2d(chans[0],chans[1],kernel_size=11, bias=bias))
        modules.append(nn.Dropout(p=noise))
        modules.append(nn.ReLU())
        modules.append(Flatten())
        modules.append(AbsBatchNorm1d(chans[1]*26*26, eps=1e-3, momentum=bnorm_momentum))
        modules.append(nn.Linear(chans[1]*26*26,n_units, bias=linear_bias))
        modules.append(nn.BatchNorm1d(n_units, momentum=bnorm_momentum))
        if softplus:
            modules.append(nn.Softplus())
        else:
            modules.append(Exponential(train_off=True))
        self.sequential = nn.Sequential(*modules)
        
    def forward(self, x):
        if not self.training and self.infr_exp:
            return torch.exp(self.sequential(x))
        return self.sequential(x)
        
class PracticalMeanBNCNN(nn.Module):
    def __init__(self, n_units=5, noise=.1, bias=True, linear_bias=None,chans=[8,8], bnorm_momentum=0.1, softplus=True, inference_exp=False):
        super(PracticalMeanBNCNN,self).__init__()
        self.chans=chans
        self.infr_exp = inference_exp
        self.n_units = n_units
        if linear_bias is None:
            linear_bias = bias
        modules = []
        modules.append(nn.Conv2d(40,chans[0],kernel_size=15, bias=bias))
        modules.append(nn.Dropout(p=noise/2))
        modules.append(nn.ReLU())
        modules.append(Flatten())
        modules.append(MeanOnlyBatchNorm1d(chans[0]*36*36, eps=1e-3, momentum=bnorm_momentum))
        modules.append(Reshape((-1,chans[0],36,36)))
        modules.append(nn.Conv2d(chans[0],chans[1],kernel_size=11, bias=bias))
        modules.append(nn.Dropout(p=noise))
        modules.append(nn.ReLU())
        modules.append(Flatten())
        modules.append(MeanOnlyBatchNorm1d(chans[1]*26*26, eps=1e-3, momentum=bnorm_momentum))
        modules.append(nn.Linear(chans[1]*26*26,n_units, bias=linear_bias))
        modules.append(nn.BatchNorm1d(n_units, momentum=bnorm_momentum))
        if softplus:
            modules.append(nn.Softplus())
        else:
            modules.append(Exponential(train_off=True))
        self.sequential = nn.Sequential(*modules)
        
    def forward(self, x):
        if not self.training and self.infr_exp:
            return torch.exp(self.sequential(x))
        return self.sequential(x)

class PracticalBNCNN(nn.Module):
    def __init__(self, n_units=5, noise=.1, bias=True, linear_bias=None,chans=[8,8], bnorm_momentum=0.1, softplus=True, inference_exp=False):
        super(PracticalBNCNN,self).__init__()
        self.chans=chans
        self.infr_exp = inference_exp
        self.n_units = n_units
        if linear_bias is None:
            linear_bias = bias
        modules = []
        modules.append(nn.Conv2d(40,chans[0],kernel_size=15, bias=bias))
        modules.append(nn.Dropout(p=noise/2))
        modules.append(nn.ReLU())
        modules.append(Flatten())
        modules.append(nn.BatchNorm1d(chans[0]*36*36, eps=1e-3, momentum=bnorm_momentum))
        modules.append(Reshape((-1,chans[0],36,36)))
        modules.append(nn.Conv2d(chans[0],chans[1],kernel_size=11, bias=bias))
        modules.append(nn.Dropout(p=noise))
        modules.append(nn.ReLU())
        modules.append(Flatten())
        modules.append(nn.BatchNorm1d(chans[1]*26*26, eps=1e-3, momentum=bnorm_momentum))
        modules.append(nn.Linear(chans[1]*26*26,n_units, bias=linear_bias))
        modules.append(nn.BatchNorm1d(n_units, momentum=bnorm_momentum))
        if softplus:
            modules.append(nn.Softplus())
        else:
            modules.append(Exponential(train_off=True))
        self.sequential = nn.Sequential(*modules)
        
    def forward(self, x):
        if not self.training and self.infr_exp:
            return torch.exp(self.sequential(x))
        return self.sequential(x)

class PracticalBNCNN2D(nn.Module):
    def __init__(self, n_units=5, noise=.3, bias=True, linear_bias=None,chans=[8,8], bnorm_momentum=0.1, softplus=True, inference_exp=False):
        super(PracticalBNCNN2D,self).__init__()
        self.name = 'McNiruNet'
        self.chans=[8,8]
        self.infr_exp = inference_exp
        self.n_units = n_units
        if linear_bias is None:
            linear_bias = bias
        modules = []
        modules.append(nn.Conv2d(40,chans[0],kernel_size=15, bias=bias))
        modules.append(nn.Dropout(p=noise/2))
        modules.append(nn.ReLU())
        modules.append(nn.BatchNorm2d(chans[0], eps=1e-3, momentum=bnorm_momentum))

        modules.append(nn.Conv2d(chans[0],chans[1],kernel_size=11, bias=bias))
        modules.append(nn.Dropout(p=noise))
        modules.append(nn.ReLU())
        modules.append(nn.BatchNorm2d(chans[1], eps=1e-3, momentum=bnorm_momentum))

        modules.append(Flatten())
        modules.append(nn.Linear(chans[1]*26*26,n_units, bias=linear_bias))
        modules.append(nn.BatchNorm1d(n_units, momentum=bnorm_momentum))
        if softplus:
            modules.append(nn.Softplus())
        else:
            modules.append(Exponential(train_off=True))
        self.sequential = nn.Sequential(*modules)
        
    def forward(self, x):
        if not self.training and self.infr_exp:
            return torch.exp(self.sequential(x))
        return self.sequential(x)

class SkipBNBNCNN(nn.Module):
    def __init__(self, n_units=5, noise=.05, bias=True, linear_bias=None,x_shape=(40,50,50), chans=[8,8], adapt_gauss=False, bnorm_momentum=0.1, softplus=True, inference_exp=False):
        super(SkipBNBNCNN,self).__init__()
        self.name = 'SkipNet'
        self.chans = chans
        self.n_units = n_units
        self.infr_exp = inference_exp
        if linear_bias is None:
            linear_bias = bias
        modules = []
        #in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True
        modules.append(SkipConnectionBN(40,2,15,x_shape=x_shape,bias=bias, noise=noise))
        modules.append(nn.Conv2d(48,8,kernel_size=15, bias=bias))
        modules.append(Flatten())
        modules.append(nn.BatchNorm1d(8*36*36, eps=1e-3, momentum=bnorm_momentum))
        modules.append(GaussianNoise(std=noise, adapt=adapt_gauss))
        modules.append(nn.ReLU())
        modules.append(Reshape((-1,8,36,36)))
        modules.append(nn.Conv2d(8,8,kernel_size=11, bias=bias))
        modules.append(Flatten())
        modules.append(nn.BatchNorm1d(8*26*26, eps=1e-3, momentum=bnorm_momentum))
        modules.append(GaussianNoise(std=noise, adapt=adapt_gauss))
        modules.append(nn.ReLU())
        modules.append(nn.Linear(8*26*26,n_units, bias=linear_bias))
        modules.append(nn.BatchNorm1d(n_units, eps=1e-3, momentum=bnorm_momentum))
        if softplus:
            modules.append(nn.Softplus())
        else:
            modules.append(Exponential(train_off=True))
        self.sequential = nn.Sequential(*modules)
        
    def forward(self, x):
        if not self.training and self.infr_exp:
            return torch.exp(self.sequential(x))
        return self.sequential(x)

class SSCNN(nn.Module):
    def __init__(self, n_units, scale=True, shift=False, bias=True, linear_bias=None, 
                             noise=0.05, chans=[8,8], adapt_gauss=False, softplus=True, inference_exp=False):
        super(SSCNN,self).__init__()
        self.name = 'McNiruNet'
        self.chans=chans
        self.n_units = n_units
        self.infr_exp = inference_exp
        if linear_bias is None:
            linear_bias = bias
        modules = []
        modules.append(nn.Conv2d(40,chans[0],kernel_size=15, bias=bias))
        modules.append(ScaleShift((chans[0],36,36)))
        modules.append(GaussianNoise(std=noise, adapt=adapt_gauss))
        modules.append(nn.ReLU())
        modules.append(nn.Conv2d(chans[0],chans[1],kernel_size=11, bias=bias))
        modules.append(ScaleShift((chans[1],26,26)))
        modules.append(GaussianNoise(std=noise, adapt=adapt_gauss))
        modules.append(nn.ReLU())
        modules.append(Flatten())
        modules.append(nn.Linear(chans[1]*26*26,n_units, bias=linear_bias))
        modules.append(ScaleShift(n_units))
        if softplus:
            modules.append(nn.Softplus())
        else:
            modules.append(Exponential(train_off=True))
        self.sequential = nn.Sequential(*modules)
        
    def forward(self, x):
        if not self.training and self.infr_exp:
            return torch.exp(self.sequential(x))
        return self.sequential(x)

class StackedBNCNN(nn.Module):
    def __init__(self, n_units=5, noise=.05, bias=True, linear_bias=None, bnorm_momentum=0.1, adapt_gauss=False, chans=[8,8], softplus=True, inference_exp=False):
        super(StackedBNCNN,self).__init__()
        self.name = 'StackedNet'
        self.chans = chans
        self.n_units = n_units
        self.infr_exp = inference_exp
        if linear_bias is None:
            linear_bias = bias
        modules = []
        modules.append(StackedConv2d(40,chans[0],kernel_size=15, bias=bias))
        modules.append(Flatten())
        modules.append(nn.BatchNorm1d(chans[0]*36*36, eps=1e-3, momentum=bnorm_momentum))
        modules.append(GaussianNoise(std=noise, adapt=adapt_gauss))
        modules.append(nn.ReLU())
        modules.append(Reshape((-1,chans[0],36,36)))
        modules.append(StackedConv2d(chans[0],chans[1],kernel_size=11, bias=bias))
        modules.append(Flatten())
        modules.append(nn.BatchNorm1d(chans[1]*26*26, eps=1e-3, momentum=bnorm_momentum))
        modules.append(GaussianNoise(std=noise, adapt=adapt_gauss))
        modules.append(nn.ReLU())
        modules.append(nn.Linear(chans[1]*26*26,n_units, bias=linear_bias))
        modules.append(nn.BatchNorm1d(n_units, eps=1e-3, momentum=bnorm_momentum))
        if softplus:
            modules.append(nn.Softplus())
        else:
            modules.append(Exponential(train_off=True))
        self.sequential = nn.Sequential(*modules)

    def forward(self, x):
        if not self.training and self.infr_exp:
            return torch.exp(self.sequential(x))
        return self.sequential(x)
    
class AbsBNStackedBNCNN(nn.Module):
    def __init__(self, n_units=5, noise=.05, bias=True, linear_bias=None,adapt_gauss=False, chans=[8,8], bnorm_momentum=0.1, softplus=True, inference_exp=False):
        super(AbsBNStackedBNCNN,self).__init__()
        self.name = 'StackedNet'
        self.chans = chans
        self.n_units = n_units
        self.infr_exp = inference_exp
        if linear_bias is None:
            linear_bias = bias
        modules = []
        modules.append(StackedConv2d(40,chans[0],kernel_size=15, abs_bnorm=True, bias=bias))
        modules.append(Flatten())
        modules.append(AbsBatchNorm1d(chans[0]*36*36, eps=1e-3, momentum=bnorm_momentum))
        modules.append(GaussianNoise(std=noise, adapt=adapt_gauss))
        modules.append(nn.ReLU())
        modules.append(Reshape((-1,chans[0],36,36)))
        modules.append(StackedConv2d(chans[0],chans[1],kernel_size=11, abs_bnorm=True, bias=bias))
        modules.append(Flatten())
        modules.append(AbsBatchNorm1d(chans[1]*26*26, eps=1e-3, momentum=bnorm_momentum))
        modules.append(GaussianNoise(std=noise, adapt=adapt_gauss))
        modules.append(nn.ReLU())
        modules.append(nn.Linear(chans[1]*26*26,n_units, bias=linear_bias))
        modules.append(AbsBatchNorm1d(n_units, eps=1e-3, momentum=bnorm_momentum))
        if softplus:
            modules.append(nn.Softplus())
        else:
            modules.append(Exponential(train_off=True))
        self.sequential = nn.Sequential(*modules)

    def forward(self, x):
        if not self.training and self.infr_exp:
            return torch.exp(self.sequential(x))
        return self.sequential(x)
    