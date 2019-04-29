import torch
import torch.nn as nn
from torchdeepretina.torch_utils import *
import numpy as np

class BNCNN(nn.Module):
    def __init__(self, output_units=5, noise=.05, bias=True, adapt_gauss=False):
        super(BNCNN,self).__init__()
        self.name = 'McNiruNet'
        modules = []
        modules.append(nn.Conv2d(40,8,kernel_size=15, bias=bias))
        modules.append(Flatten())
        modules.append(nn.BatchNorm1d(8*36*36, eps=1e-3, momentum=.99))
        modules.append(GaussianNoise(std=noise, adapt=adapt_gauss))
        modules.append(nn.ReLU())
        modules.append(Reshape((-1,8,36,36)))
        modules.append(nn.Conv2d(8,8,kernel_size=11, bias=bias))
        modules.append(Flatten())
        modules.append(nn.BatchNorm1d(8*26*26, eps=1e-3, momentum=.99))
        modules.append(GaussianNoise(std=noise, adapt=adapt_gauss))
        modules.append(nn.ReLU())
        modules.append(nn.Linear(8*26*26,output_units, bias=bias))
        modules.append(nn.BatchNorm1d(output_units))
        modules.append(nn.Softplus())
        self.sequential = nn.Sequential(*modules)
        
    def forward(self, x):
        return self.sequential(x)

    def extra_repr(self):
        try:
            return 'adapt_gauss={}'.format(self.adapt_gauss)
        except:
            pass

class AbsBNBNCNN(nn.Module):
    def __init__(self, output_units=5, noise=.05, bias=True, adapt_gauss=False, chans=[8,8]):
        super(AbsBNBNCNN,self).__init__()
        self.name = 'McNiruNet'
        modules = []
        self.chans = chans
        modules.append(nn.Conv2d(40,chans[0],kernel_size=15, bias=bias))
        modules.append(Flatten())
        modules.append(AbsBatchNorm1d(chans[0]*36*36, eps=1e-3, momentum=.99))
        modules.append(GaussianNoise(std=noise, adapt=adapt_gauss))
        modules.append(nn.ReLU())
        modules.append(Reshape((-1,chans[0],36,36)))
        modules.append(nn.Conv2d(chans[0],chans[1],kernel_size=11, bias=bias))
        modules.append(Flatten())
        modules.append(AbsBatchNorm1d(chans[1]*26*26, eps=1e-3, momentum=.99))
        modules.append(GaussianNoise(std=noise, adapt=adapt_gauss))
        modules.append(nn.ReLU())
        modules.append(nn.Linear(chans[1]*26*26,output_units, bias=bias))
        modules.append(AbsBatchNorm1d(output_units))
        modules.append(nn.Softplus())
        self.sequential = nn.Sequential(*modules)
        
    def forward(self, x):
        return self.sequential(x)

class AbsSSSSCNN(nn.Module):
    def __init__(self, output_units, scale=True, shift=False, bias=True, noise=0.05, adapt_gauss=False):
        super(AbsSSSSCNN,self).__init__()
        self.name = 'McNiruNet'
        modules = []
        modules.append(nn.Conv2d(40,8,kernel_size=15, bias=bias))
        modules.append(AbsScaleShift((8,36,36)))
        modules.append(GaussianNoise(std=noise, adapt=adapt_gauss))
        modules.append(nn.ReLU())
        modules.append(nn.Conv2d(8,8,kernel_size=11, bias=bias))
        modules.append(AbsScaleShift((8,26,26)))
        modules.append(GaussianNoise(std=noise, adapt=adapt_gauss))
        modules.append(nn.ReLU())
        modules.append(Flatten())
        modules.append(nn.Linear(8*26*26,output_units, bias=bias))
        modules.append(AbsScaleShift(output_units))
        modules.append(nn.Softplus())
        self.sequential = nn.Sequential(*modules)
        
    def forward(self, x):
        return self.sequential(x)

class BNCNN2D(nn.Module):
    def __init__(self, output_units=5, noise=.05, bias=True, chans=None):
        super(BNCNN2D,self).__init__()
        self.name = 'McNiruNet'
        modules = []
        if chans is None:
            chans = [8,8]
        modules.append(nn.Conv2d(40,chans[0],kernel_size=15, bias=bias))
        modules.append(nn.BatchNorm2d(chans[0], eps=1e-3, momentum=.99))
        modules.append(GaussianNoise(std=noise))
        modules.append(nn.ReLU())

        modules.append(nn.Conv2d(chans[0],chans[1],kernel_size=11, bias=bias))
        modules.append(nn.BatchNorm2d(chans[1], eps=1e-3, momentum=.99))
        modules.append(GaussianNoise(std=noise))
        modules.append(nn.ReLU())

        modules.append(Flatten())
        modules.append(nn.Linear(chans[1]*26*26,output_units, bias=bias))
        modules.append(nn.BatchNorm1d(output_units))
        modules.append(nn.Softplus())
        self.sequential = nn.Sequential(*modules)
        
    def forward(self, x):
        return self.sequential(x)

class CNN(nn.Module):
    def __init__(self, output_units, bias=False, noise=0.05):
        super(CNN,self).__init__()
        self.name = 'McNiruNet'
        modules = []
        modules.append(nn.Conv2d(40,8,kernel_size=15, bias=bias))
        modules.append(GaussianNoise(std=noise))
        modules.append(nn.ReLU())
        modules.append(nn.Conv2d(8,8,kernel_size=11, bias=bias))
        modules.append(GaussianNoise(std=noise))
        modules.append(nn.ReLU())
        modules.append(Flatten())
        modules.append(nn.Linear(8*26*26,output_units, bias=bias))
        modules.append(nn.Softplus())
        self.sequential = nn.Sequential(*modules)
        
    def forward(self, x):
        return self.sequential(x)

class DalesBNCNN(nn.Module):
    def __init__(self, output_units=5, bias=True, noise=0.1, neg_p=0.5):
        super(DalesBNCNN,self).__init__()
        self.name = 'DaleNet'
        modules = []
        modules.append(AbsConv2d(40,8,kernel_size=15, bias=bias))
        self.diminish_weight_magnitude(modules[-1].parameters())
        modules.append(Flatten())
        modules.append(nn.BatchNorm1d(8*36*36, eps=1e-3))
        modules.append(GaussianNoise(std=noise))
        modules.append(nn.ReLU())
        modules.append(Reshape((-1,8,36,36)))
        modules.append(DaleActivations(8, neg_p))
        modules.append(AbsConv2d(8,8,kernel_size=11, bias=bias))
        self.diminish_weight_magnitude(modules[-1].parameters())
        modules.append(Flatten())
        modules.append(nn.BatchNorm1d(8*26*26, eps=1e-3))
        modules.append(GaussianNoise(std=noise))
        modules.append(nn.ReLU())
        modules.append(Reshape((-1,8,26,26)))
        modules.append(DaleActivations(8, neg_p))
        modules.append(Flatten())
        modules.append(AbsLinear(8*26*26,output_units, bias=bias))
        self.diminish_weight_magnitude(modules[-1].parameters())
        modules.append(nn.BatchNorm1d(output_units))
        modules.append(nn.Softplus())
        self.sequential = nn.Sequential(*modules)
        
    def forward(self, x):
        return self.sequential(x)
    
    def diminish_weight_magnitude(self, params):
        for param in params:
            divisor = float(np.prod(param.data.shape))/2
            if param.data is not None and divisor >= 0:
                param.data = param.data/divisor

class DalesSkipBNCNN(nn.Module):
    def __init__(self, output_units=5, noise=.05, bias=True, x_shape=(40,50,50), skip_depth=2, neg_p=.5):
        super(DalesSkipBNCNN,self).__init__()
        self.name = 'SkipNet'
        modules = []
        #in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True
        modules.append(DalesSkipConnection(40,skip_depth,15,x_shape=x_shape,bias=bias, noise=noise, neg_p=1))
        modules.append(AbsConv2d(40+skip_depth,8,kernel_size=15, bias=bias))
        diminish_weight_magnitude(modules[-1].parameters())
        modules.append(Flatten())
        modules.append(nn.BatchNorm1d(8*36*36, eps=1e-3))
        modules.append(GaussianNoise(std=noise))
        modules.append(nn.ReLU())
        modules.append(Reshape((-1,8,36,36)))
        modules.append(DaleActivations(8, neg_p))
        modules.append(AbsConv2d(8,8,kernel_size=11, bias=bias))
        diminish_weight_magnitude(modules[-1].parameters())
        modules.append(Flatten())
        modules.append(nn.BatchNorm1d(8*26*26, eps=1e-3))
        modules.append(GaussianNoise(std=noise))
        modules.append(nn.ReLU())
        modules.append(Reshape((-1,8,26,26)))
        modules.append(DaleActivations(8, neg_p))
        modules.append(Flatten())
        modules.append(AbsLinear(8*26*26,output_units, bias=bias))
        diminish_weight_magnitude(modules[-1].parameters())
        modules.append(nn.BatchNorm1d(output_units))
        modules.append(nn.Softplus())
        self.sequential = nn.Sequential(*modules)
        
    def forward(self, x):
        return self.sequential(x)

class DalesSSCNN(nn.Module):
    def __init__(self, output_units=5, bias=True, noise=0.1, neg_p=0.5, scale=True, shift=True):
        super(DalesSSCNN,self).__init__()
        self.name = 'DaleNet'
        modules = []
        modules.append(AbsConv2d(40,8,kernel_size=15, bias=bias))
        self.diminish_weight_magnitude(modules[-1].parameters())
        modules.append(ScaleShift((8,36,36), scale=scale, shift=shift))
        modules.append(GaussianNoise(std=noise))
        modules.append(nn.ReLU())
        modules.append(DaleActivations(8, neg_p))
        modules.append(AbsConv2d(8,8,kernel_size=11, bias=bias))
        self.diminish_weight_magnitude(modules[-1].parameters())
        modules.append(ScaleShift((8,26,26), scale=scale, shift=shift))
        modules.append(GaussianNoise(std=noise))
        modules.append(nn.ReLU())
        modules.append(DaleActivations(8, neg_p))
        modules.append(Flatten())
        modules.append(AbsLinear(8*26*26,output_units, bias=bias))
        self.diminish_weight_magnitude(modules[-1].parameters())
        modules.append(ScaleShift(output_units, scale=scale, shift=shift))
        modules.append(nn.Softplus())
        self.sequential = nn.Sequential(*modules)
        
    def forward(self, x):
        return self.sequential(x)
    
    def diminish_weight_magnitude(self, params):
        for param in params:
            divisor = float(np.prod(param.data.shape))/2
            if param.data is not None and divisor >= 0:
                param.data = param.data/divisor

class Gauss1dBNCNN(nn.Module):
    def __init__(self, output_units=5, noise=.05, bias=True, adapt_gauss=False):
        super(Gauss1dBNCNN,self).__init__()
        self.name = 'McNiruNet'
        modules = []
        modules.append(nn.Conv2d(40,8,kernel_size=15, bias=bias))
        modules.append(Flatten())
        modules.append(nn.BatchNorm1d(8*36*36, eps=1e-3, momentum=.99))
        modules.append(GaussianNoise1d(8*36*36, noise=noise))
        modules.append(nn.ReLU())
        modules.append(Reshape((-1,8,36,36)))
        modules.append(nn.Conv2d(8,8,kernel_size=11, bias=bias))
        modules.append(Flatten())
        modules.append(nn.BatchNorm1d(8*26*26, eps=1e-3, momentum=.99))
        modules.append(GaussianNoise1d(8*26*26, noise=noise))
        modules.append(nn.ReLU())
        modules.append(nn.Linear(8*26*26,output_units, bias=bias))
        modules.append(nn.BatchNorm1d(output_units))
        modules.append(nn.Softplus())
        self.sequential = nn.Sequential(*modules)
        
    def forward(self, x):
        return self.sequential(x)

class ParallelDataBNCNN(nn.Module):
    def __init__(self, output_units=[5], noise=.05, bias=True):
        super(ParallelDataBNCNN,self).__init__()
        self.name = 'McNiruNet'

        # Create base of model
        modules = []
        modules.append(nn.Conv2d(40,8,kernel_size=15, bias=bias))
        modules.append(Flatten())
        modules.append(nn.BatchNorm1d(8*36*36, eps=1e-3, momentum=.99))
        modules.append(GaussianNoise(std=noise))
        modules.append(nn.ReLU())
        modules.append(Reshape((-1,8,36,36)))
        modules.append(nn.Conv2d(8,8,kernel_size=11, bias=bias))
        modules.append(Flatten())
        modules.append(nn.BatchNorm1d(8*26*26, eps=1e-3, momentum=.99))
        modules.append(GaussianNoise(std=noise))
        modules.append(nn.ReLU())
        self.features = nn.Sequential(*modules)

        # Create output layers
        self.output_layers = nn.ModuleList([])
        for n_units in output_units:
            modules = []
            modules.append(nn.Linear(8*26*26, n_units, bias=bias))
            modules.append(nn.BatchNorm1d(n_units))
            modules.append(nn.Softplus())
            self.output_layers.append(nn.Sequential(*modules))
        
    def forward(self, x, output_idx):
        feats = self.features(x)
        return self.output_layers[output_idx](feats)

    def calc_grad(self, mode):
        """
        If mode is false, gradients are not calculated. If mode is true gradients are calculated.

        mode : bool
        """
        for p in self.parameters():
            try:
                p.requires_grad = mode
            except:
                pass
 
class PracticalBNCNN(nn.Module):
    def __init__(self, output_units=5, noise=.1, bias=True, chans=[8,8]):
        super(PracticalBNCNN,self).__init__()
        self.chans=chans
        modules = []
        modules.append(nn.Conv2d(40,chans[0],kernel_size=15, bias=bias))
        modules.append(nn.Dropout(p=noise/2))
        modules.append(nn.ReLU())
        modules.append(Flatten())
        modules.append(nn.BatchNorm1d(chans[0]*36*36, eps=1e-3))
        modules.append(Reshape((-1,chans[0],36,36)))
        modules.append(nn.Conv2d(chans[0],chans[1],kernel_size=11, bias=bias))
        modules.append(nn.Dropout(p=noise))
        modules.append(nn.ReLU())
        modules.append(Flatten())
        modules.append(nn.BatchNorm1d(chans[1]*26*26, eps=1e-3))
        modules.append(nn.Linear(chans[1]*26*26,output_units, bias=bias))
        modules.append(nn.BatchNorm1d(output_units))
        modules.append(nn.Softplus())
        self.sequential = nn.Sequential(*modules)
        
    def forward(self, x):
        return self.sequential(x)

class PracticalBNCNN2D(nn.Module):
    def __init__(self, output_units=5, noise=.3, bias=True, chans=[8,8]):
        super(PracticalBNCNN2D,self).__init__()
        self.name = 'McNiruNet'
        self.chans=[8,8]
        modules = []
        modules.append(nn.Conv2d(40,chans[0],kernel_size=15, bias=bias))
        modules.append(nn.Dropout(p=noise/2))
        modules.append(nn.ReLU())
        modules.append(nn.BatchNorm2d(chans[0], eps=1e-3, momentum=.99))

        modules.append(nn.Conv2d(chans[0],chans[1],kernel_size=11, bias=bias))
        modules.append(nn.Dropout(p=noise))
        modules.append(nn.ReLU())
        modules.append(nn.BatchNorm2d(chans[1], eps=1e-3, momentum=.99))

        modules.append(Flatten())
        modules.append(nn.Linear(chans[1]*26*26,output_units, bias=bias))
        modules.append(nn.BatchNorm1d(output_units))
        modules.append(nn.Softplus())
        self.sequential = nn.Sequential(*modules)
        
    def forward(self, x):
        return self.sequential(x)

class SkipBNBNCNN(nn.Module):
    def __init__(self, output_units=5, noise=.05, bias=True, x_shape=(40,50,50), chans=[8,8]):
        super(SkipBNBNCNN,self).__init__()
        self.name = 'SkipNet'
        self.chans = chans
        modules = []
        #in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True
        modules.append(SkipConnectionBN(40,2,15,x_shape=x_shape,bias=bias, noise=noise))
        modules.append(nn.Conv2d(48,8,kernel_size=15, bias=bias))
        modules.append(Flatten())
        modules.append(nn.BatchNorm1d(8*36*36, eps=1e-3, momentum=.99))
        modules.append(GaussianNoise(std=noise))
        modules.append(nn.ReLU())
        modules.append(Reshape((-1,8,36,36)))
        modules.append(nn.Conv2d(8,8,kernel_size=11, bias=bias))
        modules.append(Flatten())
        modules.append(nn.BatchNorm1d(8*26*26, eps=1e-3, momentum=.99))
        modules.append(GaussianNoise(std=noise))
        modules.append(nn.ReLU())
        modules.append(nn.Linear(8*26*26,output_units, bias=bias))
        modules.append(nn.BatchNorm1d(output_units))
        modules.append(nn.Softplus())
        self.sequential = nn.Sequential(*modules)
        
    def forward(self, x):
        return self.sequential(x)

class SSCNN(nn.Module):
    def __init__(self, output_units, scale=True, shift=False, bias=True, noise=0.05, chans=[8,8]):
        super(SSCNN,self).__init__()
        self.name = 'McNiruNet'
        self.chans=chans
        modules = []
        modules.append(nn.Conv2d(40,chans[0],kernel_size=15, bias=bias))
        modules.append(ScaleShift((chans[0],36,36)))
        modules.append(GaussianNoise(std=noise))
        modules.append(nn.ReLU())
        modules.append(nn.Conv2d(chans[0],chans[1],kernel_size=11, bias=bias))
        modules.append(ScaleShift((chans[1],26,26)))
        modules.append(GaussianNoise(std=noise))
        modules.append(nn.ReLU())
        modules.append(Flatten())
        modules.append(nn.Linear(chans[1]*26*26,output_units, bias=bias))
        modules.append(ScaleShift(output_units))
        modules.append(nn.Softplus())
        self.sequential = nn.Sequential(*modules)
        
    def forward(self, x):
        return self.sequential(x)

class StackedBNCNN(nn.Module):
    def __init__(self, output_units=5, noise=.05, bias=True, adapt_gauss=False, chans=[8,8]):
        super(StackedBNCNN,self).__init__()
        self.name = 'StackedNet'
        self.chans = chans
        module_list = []
        module_list.append(StackedConv2d(40,chans[0],kernel_size=15))
        module_list.append(Flatten())
        module_list.append(nn.BatchNorm1d(chans[0]*36*36, eps=1e-3, momentum=.99))
        module_list.append(GaussianNoise(std=noise, adapt=adapt_gauss))
        module_list.append(nn.ReLU())
        module_list.append(Reshape((-1,chans[0],36,36)))
        module_list.append(StackedConv2d(chans[0],chans[1],kernel_size=11))
        module_list.append(Flatten())
        module_list.append(nn.BatchNorm1d(chans[1]*26*26, eps=1e-3, momentum=.99))
        module_list.append(GaussianNoise(std=noise, adapt=adapt_gauss))
        module_list.append(nn.ReLU())
        module_list.append(nn.Linear(chans[1]*26*26,output_units, bias=False))
        module_list.append(nn.BatchNorm1d(output_units, eps=1e-3, momentum=.99))
        module_list.append(nn.Softplus())
        self.sequential = nn.Sequential(*module_list)

    def forward(self, x):
        return self.sequential(x)
    
