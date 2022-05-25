import numpy as np
import torch
import torch.nn as nn
from fnn.custom_modules import *
from torchdeepretina.torch_utils import *



class BN_CNN_Stack_Old(nn.Module):
    def __init__(self, n_units=5, noise=.05, chans=[8,8], bn_moment=0.01, softplus=True, 
                 img_shape=(40,50,50), ksizes=(15,11), **kwargs):
        super().__init__()
        self.n_units = n_units
        self.noise = noise
        self.chans = chans
        self.bn_moment = bn_moment
        self.softplus = softplus
        self.image_shape = img_shape
        self.ksizes = ksizes
        self.img_shape = img_shape
        
        shape = self.image_shape[1:]
        
        modules = []
        modules.append(LinearStackedConv2d(self.img_shape[0], self.chans[0],
                                           kernel_size=self.ksizes[0], conv_bias=False))
        shape = update_shape(shape, self.ksizes[0])
        modules.append(Flatten())
        modules.append(nn.BatchNorm1d(self.chans[0]*shape[0]*shape[1], momentum=self.bn_moment))
        modules.append(GaussianNoise(std=self.noise))
        modules.append(nn.ReLU())
        self.bipolar = nn.Sequential(*modules)
        
        modules = []
        modules.append(Reshape((-1, self.chans[0], shape[0], shape[1])))
        modules.append(LinearStackedConv2d(self.chans[0], self.chans[1],
                                           kernel_size=self.ksizes[1], conv_bias=False))
        shape = update_shape(shape, self.ksizes[1])
        modules.append(Flatten())
        modules.append(nn.BatchNorm1d(self.chans[1]*shape[0]*shape[1], momentum=self.bn_moment))
        modules.append(GaussianNoise(std=self.noise))
        modules.append(nn.ReLU())
        self.amacrine = nn.Sequential(*modules)
        
        modules = []
        modules.append(nn.Linear(self.chans[1]*shape[0]*shape[1], self.n_units, bias=False))
        modules.append(nn.BatchNorm1d(self.n_units, momentum=self.bn_moment))
        modules.append(nn.Softplus())
        self.ganglion = nn.Sequential(*modules)

    def forward(self,x):
        out = self.bipolar(x)
        out = self.amacrine(out)
        out = self.ganglion(out)
        return out
    
class BN_CNN_Stack(nn.Module):
    def __init__(self, n_units=5, bias=True, linear_bias=True, chans=[8,8], bn_moment=0.01, 
                 img_shape=(40,50,50), ksizes=(15,11), **kwargs):
        super().__init__()
        self.n_units = n_units
        self.chans = chans
        self.bn_moment = bn_moment
        self.image_shape = img_shape
        self.ksizes = ksizes
        self.img_shape = img_shape
        
        shape = self.image_shape[1:]
        
        modules = []
        modules.append(LinearStackedConv2d(self.img_shape[0], self.chans[0], bias=bias, kernel_size=self.ksizes[0]))
        shape = update_shape(shape, self.ksizes[0])
        modules.append(Flatten())
        modules.append(nn.BatchNorm1d(self.chans[0]*shape[0]*shape[1], momentum=self.bn_moment))
        modules.append(nn.ReLU())
        self.bipolar = nn.Sequential(*modules)
        
        modules = []
        modules.append(Reshape((-1, self.chans[0], shape[0], shape[1])))
        modules.append(LinearStackedConv2d(self.chans[0], self.chans[1], bias=bias, kernel_size=self.ksizes[1]))
        shape = update_shape(shape, self.ksizes[1])
        modules.append(Flatten())
        modules.append(nn.BatchNorm1d(self.chans[1]*shape[0]*shape[1], momentum=self.bn_moment))
        modules.append(nn.ReLU())
        self.amacrine = nn.Sequential(*modules)
        
        modules = []
        modules.append(nn.Linear(self.chans[1]*shape[0]*shape[1], self.n_units, bias=linear_bias))
        modules.append(nn.BatchNorm1d(self.n_units, momentum=self.bn_moment))
        modules.append(nn.Softplus())
        self.ganglion = nn.Sequential(*modules)

    def forward(self,x):
        out = self.bipolar(x)
        out = self.amacrine(out)
        out = self.ganglion(out)
        return out
    
class BN_CNN_Stack_NoNorm(nn.Module):
    def __init__(self, n_units=5, bias=True, linear_bias=True, chans=[8,8],
                 img_shape=(40,50,50), ksizes=(15,11), **kwargs):
        super().__init__()
        self.n_units = n_units
        self.chans = chans
        self.image_shape = img_shape
        self.ksizes = ksizes
        self.img_shape = img_shape
        
        shape = self.image_shape[1:]
        
        modules = []
        modules.append(LinearStackedConv2d(self.img_shape[0], self.chans[0], bias=bias, kernel_size=self.ksizes[0]))
        shape = update_shape(shape, self.ksizes[0])
        modules.append(nn.ReLU())
        self.bipolar = nn.Sequential(*modules)
        
        modules = []
        modules.append(LinearStackedConv2d(self.chans[0], self.chans[1], bias=bias, kernel_size=self.ksizes[1]))
        shape = update_shape(shape, self.ksizes[1])
        modules.append(Flatten())
        modules.append(nn.ReLU())
        self.amacrine = nn.Sequential(*modules)
        
        modules = []
        modules.append(nn.Linear(self.chans[1]*shape[0]*shape[1], self.n_units, bias=linear_bias))
        modules.append(nn.Softplus())
        self.ganglion = nn.Sequential(*modules)

    def forward(self,x):
        out = self.bipolar(x)
        out = self.amacrine(out)
        out = self.ganglion(out)
        return out
    
class BN_CNN_Stack_poly(nn.Module):
    def __init__(self, n_units=5, noise=.05, chans=[8,8], bn_moment=0.01, softplus=True, 
                 img_shape=(40,50,50), ksizes=(15,11)):
        super().__init__()
        self.n_units = n_units
        self.noise = noise
        self.chans = chans
        self.bn_moment = bn_moment
        self.softplus = softplus
        self.image_shape = img_shape
        self.ksizes = ksizes
        self.img_shape = img_shape
        
        shape = self.image_shape[1:]
        
        modules = []
        modules.append(LinearStackedConv2d(self.img_shape[0], self.chans[0],
                                           kernel_size=self.ksizes[0]))
        shape = update_shape(shape, self.ksizes[0])
        modules.append(Flatten())
        modules.append(nn.BatchNorm1d(self.chans[0]*shape[0]*shape[1], momentum=self.bn_moment))
        modules.append(GaussianNoise(std=self.noise))
        modules.append(nn.ReLU())
        self.bipolar = nn.Sequential(*modules)
        
        modules = []
        modules.append(Reshape((-1, self.chans[0], shape[0], shape[1])))
        modules.append(LinearStackedConv2d(self.chans[0], self.chans[1],
                                           kernel_size=self.ksizes[1]))
        shape = update_shape(shape, self.ksizes[1])
        modules.append(Flatten())
        modules.append(nn.BatchNorm1d(self.chans[1]*shape[0]*shape[1], momentum=self.bn_moment))
        modules.append(GaussianNoise(std=self.noise))
        modules.append(nn.ReLU())
        self.amacrine = nn.Sequential(*modules)
        
        modules = []
        modules.append(nn.Linear(self.chans[1]*shape[0]*shape[1], self.n_units, bias=False))
        modules.append(nn.BatchNorm1d(self.n_units, momentum=self.bn_moment))
        modules.append(nn.Softplus())
        self.ganglion = nn.Sequential(*modules)
        
        self.poly_coef = nn.Parameter(torch.randn(5, self.n_units, requires_grad=True)/10)
        self.poly_coef.data[-2,:] = torch.ones(self.n_units)

    def forward(self,x):
        out = self.bipolar(x)
        out = self.amacrine(out)
        out = self.ganglion(out)
        out = out/100
        out = self.poly_coef[0]*out**4+self.poly_coef[1]*out**3+self.poly_coef[2]*out**2+self.poly_coef[3]*out+self.poly_coef[4]
        out = out*100
        return out
    
class BN_CNN_Stack_FC(nn.Module):
    def __init__(self, n_units=5, bias=True, linear_bias=True, chans=[8,8], bn_moment=0.01, 
                 img_shape=(40,50,50), ksizes=(15,11,9), **kwargs):
        super().__init__()
        self.n_units = n_units
        self.chans = chans
        self.bn_moment = bn_moment
        self.image_shape = img_shape
        self.ksizes = ksizes
        self.img_shape = img_shape
        
        shape = self.image_shape[1:]
        
        modules = []
        modules.append(LinearStackedConv2d(self.img_shape[0], self.chans[0], bias=bias, kernel_size=self.ksizes[0]))
        shape = update_shape(shape, self.ksizes[0])
        modules.append(Flatten())
        modules.append(nn.BatchNorm1d(self.chans[0]*shape[0]*shape[1], momentum=self.bn_moment))
        modules.append(nn.ReLU())
        self.bipolar = nn.Sequential(*modules)
        
        modules = []
        modules.append(Reshape((-1, self.chans[0], shape[0], shape[1])))
        modules.append(LinearStackedConv2d(self.chans[0], self.chans[1], bias=bias, kernel_size=self.ksizes[1]))
        shape = update_shape(shape, self.ksizes[1])
        modules.append(Flatten())
        modules.append(nn.BatchNorm1d(self.chans[1]*shape[0]*shape[1], momentum=self.bn_moment))
        modules.append(nn.ReLU())
        self.amacrine = nn.Sequential(*modules)
        
        modules = []
        modules.append(Reshape((-1, self.chans[1], shape[0], shape[1])))
        shape = update_shape(shape, self.ksizes[2])
        #modules.append(nn.Conv2d(self.chans[1], self.n_units, bias=linear_bias, kernel_size=self.ksizes[2]))
        modules.append(LinearStackedConv2d(self.chans[1], self.n_units, bias=linear_bias, kernel_size=self.ksizes[2]))
        modules.append(Flatten())
        modules.append(nn.BatchNorm1d(self.n_units*shape[0]*shape[1], momentum=self.bn_moment))
        modules.append(Reshape((-1, self.n_units, shape[0], shape[1])))
        modules.append(nn.Softplus())
        modules.append(OneHot((self.n_units,*shape)))
        self.ganglion = nn.Sequential(*modules)

    def forward(self,x):
        out = self.bipolar(x)
        out = self.amacrine(out)
        out = self.ganglion(out)
        #out = torch.exp(out)
        return out
    
class BN_CNN_Stack_NoNorm_FC(nn.Module):
    def __init__(self, n_units=5, bias=True, linear_bias=True, chans=[8,8],
                 img_shape=(40,50,50), ksizes=(15,11,9), **kwargs):
        super().__init__()
        self.n_units = n_units
        self.chans = chans
        self.image_shape = img_shape
        self.ksizes = ksizes
        self.img_shape = img_shape
        
        shape = self.image_shape[1:]
        
        modules = []
        modules.append(LinearStackedConv2d(self.img_shape[0], self.chans[0], bias=bias, kernel_size=self.ksizes[0]))
        shape = update_shape(shape, self.ksizes[0])
        modules.append(nn.ReLU())
        self.bipolar = nn.Sequential(*modules)
        
        modules = []
        modules.append(LinearStackedConv2d(self.chans[0], self.chans[1], bias=bias, kernel_size=self.ksizes[1]))
        shape = update_shape(shape, self.ksizes[1])
        modules.append(nn.ReLU())
        self.amacrine = nn.Sequential(*modules)
        
        modules = []
        shape = update_shape(shape, self.ksizes[2])
        modules.append(nn.Conv2d(self.chans[1], self.n_units, bias=linear_bias, kernel_size=self.ksizes[2]))
        modules.append(nn.Softplus())
        modules.append(OneHot((self.n_units,*shape)))
        self.ganglion = nn.Sequential(*modules)

    def forward(self,x):
        out = self.bipolar(x)
        out = self.amacrine(out)
        out = self.ganglion(out)
        out = torch.exp(out)
        return out