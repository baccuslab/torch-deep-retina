import numpy as np
import sys
sys.path.insert(0, '/home/xhding/workspaces/torch-deep-retina/')
import torch
import torch.nn as nn
from torchdeepretina.torch_utils import *

class BNCNN_3D(nn.Module):
    def __init__(self, n_units=5, noise=.05, chans=[8,8], bn_moment=0.01, softplus=True, 
                 img_shape=[80,50,50], ksizes=([40,15,15],[3,11,11]), strides=([5,1,1],[1,1,1])):
        super(BNCNN_3D, self).__init__()
        self.n_units = n_units
        self.noise = noise
        self.chans = chans
        self.bn_moment = bn_moment
        self.softplus = softplus
        self.image_shape = img_shape
        self.ksizes = [np.array(ksize) for ksize in ksizes]
        self.strides = [np.array(stride) for stride in strides]
        
        shape = np.array(self.image_shape)
        
        modules = []
        modules.append(Reshape((-1, 1, self.image_shape[0], 
                                self.image_shape[1], self.image_shape[2])))
        modules.append(nn.Conv3d(in_channels=1, out_channels=self.chans[0],
                                 kernel_size=tuple(self.ksizes[0]), stride=tuple(self.strides[0])))
        shape = update_shape(shape, self.ksizes[0], np.zeros(3), self.strides[0])
        modules.append(Flatten())
        modules.append(nn.BatchNorm1d(self.chans[0]*shape[0]*shape[1]*shape[2]))
        modules.append(GaussianNoise(std=self.noise))
        modules.append(nn.ReLU())
        self.bipolar = nn.Sequential(*modules)
        
        modules = []
        modules.append(Reshape((-1, self.chans[0], shape[0], shape[1], shape[2])))
        modules.append(nn.Conv3d(in_channels=self.chans[0], out_channels=self.chans[1], 
                                 kernel_size=tuple(self.ksizes[1]), stride=tuple(self.strides[1])))
        shape = update_shape(shape, self.ksizes[1], np.zeros(3), self.strides[1])
        modules.append(Flatten())
        modules.append(nn.BatchNorm1d(self.chans[1]*shape[0]*shape[1]*shape[2]))
        modules.append(GaussianNoise(std=self.noise))
        modules.append(nn.ReLU())
        self.amacrine = nn.Sequential(*modules)
        
        modules = []
        modules.append(Flatten())
        modules.append(nn.Linear(self.chans[1]*shape[0]*shape[1]*shape[2], 
                                 self.n_units, bias=False))
        modules.append(nn.BatchNorm1d(self.n_units))
        modules.append(nn.Softplus())
        self.ganglion = nn.Sequential(*modules)
        
    def forward(self, x):
        out = self.bipolar(x)
        out = self.amacrine(out)
        out = self.ganglion(out)
        return out
    
class BN_CNN_Net(nn.Module):
    def __init__(self, n_units=5, noise=.05, chans=[8,8], bn_moment=0.01, softplus=True, 
                 img_shape=(40,50,50), ksizes=(15,11)):
        super(BN_CNN_Net,self).__init__()
        self.n_units = n_units
        self.noise = noise
        self.chans = chans
        self.bn_moment = bn_moment
        self.softplus = softplus
        self.image_shape = img_shape
        self.ksizes = ksizes
        
        shape = self.image_shape[1:]
        
        modules = []
        modules.append(nn.Conv2d(self.image_shape[0],self.chans[0],kernel_size=self.ksizes[0]))
        shape = update_shape(shape, self.ksizes[0])
        modules.append(Flatten())
        modules.append(nn.BatchNorm1d(self.chans[0]*shape[0]*shape[1], momentum=self.bn_moment))
        modules.append(GaussianNoise(std=self.noise))
        modules.append(nn.ReLU())
        self.bipolar = nn.Sequential(*modules)
        
        modules = []
        modules.append(Reshape((-1, self.chans[0], shape[0], shape[1])))
        modules.append(nn.Conv2d(self.chans[0],self.chans[1],kernel_size=self.ksizes[1]))
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
    
class BNCNN_3D2(nn.Module):
    def __init__(self, n_units=5, noise=.05, chans=[8,8], bn_moment=0.01, softplus=True, 
                 img_shape=[50,50,50], ksizes=([40,15,15],[1,11,11]), strides=([1,1,1],[1,1,1])):
        super(BNCNN_3D2, self).__init__()
        self.n_units = n_units
        self.noise = noise
        self.chans = chans
        self.bn_moment = bn_moment
        self.softplus = softplus
        self.image_shape = img_shape
        self.ksizes = [np.array(ksize) for ksize in ksizes]
        self.strides = [np.array(stride) for stride in strides]
        
        shape = np.array(self.image_shape)
        
        modules = []
        modules.append(Reshape((-1, 1, self.image_shape[0], 
                                self.image_shape[1], self.image_shape[2])))
        modules.append(nn.Conv3d(in_channels=1, out_channels=self.chans[0],
                                 kernel_size=tuple(self.ksizes[0]), stride=tuple(self.strides[0])))
        shape = update_shape(shape, self.ksizes[0], np.zeros(3), self.strides[0])
        modules.append(Flatten())
        modules.append(nn.BatchNorm1d(self.chans[0]*shape[0]*shape[1]*shape[2]))
        modules.append(GaussianNoise(std=self.noise))
        modules.append(nn.ReLU())
        self.bipolar = nn.Sequential(*modules)
        
        modules = []
        modules.append(Reshape((-1, self.chans[0], shape[0], shape[1], shape[2])))
        modules.append(nn.Conv3d(in_channels=self.chans[0], out_channels=self.chans[1], 
                                 kernel_size=tuple(self.ksizes[1]), stride=tuple(self.strides[1])))
        shape = update_shape(shape, self.ksizes[1], np.zeros(3), self.strides[1])
        modules.append(Flatten())
        modules.append(nn.BatchNorm1d(self.chans[1]*shape[0]*shape[1]*shape[2]))
        modules.append(GaussianNoise(std=self.noise))
        modules.append(nn.ReLU())
        self.amacrine = nn.Sequential(*modules)
        
        modules = []
        modules.append(Reshape((-1, self.chans[1], shape[0], shape[1], shape[2])))
        modules.append(Temperal_Filter(shape[0], 2))
        modules.append(Flatten())
        modules.append(nn.Linear(self.chans[1]*shape[1]*shape[2], 
                                 self.n_units, bias=False))
        modules.append(nn.BatchNorm1d(self.n_units))
        modules.append(nn.Softplus())
        self.ganglion = nn.Sequential(*modules)
        
    def forward(self, x):
        out = self.bipolar(x)
        out = self.amacrine(out)
        out = self.ganglion(out)
        return out