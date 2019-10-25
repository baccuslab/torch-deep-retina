import numpy as np
import sys
sys.path.insert(0, '/home/xhding/workspaces/torch-deep-retina/')
import torch
import torch.nn as nn
from torchdeepretina.torch_utils import *

class BNCNN_3D(nn.Module):
    def __init__(self, noise):
        super(BNCNN_3D, self).__init__()
        self.noise = noise
        
        modules = []
        modules.append(nn.Conv3d(in_channels=1, out_channels=8, kernel_size=(40,15,15),
                                 stride=(8,1,1)))
        modules.append(Flatten())
        modules.append(nn.Batchnorm1d(8*6*36*36))
        modules.append(GaussianNoise(std=self.noise))
        modules.append(nn.ReLU())
        self.bipolar = nn.Sequential(*modules)
        
        modules = []
        modules.append(nn.Conv3d(in_channels=8, out_channels=8, kernel_size=(1,11,11),
                                 stride=(1,1,1)))
        modules.append(Flatten())
        modules.append(nn.Batchnorm1d(8*6*26*26))
        modules.append(GaussianNoise(std=self.noise))
        modules.append(nn.ReLU())
        self.amacrine = nn.Sequential(*modules)
        
        modules = []
        modules.append(Flatten())
        modules.append(nn.Linear(8*6*26*26, 5, bias=False))
        modules.append(nn.Batchnorm1d(5))
        modules.append(nn.Softplus())
        self.ganglion = nn.Sequential(*modules)
        
    def forward(self, x):
        out = self.bipolar(x)
        out = self.amacrine(out)
        out = self.gangion(out)
        return out
    
class KineticsModel(nn.Module):
    def __init__(self, bnorm=True, drop_p=0, scale_kinet=False, recur_seq_len=5, n_units=5, 
                 noise=.05, bias=True, linear_bias=False, chans=[8,8], bn_moment=.01, softplus=True, 
                 inference_exp=False, img_shape=(40,50,50), ksizes=(15,11), centers=None):
        super(KineticsModel, self).__init__()
        
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
        self.centers = centers
        
        self.bnorm = bnorm
        self.drop_p = drop_p
        self.scale_kinet = scale_kinet
        self.seq_len = recur_seq_len
        shape = self.img_shape[1:] # (H, W)
        self.shapes = []
        self.h_shapes = []

        modules = []
        modules.append(LinearStackedConv2d(self.img_shape[0],self.chans[0],
                                           kernel_size=self.ksizes[0], abs_bnorm=False, 
                                           bias=self.bias, drop_p=self.drop_p))
        shape = update_shape(shape, self.ksizes[0])
        self.shapes.append(tuple(shape))
        n_states = 4
        self.h_shapes.append((n_states, self.chans[0]*shape[0]*shape[1]))
        self.h_shapes.append((self.chans[0]*shape[0]*shape[1],))
        modules.append(Flatten())
        modules.append(AbsBatchNorm1d(self.chans[0]*shape[0]*shape[1], eps=1e-3, momentum=self.bn_moment))
        modules.append(GaussianNoise(std=self.noise))
        #modules.append(Add(-1.5))
        modules.append(nn.Softplus())
        max_clamp = 10
        modules.append(Clamp(0,max_clamp))
        modules.append(Multiply(1/max_clamp))
        self.bipolar = nn.Sequential(*modules)

        self.kinetics = Kinetics()
        if scale_kinet:
            self.kinet_scale = ScaleShift(self.seq_len*self.chans[0]*shape[0]*shape[1])

        modules = []
        modules.append(Reshape((-1,self.seq_len*self.chans[0], shape[0], shape[1])))
        modules.append(LinearStackedConv2d(self.seq_len*self.chans[0],self.chans[1],
                                           kernel_size=self.ksizes[1], abs_bnorm=False, 
                                           bias=self.bias, drop_p=self.drop_p))
        shape = update_shape(shape, self.ksizes[1])
        self.shapes.append(tuple(shape))
        modules.append(Flatten())
        modules.append(AbsBatchNorm1d(self.chans[1]*shape[0]*shape[1], eps=1e-3, 
                                      momentum=self.bn_moment))
        modules.append(GaussianNoise(std=self.noise))
        modules.append(nn.ReLU())
        self.amacrine = nn.Sequential(*modules)

        modules = []
        modules.append(nn.Linear(self.chans[1]*shape[0]*shape[1], 
                                 self.n_units, bias=self.linear_bias))
        modules.append(AbsBatchNorm1d(self.n_units, eps=1e-3, momentum=self.bn_moment))
        if self.softplus:
            modules.append(nn.Softplus())
        else:
            modules.append(Exponential(train_off=True))
        self.ganglion = nn.Sequential(*modules)

    def forward(self, x, hs):
        """
        x - FloatTensor (B, C, H, W)
        hs - list [(B,S,N),(D,B,N)]
            First list element should be a torch FloatTensor of state population values.
            Second element should be deque of activated population values over past D time steps
        """
        fx = self.bipolar(x)
        fx, h0 = self.kinetics(fx, hs[0])
        hs[1].append(fx)
        h1 = hs[1]
        fx = torch.cat(list(h1), dim=1) #(B,D*N)
        if self.scale_kinet:
            fx = self.kinet_scale(fx)
        fx = self.amacrine(fx)
        fx = self.ganglion(fx)
        if not self.training and self.infr_exp:
            fx = torch.exp(fx)
        return fx, [h0, h1]