import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchdeepretina.torch_utils import *
    
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
        modules.append(nn.BatchNorm1d(self.chans[0]*shape[0]*shape[1], momentum=self.bn_moment))
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
        modules.append(nn.BatchNorm1d(self.chans[1]*shape[0]*shape[1], momentum=self.bn_moment))
        modules.append(GaussianNoise(std=self.noise))
        modules.append(nn.ReLU())
        self.amacrine = nn.Sequential(*modules)

        modules = []
        modules.append(nn.Linear(self.chans[1]*shape[0]*shape[1], 
                                 self.n_units, bias=self.linear_bias))
        modules.append(nn.BatchNorm1d(self.n_units, momentum=self.bn_moment))
        modules.append(nn.Softplus())
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
        return fx, [h0, h1]
    
class KineticsChannelModel(nn.Module):
    def __init__(self, bnorm=True, drop_p=0, scale_kinet=False, recur_seq_len=5, n_units=5, 
                 noise=.05, bias=True, linear_bias=False, chans=[8,8], bn_moment=.01, softplus=True, 
                 inference_exp=False, img_shape=(40,50,50), ksizes=(15,11), centers=None):
        super(KineticsChannelModel, self).__init__()
        
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
        self.h_shapes.append((n_states, self.chans[0], shape[0]*shape[1]))
        self.h_shapes.append((self.chans[0], shape[0]*shape[1]))
        modules.append(Flatten())
        modules.append(nn.BatchNorm1d(self.chans[0]*shape[0]*shape[1], momentum=self.bn_moment))
        modules.append(GaussianNoise(std=self.noise))     
        #modules.append(Add(-1.5))
        modules.append(nn.Softplus())
        max_clamp = 10
        modules.append(Clamp(0,max_clamp))
        modules.append(Multiply(1/max_clamp))
        modules.append(Reshape((-1, self.chans[0], shape[0] * shape[1])))
        self.bipolar = nn.Sequential(*modules)

        self.kinetics = Kinetics_channel(chan=self.chans[0])
        
        if scale_kinet:
            self.kinet_scale = ScaleShift((self.seq_len*self.chans[0], shape[0]*shape[1]))

        modules = []
        modules.append(Reshape((-1,self.seq_len*self.chans[0], shape[0], shape[1])))
        modules.append(LinearStackedConv2d(self.seq_len*self.chans[0],self.chans[1],
                                           kernel_size=self.ksizes[1], abs_bnorm=False, 
                                           bias=self.bias, drop_p=self.drop_p))
        shape = update_shape(shape, self.ksizes[1])
        self.shapes.append(tuple(shape))
        modules.append(Flatten())
        modules.append(nn.BatchNorm1d(self.chans[1]*shape[0]*shape[1], momentum=self.bn_moment))
        modules.append(GaussianNoise(std=self.noise))
        modules.append(nn.ReLU())
        self.amacrine = nn.Sequential(*modules)

        modules = []
        modules.append(nn.Linear(self.chans[1]*shape[0]*shape[1], 
                                 self.n_units, bias=self.linear_bias))
        modules.append(nn.BatchNorm1d(self.n_units, momentum=self.bn_moment))
        modules.append(nn.Softplus())
        self.ganglion = nn.Sequential(*modules)

    def forward(self, x, hs):
        """
        x - FloatTensor (B, C, H, W)
        hs - list [(B,S,C,N),(D,B,C,N)]
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
        return fx, [h0, h1]
    
class KineticsChannelModelFilter(nn.Module):
    def __init__(self, bnorm=True, drop_p=0, scale_kinet=False, recur_seq_len=5, n_units=5, 
                 noise=.05, bias=True, linear_bias=False, chans=[8,8], bn_moment=.01, softplus=True, 
                 inference_exp=False, img_shape=(40,50,50), ksizes=(15,11), centers=None):
        super(KineticsChannelModelFilter, self).__init__()
        
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
        self.h_shapes.append((n_states, self.chans[0], shape[0]*shape[1]))
        self.h_shapes.append((self.chans[0], shape[0]*shape[1]))
        modules.append(Flatten())
        modules.append(nn.BatchNorm1d(self.chans[0]*shape[0]*shape[1], momentum=self.bn_moment))
        modules.append(GaussianNoise(std=self.noise))     
        #modules.append(Add(-1.5))
        modules.append(nn.Softplus())
        max_clamp = 10
        modules.append(Clamp(0,max_clamp))
        modules.append(Multiply(1/max_clamp))
        modules.append(Reshape((-1, self.chans[0], shape[0] * shape[1])))
        self.bipolar = nn.Sequential(*modules)

        self.kinetics = Kinetics_channel(chan=self.chans[0])
        
        if scale_kinet:
            self.kinet_scale = ScaleShift((self.seq_len, self.chans[0], shape[0]*shape[1]))

        modules = []
        modules.append(Reshape((-1,self.seq_len, self.chans[0], shape[0], shape[1])))
        modules.append(Temperal_Filter(self.seq_len, 3))
        modules.append(LinearStackedConv2d(self.chans[0],self.chans[1],
                                           kernel_size=self.ksizes[1], abs_bnorm=False, 
                                           bias=self.bias, drop_p=self.drop_p))
        shape = update_shape(shape, self.ksizes[1])
        self.shapes.append(tuple(shape))
        modules.append(Flatten())
        modules.append(nn.BatchNorm1d(self.chans[1]*shape[0]*shape[1], momentum=self.bn_moment))
        modules.append(GaussianNoise(std=self.noise))
        modules.append(nn.ReLU())
        self.amacrine = nn.Sequential(*modules)

        modules = []
        modules.append(nn.Linear(self.chans[1]*shape[0]*shape[1], 
                                 self.n_units, bias=self.linear_bias))
        modules.append(nn.BatchNorm1d(self.n_units, momentum=self.bn_moment))
        modules.append(nn.Softplus())
        self.ganglion = nn.Sequential(*modules)
        
    def forward(self, x, hs):
        """
        x - FloatTensor (B, C, H, W)
        hs - list [(B,S,C,N),(D,B,C,N)]
            First list element should be a torch FloatTensor of state population values.
            Second element should be deque of activated population values over past D time steps
        """
        fx = self.bipolar(x)
        fx, h0 = self.kinetics(fx, hs[0]) 
        hs[1].append(fx)
        h1 = hs[1]
        fx = torch.stack(list(h1), dim=1) #(B,D*N)
        if self.scale_kinet:
            fx = self.kinet_scale(fx)
        fx = self.amacrine(fx)
        fx = self.ganglion(fx)
        return fx, [h0, h1]
    
class KineticsChannelModelFilterBipolar(nn.Module):
    def __init__(self, bnorm=True, drop_p=0, scale_kinet=False, recur_seq_len=5, n_units=5, 
                 noise=.05, bias=True, linear_bias=False, chans=[8,8], bn_moment=.01, softplus=True, 
                 inference_exp=False, img_shape=(40,50,50), ksizes=(15,11), centers=None):
        super(KineticsChannelModelFilterBipolar, self).__init__()
        
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
        self.h_shapes.append((n_states, self.chans[0], shape[0]*shape[1]))
        self.h_shapes.append((self.chans[0], shape[0]*shape[1]))
        modules.append(Flatten())
        modules.append(nn.BatchNorm1d(self.chans[0]*shape[0]*shape[1], momentum=self.bn_moment))
        modules.append(GaussianNoise(std=self.noise))     
        modules.append(nn.Sigmoid())
        modules.append(Reshape((-1, self.chans[0], shape[0] * shape[1])))
        self.bipolar = nn.Sequential(*modules)

        self.kinetics = Kinetics_channel(chan=self.chans[0], dt=0.01)
        
        if scale_kinet:
            self.kinet_scale = ScaleShift((self.seq_len, self.chans[0], shape[0]*shape[1]))

        modules = []
        modules.append(Reshape((-1,self.seq_len, self.chans[0], shape[0], shape[1])))
        modules.append(Temperal_Filter(self.seq_len, 3))
        modules.append(LinearStackedConv2d(self.chans[0],self.chans[1],
                                           kernel_size=self.ksizes[1], abs_bnorm=False, 
                                           bias=self.bias, drop_p=self.drop_p))
        shape = update_shape(shape, self.ksizes[1])
        self.shapes.append(tuple(shape))
        modules.append(Flatten())
        modules.append(nn.BatchNorm1d(self.chans[1]*shape[0]*shape[1], momentum=self.bn_moment))
        modules.append(GaussianNoise(std=self.noise))
        modules.append(nn.ReLU())
        self.amacrine = nn.Sequential(*modules)

        modules = []
        modules.append(nn.Linear(self.chans[1]*shape[0]*shape[1], 
                                 self.n_units, bias=self.linear_bias))
        modules.append(nn.BatchNorm1d(self.n_units, momentum=self.bn_moment))
        modules.append(nn.Softplus())
        self.ganglion = nn.Sequential(*modules)
        
    def forward(self, x, hs):
        """
        x - FloatTensor (B, C, H, W)
        hs - list [(B,S,C,N),(D,B,C,N)]
            First list element should be a torch FloatTensor of state population values.
            Second element should be deque of activated population values over past D time steps
        """
        fx = self.bipolar(x)
        fx, h0 = self.kinetics(fx, hs[0]) 
        hs[1].append(fx)
        h1 = hs[1]
        fx = torch.stack(list(h1), dim=1) #(B,D*N)
        if self.scale_kinet:
            fx = self.kinet_scale(fx)
        fx = self.amacrine(fx)
        fx = self.ganglion(fx)
        return fx, [h0, h1]
    
class KineticsChannelModelFilterBipolarNoNorm(nn.Module):
    def __init__(self, bnorm=True, drop_p=0, scale_kinet=False, recur_seq_len=5, n_units=5, 
                 noise=.05, bias=True, linear_bias=False, chans=[8,8], softplus=True, 
                 inference_exp=False, img_shape=(40,50,50), ksizes=(15,11), centers=None):
        super(KineticsChannelModelFilterBipolarNoNorm, self).__init__()
        
        self.kinetic = True
        self.n_units = n_units
        self.chans = chans 
        self.softplus = softplus 
        self.infr_exp = inference_exp 
        self.bias = bias 
        self.img_shape = img_shape 
        self.ksizes = ksizes 
        self.linear_bias = linear_bias 
        self.noise = noise 
        self.centers = centers
        
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
        self.h_shapes.append((n_states, self.chans[0], shape[0]*shape[1]))
        self.h_shapes.append((self.chans[0], shape[0]*shape[1]))
        modules.append(GaussianNoise(std=self.noise))     
        modules.append(nn.Sigmoid())
        modules.append(Reshape((-1, self.chans[0], shape[0] * shape[1])))
        self.bipolar = nn.Sequential(*modules)

        self.kinetics = Kinetics_channel(chan=self.chans[0], dt=0.01)
        
        if scale_kinet:
            #self.kinet_scale = ScaleShift((self.seq_len, self.chans[0], shape[0]*shape[1]))
            self.kinet_scale = ScaleShift((self.seq_len, self.chans[0], 1), scale=True, shift=False)

        modules = []
        modules.append(Reshape((-1,self.seq_len, self.chans[0], shape[0], shape[1])))
        modules.append(Temperal_Filter(self.seq_len, 3))
        modules.append(LinearStackedConv2d(self.chans[0],self.chans[1],
                                           kernel_size=self.ksizes[1], abs_bnorm=False, 
                                           bias=self.bias, drop_p=self.drop_p))
        shape = update_shape(shape, self.ksizes[1])
        self.shapes.append(tuple(shape))
        modules.append(Flatten())
        modules.append(GaussianNoise(std=self.noise))
        modules.append(nn.ReLU())
        self.amacrine = nn.Sequential(*modules)

        modules = []
        modules.append(nn.Linear(self.chans[1]*shape[0]*shape[1], 
                                 self.n_units, bias=self.linear_bias))
        modules.append(nn.Softplus())
        self.ganglion = nn.Sequential(*modules)
        
    def forward(self, x, hs):
        """
        x - FloatTensor (B, C, H, W)
        hs - list [(B,S,C,N),(D,B,C,N)]
            First list element should be a torch FloatTensor of state population values.
            Second element should be deque of activated population values over past D time steps
        """
        fx = self.bipolar(x)
        fx, h0 = self.kinetics(fx, hs[0]) 
        hs[1].append(fx)
        h1 = hs[1]
        fx = torch.stack(list(h1), dim=1) #(B,D*N)
        if self.scale_kinet:
            fx = self.kinet_scale(fx)
        fx = self.amacrine(fx)
        fx = self.ganglion(fx)
        return fx, [h0, h1]
        
    
class KineticsChannelModelFilterAmacrine(nn.Module):
    def __init__(self, bnorm=True, drop_p=0, scale_kinet=False, recur_seq_len=5, n_units=5, 
                 noise=.05, bias=True, linear_bias=False, chans=[8,8], softplus=True, 
                 inference_exp=False, img_shape=(40,50,50), ksizes=(15,11), centers=None):
        super(KineticsChannelModelFilterAmacrine, self).__init__()
        
        self.n_units = n_units
        self.chans = chans 
        self.softplus = softplus 
        self.infr_exp = inference_exp 
        self.bias = bias 
        self.img_shape = img_shape 
        self.ksizes = ksizes 
        self.linear_bias = linear_bias 
        self.noise = noise
        self.centers = centers
        
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
        modules.append(GaussianNoise(std=self.noise))     
        modules.append(nn.ReLU())
        self.bipolar = nn.Sequential(*modules)

        modules = []
        modules.append(LinearStackedConv2d(self.chans[0],self.chans[1],
                                           kernel_size=self.ksizes[1], abs_bnorm=False, 
                                           bias=self.bias, drop_p=self.drop_p))
        shape = update_shape(shape, self.ksizes[1])
        self.shapes.append(tuple(shape))
        n_states = 4
        self.h_shapes.append((n_states, self.chans[1], shape[0]*shape[1]))
        self.h_shapes.append((self.chans[1], shape[0]*shape[1]))
        modules.append(GaussianNoise(std=self.noise))
        modules.append(nn.Sigmoid())
        modules.append(Reshape((-1, self.chans[1], shape[0] * shape[1])))
        self.amacrine = nn.Sequential(*modules)
        
        self.kinetics = Kinetics_channel(chan=self.chans[1], dt=0.01)
        if scale_kinet:
            self.kinet_scale = ScaleShift((self.seq_len, self.chans[1], shape[0]*shape[1]))

        modules = []
        modules.append(Reshape((-1, self.seq_len, self.chans[1] * shape[0] * shape[1])))
        modules.append(Temperal_Filter(self.seq_len, 1))
        modules.append(nn.Linear(self.chans[1]*shape[0]*shape[1], 
                                 self.n_units, bias=self.linear_bias))
        modules.append(nn.Softplus())
        self.ganglion = nn.Sequential(*modules)

    def forward(self, x, hs):
        """
        x - FloatTensor (B, C, H, W)
        hs - list [(B,S,C,N),(D,B,C,N)]
            First list element should be a torch FloatTensor of state population values.
            Second element should be deque of activated population values over past D time steps
        """
        fx = self.bipolar(x)
        fx = self.amacrine(fx)
        fx, h0 = self.kinetics(fx, hs[0]) 
        hs[1].append(fx)
        h1 = hs[1]
        fx = torch.stack(list(h1), dim=1) #(B,D,N)
        if self.scale_kinet:
            fx = self.kinet_scale(fx)
        fx = self.ganglion(fx)
        return fx, [h0, h1]

    
class KineticsOnePixelChannel(nn.Module):
    def __init__(self, recur_seq_len=5, n_units=5, dt=0.01, scale_kinet=False,
                 bias=True, linear_bias=False, chans=[8,8], softplus=True, img_shape=(40,)):
        super(KineticsOnePixelChannel, self).__init__()
        
        self.n_units = n_units
        self.chans = chans 
        self.softplus = softplus 
        self.bias = bias 
        self.img_shape = img_shape 
        self.linear_bias = linear_bias 
        self.dt = dt
        self.seq_len = recur_seq_len
        self.h_shapes = []
        self.scale_kinet = scale_kinet

        self.bipolar_weight = nn.Parameter(torch.rand(self.chans[0], self.img_shape[0]))
        self.bipolar_bias = nn.Parameter(torch.rand(self.chans[0]))

        self.kinetics = Kinetics_channel(chan=self.chans[0], dt=self.dt)
        
        if scale_kinet:
            self.kinet_scale = ScaleShift((self.seq_len, self.chans[0], 1))

        self.amacrine_filter = Temperal_Filter(self.seq_len, 2)
        self.amacrine_weight = nn.Parameter(torch.rand(self.chans[1], self.chans[0]))
        self.amacrine_bias = nn.Parameter(torch.rand(self.chans[1]))

        modules = []
        modules.append(nn.Linear(self.chans[1], self.n_units, bias=self.linear_bias))
        modules.append(nn.Softplus())
        self.ganglion = nn.Sequential(*modules)
        
        n_states = 4
        self.h_shapes.append((n_states, self.chans[0], 1))
        self.h_shapes.append((self.chans[0], 1))
        
    def forward(self, x, hs):
        """
        x - FloatTensor (B, C)
        hs - list [(B,S,C),(D,B,C)]
        
        """
        
        fx = (self.bipolar_weight * x[:,None]).sum(dim=-1) + self.bipolar_bias
        fx = F.sigmoid(fx)[:,:,None] #(B,C,1)
        fx, h0 = self.kinetics(fx, hs[0]) 
        hs[1].append(fx)
        h1 = hs[1]
        fx = torch.stack(list(h1), dim=1) #(B,D,C,1)
        if self.scale_kinet:
            fx = self.kinet_scale(fx)
        fx = self.amacrine_filter(fx).squeeze(-1) #(B,C)
        fx = (self.amacrine_weight * fx[:,None]).sum(dim=-1) + self.amacrine_bias
        fx = F.relu(fx)
        fx = self.ganglion(fx)
        return fx, [h0, h1]