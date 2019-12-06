import torch
import torch.nn as nn
from torchdeepretina.torch_utils import *
import torchdeepretina.utils as tdrutils
import numpy as np
from scipy import signal

DEVICE = torch.device('cuda:0')

def try_kwarg(kwargs, key, default):
    try:
        return kwargs[key]
    except:
        return default

class TDRModel(nn.Module):
    def __init__(self, n_units=5, noise=.05, bias=True, linear_bias=None, chans=[8,8],
                                                bn_moment=.01, softplus=True, 
                                                inference_exp=False, img_shape=(40,50,50), 
                                                ksizes=(15,11), recurrent=False, 
                                                kinetic=False, convgc=False, 
                                                centers=None, bnorm_d=1, **kwargs):
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
        self.recurrent = recurrent
        self.kinetic = kinetic
        self.convgc = convgc
        self.centers = centers
        assert bnorm_d == 1 or bnorm_d == 2,\
                                "Only 1 and 2 dimensional batchnorm are currently supported"
        self.bnorm_d = bnorm_d
    
    def forward(self, x):
        return x

    def extra_repr(self):
        try:
            s = 'n_units={}, noise={}, bias={}, linear_bias={}, chans={}, bn_moment={}, '+\
                                    'softplus={}, inference_exp={}, img_shape={}, ksizes={}'
            return s.format(self.n_units, self.noise, self.bias, self.linear_bias,
                                        self.chans, self.bn_moment, self.softplus,
                                        self.inference_exp, self.img_shape, self.ksizes)
        except:
            pass
    
    def requires_grad(self, state):
        for p in self.parameters():
            try:
                p.requires_grad = state
            except:
                pass

class BNCNN(TDRModel):
    def __init__(self, gauss_prior=0, **kwargs):
        super().__init__(**kwargs)
        self.name = 'McNiruNet'
        self.gauss_prior = gauss_prior
        modules = []
        self.shapes = []
        shape = self.img_shape[1:]
        modules.append(nn.Conv2d(self.img_shape[0],self.chans[0],kernel_size=self.ksizes[0], 
                                                                            bias=self.bias))
        shape = update_shape(shape, self.ksizes[0])
        self.shapes.append(tuple(shape))
        if self.bnorm_d == 1:
            modules.append(Flatten())
            modules.append(nn.BatchNorm1d(self.chans[0]*shape[0]*shape[1], eps=1e-3,
                                                           momentum=self.bn_moment))
            modules.append(Reshape((-1,self.chans[0],*shape)))
        else:
            modules.append(nn.BatchNorm2d(self.chans[0], eps=1e-3, momentum=self.bn_moment))
        modules.append(GaussianNoise(std=self.noise))
        modules.append(nn.ReLU())
        modules.append(nn.Conv2d(self.chans[0],self.chans[1],kernel_size=self.ksizes[1], 
                                                                        bias=self.bias))
        shape = update_shape(shape, self.ksizes[1])
        self.shapes.append(tuple(shape))
        if self.bnorm_d == 1:
            modules.append(Flatten())
            modules.append(nn.BatchNorm1d(self.chans[1]*shape[0]*shape[1], eps=1e-3, 
                                                            momentum=self.bn_moment))
        else:
            modules.append(nn.BatchNorm2d(self.chans[1], eps=1e-3, momentum=self.bn_moment))
            modules.append(Flatten())
        modules.append(GaussianNoise(std=self.noise))
        modules.append(nn.ReLU())
        modules.append(nn.Linear(self.chans[1]*shape[0]*shape[1],self.n_units,
                                                       bias=self.linear_bias))
        modules.append(nn.BatchNorm1d(self.n_units, eps=1e-3, momentum=self.bn_moment))
        if self.softplus:
            modules.append(nn.Softplus())
        else:
            modules.append(Exponential(train_off=True))
        self.sequential = nn.Sequential(*modules)

        if self.gauss_prior > 0:
            for i,seq_idx in enumerate([0,6]):
                weight = self.sequential[seq_idx].weight
                filters = []
                for out_i in range(weight.shape[0]):
                    kernels = []
                    for in_i in range(weight.shape[1]):
                        prior_x = signal.gaussian(weight.shape[-1],std=self.gauss_prior)
                        prior_y = signal.gaussian(weight.shape[-2],std=self.gauss_prior)
                        prior = np.outer(prior_y, prior_x)
                        kernels.append(prior)
                    filters.append(np.asarray(kernels))
                prior = np.asarray(filters)
                prior = prior/np.max(prior)/np.sqrt(weight.shape[0]+weight.shape[1])
                self.sequential[seq_idx].weight.data = torch.FloatTensor(prior)
        
    def forward(self, x):
        if not self.training and self.infr_exp:
            return torch.exp(self.sequential(x))
        return self.sequential(x)

class LinearStackedBNCNN(TDRModel):
    def __init__(self, drop_p=0, one2one=False, stack_ksizes=[3,3], stack_chans=[None,None],
                                                 final_bias=False, paddings=None, **kwargs):
        super().__init__(**kwargs)
        self.name = 'StackedNet'
        self.drop_p = drop_p
        self.one2one = one2one
        self.stack_ksizes = stack_ksizes
        self.stack_chans = stack_chans
        self.paddings = [0 for x in stack_ksizes] if paddings is None else paddings
        self.final_bias = final_bias
        shape = self.img_shape[1:] # (H, W)
        self.shapes = []
        modules = []

        ##### First Layer
        # Convolution
        if one2one:
            modules.append(OneToOneLinearStackedConv2d(self.img_shape[0],self.chans[0],
                                                            kernel_size=self.ksizes[0], 
                                                            padding=self.paddings[0],
                                                            bias=self.bias))
        else:
            modules.append(LinearStackedConv2d(self.img_shape[0],self.chans[0],
                                                    kernel_size=self.ksizes[0], 
                                                    abs_bnorm=False, bias=self.bias, 
                                                    stack_chan=self.stack_chans[0], 
                                                    stack_ksize=self.stack_ksizes[0],
                                                    drop_p=self.drop_p, 
                                                    padding=self.paddings[0]))
        shape = update_shape(shape, self.ksizes[0], padding=self.paddings[0])
        self.shapes.append(tuple(shape))
        # BatchNorm
        if self.bnorm_d == 1:
            modules.append(Flatten())
            modules.append(AbsBatchNorm1d(self.chans[0]*shape[0]*shape[1], eps=1e-3, 
                                                            momentum=self.bn_moment))
            modules.append(Reshape((-1,self.chans[0],shape[0], shape[1])))
        else:
            modules.append(AbsBatchNorm2d(self.chans[0], eps=1e-3, momentum=self.bn_moment))
        # Noise and ReLU
        modules.append(GaussianNoise(std=self.noise))
        modules.append(nn.ReLU())

        ##### Second Layer
        # Convolution
        if one2one:
            modules.append(OneToOneLinearStackedConv2d(self.chans[0],self.chans[1],
                                                        kernel_size=self.ksizes[1], 
                                                        padding=self.paddings[1],
                                                        bias=self.bias))
        else:
            modules.append(LinearStackedConv2d(self.chans[0],self.chans[1],
                                                    kernel_size=self.ksizes[1], 
                                                    abs_bnorm=False, bias=self.bias, 
                                                    stack_chan=self.stack_chans[1], 
                                                    stack_ksize=self.stack_ksizes[1],
                                                    padding=self.paddings[1],
                                                    drop_p=self.drop_p))
        shape = update_shape(shape, self.ksizes[1], padding=self.paddings[1])
        self.shapes.append(tuple(shape))
        # BatchNorm
        if self.bnorm_d == 1:
            modules.append(Flatten())
            modules.append(AbsBatchNorm1d(self.chans[1]*shape[0]*shape[1], eps=1e-3, 
                                                        momentum=self.bn_moment))
        else:
            modules.append(AbsBatchNorm2d(self.chans[1], eps=1e-3, momentum=self.bn_moment))
            modules.append(Flatten())
        # Noise and ReLU
        modules.append(GaussianNoise(std=self.noise))
        modules.append(nn.ReLU())

        ##### Final Layer
        if self.convgc:
            modules.append(Reshape((-1, self.chans[1], shape[0], shape[1])))
            modules.append(nn.Conv2d(self.chans[1],self.n_units,kernel_size=self.ksizes[2], 
                                                                    bias=self.linear_bias))
            shape = update_shape(shape, self.ksizes[2])
            self.shapes.append(tuple(shape))
            modules.append(GrabUnits(self.centers, self.ksizes, self.img_shape, self.n_units))
            modules.append(AbsBatchNorm1d(self.n_units, momentum=self.bn_moment))
        else:
            modules.append(nn.Linear(self.chans[1]*shape[0]*shape[1], self.n_units, 
                                                                bias=self.linear_bias))
            modules.append(AbsBatchNorm1d(self.n_units, eps=1e-3, momentum=self.bn_moment))
        if self.softplus:
            modules.append(nn.Softplus())
        else:
            modules.append(Exponential(train_off=True))
        if self.final_bias:
            modules.append(Add(0,trainable=True))

        self.sequential = nn.Sequential(*modules)

    def forward(self, x):
        if not self.training and self.infr_exp:
            return torch.exp(self.sequential(x))
        return self.sequential(x)
    
    def tiled_forward(self,x):
        """
        Allows for the fully convolutional functionality
        """
        if not self.convgc:
            return self.forward(x)
        fx = self.sequential[:-3](x) # Remove GrabUnits layer
        bnorm = self.sequential[-2]
        # Perform 2d batchnorm using 1d parameters collected from training
        fx = torch.nn.functional.batch_norm(fx, bnorm.running_mean.data, bnorm.running_var.data,
                                                    weight=bnorm.scale.abs(), bias=bnorm.shift, 
                                                    eps=bnorm.eps, momentum=bnorm.momentum, 
                                                    training=self.training)
        fx = self.sequential[-1](fx)
        if not self.training and self.infr_exp:
            return torch.exp(fx)
        return fx

class LN(TDRModel):
    def __init__(self, drop_p=0, **kwargs):
        super().__init__(**kwargs)
        modules = []
        self.shapes = []
        shape = self.img_shape
        self.drop_p = drop_p
        modules.append(Flatten())
        #modules.append(GaussianNoise(std=self.noise))
        modules.append(nn.Dropout(self.drop_p))
        modules.append(nn.Linear(shape[2]*shape[0]*shape[1], self.n_units, bias=True))
        #modules.append(nn.BatchNorm1d(self.n_units, eps=1e-3, momentum=self.bn_moment))
        if self.softplus:
            modules.append(nn.Softplus())
        else:
            modules.append(nn.ELU())
        self.sequential = nn.Sequential(*modules)
 
    def forward(self, x):
        return self.sequential(x)

class ProtoAmacRNN(TDRModel):
    def __init__(self, rnn_chans=[2], bnorm=False, drop_p=0, stackconvs=False, **kwargs):
        super().__init__(**kwargs)
        self.shapes = []
        self.h_shapes = []
        shape = self.img_shape[1:] # (H, W)
        self.bnorm = bnorm
        self.recurrent = True
        self.rnn_chans = rnn_chans
        self.drop_p = drop_p
        self.stackconvs = stackconvs
        
        # Bipolar Block
        if stackconvs:
            self.bipolar1 = LinearStackedConv2d(self.img_shape[0],self.chans[0],kernel_size=self.ksizes[0], 
                                                                                        abs_bnorm=False, 
                                                                                        bias=self.bias, 
                                                                                        drop_p=self.drop_p)
        else:
            self.bipolar1 = nn.Conv2d(self.img_shape[0], self.chans[0], self.ksizes[0], bias=self.bias)
        shape = update_shape(shape, self.ksizes[0])
        self.shapes.append(tuple(shape))
        self.h_shapes.append((self.chans[0], *shape))
        self.h_shapes.append((self.rnn_chans[0],*shape))

        modules = []
        if bnorm:
            modules.append(Flatten())
            modules.append(nn.BatchNorm1d(self.chans[0]*shape[0]*shape[1], momentum=self.bn_moment))
            modules.append(Reshape((-1, self.chans[0], shape[0], shape[1])))
        modules.append(GaussianNoise(std=self.noise))
        modules.append(nn.ReLU())
        self.bipolar2 = nn.Sequential(*modules)

        # Amacrine Block
        self.amacrine1 = DalesAmacRNN(self.chans[0], self.chans[1], rnn_chans[0], self.ksizes[1], bias=self.bias, stackconvs=stackconvs)
        shape = update_shape(shape, self.ksizes[1])
        self.shapes.append(tuple(shape))

        modules = []
        modules.append(Flatten())
        if bnorm:
            modules.append(nn.BatchNorm1d(self.chans[1]*shape[0]*shape[1], momentum=self.bn_moment))
        modules.append(GaussianNoise(std=self.noise))
        modules.append(nn.ReLU())
        modules.append(InvertSign())
        self.amacrine2 = nn.Sequential(*modules)

        # Ganglion Block
        modules = []
        length = self.chans[0]*self.shapes[0][0]*self.shapes[0][1] + self.chans[1]*self.shapes[1][0]*self.shapes[1][1]
        modules.append(AbsLinear(length, self.n_units, bias=self.linear_bias))
        if self.bnorm:
            modules.append(nn.BatchNorm1d(self.n_units, eps=1e-3, momentum=self.bn_moment))
        else:
            modules.append(ScaleShift(self.n_units))
        if self.softplus:
            modules.append(nn.Softplus())
        else:
            modules.append(Exponential(train_off=True))
        self.ganglion = nn.Sequential(*modules)

    def forward(self, x, h):
        """
        x: torch FloatTensor (B, C, H, W)
            the inputs
        h: tuple or list containing h_bi, h_am
            h_bi: torch FloatTensor (B, C2, H2, W2)
                the bipolar cell states
            h_am: torch FloatTensor (B, RNN_CHAN, H, W)
                the amacrine cell states
        """
        h_bi, h_am = h

        bipolar = self.bipolar1(x) + h_bi # Conv
        bipolar = self.bipolar2(bipolar)

        amacrine, h_bi_new, h_am_new = self.amacrine1(bipolar, h_am)
        amacrine = self.amacrine2(amacrine)

        flat_bi = bipolar.view(x.shape[0], -1)
        flat_am = amacrine.view(x.shape[0], -1)
        cat = torch.cat([flat_bi, flat_am], dim=-1)
        ganglion = self.ganglion(cat)
        if not self.training and self.infr_exp:
            ganglion = torch.exp(ganglion)
        return ganglion, [h_bi_new, h_am_new]

class SkipAmacRNN(TDRModel):
    def __init__(self, rnn_chans=[2], bnorm=False, drop_p=0, stackconvs=False, **kwargs):
        super().__init__(**kwargs)
        self.shapes = []
        self.h_shapes = []
        shape = self.img_shape[1:] # (H, W)
        self.bnorm = bnorm
        self.recurrent = True
        self.rnn_chans = rnn_chans
        self.drop_p = drop_p
        self.stackconvs = stackconvs
        
        # Bipolar Block
        if stackconvs:
            self.bipolar1 = LinearStackedConv2d(self.img_shape[0],self.chans[0],
                                                kernel_size=self.ksizes[0], 
                                                abs_bnorm=False, bias=self.bias, 
                                                drop_p=self.drop_p)
        else:
            self.bipolar1 = nn.Conv2d(self.img_shape[0], self.chans[0], self.ksizes[0], 
                                                                        bias=self.bias)
        shape = update_shape(shape, self.ksizes[0])
        self.shapes.append(tuple(shape))
        self.h_shapes.append((self.chans[0], *shape))
        self.h_shapes.append((self.rnn_chans[0],*shape))

        modules = []
        if bnorm:
            modules.append(Flatten())
            modules.append(nn.BatchNorm1d(self.chans[0]*shape[0]*shape[1], 
                                                    momentum=self.bn_moment))
            modules.append(Reshape((-1, self.chans[0], shape[0], shape[1])))
        modules.append(GaussianNoise(std=self.noise))
        modules.append(nn.ReLU())
        self.bipolar2 = nn.Sequential(*modules)

        # Amacrine Block
        self.amacrine1 = AmacRNNFull(self.chans[0], self.chans[1], rnn_chans[0], 
                                                    self.ksizes[1], bias=self.bias, 
                                                    stackconvs=stackconvs)
        shape = update_shape(shape, self.ksizes[1])
        self.shapes.append(tuple(shape))

        modules = []
        modules.append(Flatten())
        if bnorm:
            modules.append(nn.BatchNorm1d(self.chans[0]*shape[0]*shape[1], 
                                                    momentum=self.bn_moment))
        modules.append(GaussianNoise(std=self.noise))
        modules.append(nn.ReLU())
        modules.append(InvertSign())
        self.amacrine2 = nn.Sequential(*modules)

        # Ganglion Block
        modules = []
        length = self.chans[0]*self.shapes[0][0]*self.shapes[0][1] + \
                                    self.chans[1]*self.shapes[1][0]*self.shapes[1][1]
        modules.append(AbsLinear(length, self.n_units, bias=self.linear_bias))
        if self.bnorm:
            modules.append(nn.BatchNorm1d(self.n_units, eps=1e-3, momentum=self.bn_moment))
        else:
            modules.append(ScaleShift(self.n_units))
        if self.softplus:
            modules.append(nn.Softplus())
        else:
            modules.append(Exponential(train_off=True))
        self.ganglion = nn.Sequential(*modules)

    def forward(self, x, h):
        """
        x: torch FloatTensor (B, C, H, W)
            the inputs
        h: tuple or list containing h_bi, h_am
            h_bi: torch FloatTensor (B, C2, H2, W2)
                the bipolar cell states
            h_am: torch FloatTensor (B, RNN_CHAN, H, W)
                the amacrine cell states
        """
        h_bi, h_am = h

        bipolar = self.bipolar1(x) + h_bi # Conv
        bipolar = self.bipolar2(bipolar)

        amacrine, h_bi_new, h_am_new = self.amacrine1(bipolar, h_am)
        amacrine = self.amacrine2(amacrine) # Inhibitory from InvertSign layer

        flat_bi = bipolar.view(x.shape[0], -1)
        flat_am = amacrine.view(x.shape[0], -1)
        cat = torch.cat([flat_bi, flat_am], dim=-1)
        ganglion = self.ganglion(cat)
        if not self.training and self.infr_exp:
            ganglion = torch.exp(ganglion)
        return ganglion, [h_bi_new, h_am_new]

class RNNCNN(TDRModel):
    def __init__(self, rnn_chans=[2,2], bnorm=False, rnn_type="ConvGRUCell", **kwargs):
        super().__init__(**kwargs)
        self.rnns = nn.ModuleList([])
        self.shapes = []
        self.h_shapes = []
        shape = self.img_shape[1:] # (H, W)
        self.h_shapes.append((rnn_chans[0], *shape))
        self.bnorm = bnorm
        self.recurrent = True
        self.rnn_type = rnn_type
        rnn_class = globals()[rnn_type]

        # Block 1
        modules = []
        self.rnns.append(rnn_class(self.img_shape[0], self.chans[0], rnn_chans[0], 
                                                        kernel_size=self.ksizes[0], 
                                                        bias=self.bias))
        shape = update_shape(shape, self.ksizes[0])
        self.shapes.append(tuple(shape))
        self.h_shapes.append((rnn_chans[1], *shape))
        modules.append(Flatten())
        if self.bnorm:
            modules.append(nn.BatchNorm1d(self.chans[0]*shape[0]*shape[1], eps=1e-3, 
                                                            momentum=self.bn_moment))
        modules.append(GaussianNoise(std=self.noise))
        modules.append(nn.ReLU())
        modules.append(Reshape((-1,self.chans[0],*shape)))
        self.sequential1 = nn.Sequential(*modules)

        # Block 2
        modules = []
        self.rnns.append(rnn_class(self.chans[0], self.chans[1], rnn_chans[1], 
                                                    kernel_size=self.ksizes[1],
                                                    bias=self.bias))
        shape = update_shape(shape, self.ksizes[1])
        self.shapes.append(tuple(shape))
        modules.append(Flatten())
        if self.bnorm:
            modules.append(nn.BatchNorm1d(self.chans[1]*shape[0]*shape[1], eps=1e-3,
                                                           momentum=self.bn_moment))
        modules.append(GaussianNoise(std=self.noise))
        modules.append(nn.ReLU())
        modules.append(nn.Linear(self.chans[1]*shape[0]*shape[1],self.n_units,
                                                        bias=self.linear_bias))
        if self.bnorm:
            modules.append(nn.BatchNorm1d(self.n_units, eps=1e-3, momentum=self.bn_moment))
        else:
            modules.append(ScaleShift(self.n_units))
        if self.softplus:
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
        fx, h2 = self.rnns[1](fx, hs[1])
        fx = self.sequential2(fx)
        if not self.training and self.infr_exp:
            fx = torch.exp(fx)
        return fx, [h1, h2]

class KineticsModel(TDRModel):
    def __init__(self, bnorm=True, drop_p=0, scale_kinet=False, recur_seq_len=5, **kwargs):
        super().__init__(**kwargs)
        self.bnorm = bnorm
        self.drop_p = drop_p
        self.recurrent = True
        self.kinetic = True
        self.scale_kinet = scale_kinet
        self.seq_len = recur_seq_len
        shape = self.img_shape[1:] # (H, W)
        self.shapes = []
        self.h_shapes = []

        modules = []
        modules.append(LinearStackedConv2d(self.img_shape[0],self.chans[0],kernel_size=self.ksizes[0], abs_bnorm=False, 
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
        modules.append(Reshape((-1,self.seq_len*self.chans[0],shape[0], shape[1])))
        modules.append(LinearStackedConv2d(self.seq_len*self.chans[0],self.chans[1],kernel_size=self.ksizes[1], abs_bnorm=False, 
                                                                                bias=self.bias, drop_p=self.drop_p))
        shape = update_shape(shape, self.ksizes[1])
        self.shapes.append(tuple(shape))
        modules.append(Flatten())
        modules.append(AbsBatchNorm1d(self.chans[1]*shape[0]*shape[1], eps=1e-3, momentum=self.bn_moment))
        modules.append(GaussianNoise(std=self.noise))
        modules.append(nn.ReLU())
        self.amacrine = nn.Sequential(*modules)

        modules = []
        modules.append(nn.Linear(self.chans[1]*shape[0]*shape[1], self.n_units, bias=self.linear_bias))
        modules.append(AbsBatchNorm1d(self.n_units, eps=1e-3, momentum=self.bn_moment))
        if self.softplus:
            modules.append(nn.Softplus())
        else:
            modules.append(Exponential(train_off=True))
        self.ganglion = nn.Sequential(*modules)

    def forward(self, x, hs):
        """
        x - FloatTensor (B, C, H, W)
        hs - list [(B,S,N),(B,D,H1,W1)]
            First list element should be a torch FloatTensor of state population values.
            Second element should be deque of activated population values over past D time steps
        """
        fx = self.bipolar(x)
        fx, h0 = self.kinetics(fx, hs[0])
        hs[1].append(fx)
        h1 = hs[1]
        fx = torch.cat(list(h1), dim=1)
        if self.scale_kinet:
            fx = self.kinet_scale(fx)
        fx = self.amacrine(fx)
        fx = self.ganglion(fx)
        if not self.training and self.infr_exp:
            fx = torch.exp(fx)
        return fx, [h0, h1]

class RevCorLN:
    """
    LN model made by reverse correlation with fitted polynomial nonlinearity.
    """
    def __init__(self, filt, ln_cutout_size, center, norm_stats=[0,1], fit=[1,0], cell_file=None,
                                                             img_shape=(40,50,50),cell_idx=None,
                                                             **kwargs):
        if type(filt) == type(np.array([])):
            filt = torch.FloatTensor(filt)
        self.filt = filt.reshape(-1)
        self.span = ln_cutout_size
        self.center = center
        self.poly = tdrutils.poly1d(fit)
        self.norm_stats = norm_stats
        self.cell_file = cell_file
        self.cell_idx = cell_idx
        self.img_shape = img_shape

    def normalize(self, x):
        """
        Normalizes x using the mean and std that were used during training of this model.
        """
        mu,sigma = self.norm_stats
        shape = x.shape
        x = x.reshape(len(x),-1)
        try:
            normed_x = (x-mu)/(sigma+1e-7)
            return normed_x.reshape(shape)
        except MemoryError as e:
            step_size = 1000
            normed = torch.empty_like(x)
            for i in range(0,len(x),step_size):
                temp = x[i:i+step_size]
                normed_x = (temp-mu)/(sigma+1e-7)
                normed[i:i+step_size] = normed_x.reshape(-1,*normed.shape[1:])
            return normed.reshape(shape)

    def convolve(self, x):
        if type(x) == type(np.array([])):
            x = torch.FloatTensor(x)
        batch_size = 500
        self.filt = self.filt.to(DEVICE)
        x = x.reshape(len(x), -1)
        outputs = torch.empty(len(x)).float()
        for i in range(0,len(x),batch_size):
            outs = torch.einsum("ij,j->i", x[i:i+batch_size].to(DEVICE), self.filt)
            outputs[i:i+len(outs)] = outs.cpu()
        return outputs

    def __call__(self,x):
        fx = self.convolve(x)
        return self.poly(fx)

