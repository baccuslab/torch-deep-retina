import torch
import torch.nn as nn
from torchdeepretina.custom_modules import *
import numpy as np
from scipy import signal

DEVICE = torch.device('cuda:0')

def try_kwarg(kwargs, key, default):
    try:
        return kwargs[key]
    except:
        return default

class TDRModel(nn.Module):
    """
    Base class for most models. Handles setting most of the member
    variables that are shared for most model definitions.
    """
    def __init__(self, n_units=5, noise=.05, bias=True, gc_bias=None,
                                                chans=[8,8],
                                                bn_moment=.01,
                                                softplus=True,
                                                inference_exp=False,
                                                img_shape=(40,50,50),
                                                ksizes=(15,11,11),
                                                recurrent=False,
                                                kinetic=False,
                                                convgc=False,
                                                centers=None,
                                                bnorm_d=1,
                                                activ_fxn='ReLU',
                                                **kwargs):
        """
        n_units: int
            number of different ganglion cells being fit
        noise: float
            the standard deviation for the gaussian noise layers
        bias: bool
            if true, convoluviontal layers will use a trainable bias
        gc_bias: bool
            if true, the final linear layer will include a bias
        chans: list of ints
            the channel depths for each layer. do not include the gc
            layer.
        bn_moment: float
            the batchnorm momentum
        softplus: bool
            if true, a softplus nonlinearity is included at the final
            layer for most models
        inference_exp: bool
            if true, an exponential is applied at the final layer
            during inference (when model is in eval mode)
        img_shape: tuple of ints (input_depth, height, width)
            the shape of the inputs, no including batch_size
        ksizes: tuple of ints
            the kernel sizes for each layer in the model. If using
            fully convolutional model, must include kernel size for
            final layer.
        recurrent: bool
            a switch to denote when a model is recurrent
        kinetic: bool
            a switch to determine when a model includes a kinetics
            layer
        convgc: bool
            if true, final layer is convolutional. If false, final
            layer is fully connected.
        centers: list of tuples of ints
            the list should have a row, col coordinate for the center
            of each ganglion cell receptive field.
        bnorm_d: int
            the dimension of the batchnorm layers. 1 indicates a
            BatchNorm1d layer type. This is spatially heterogeneous.
            A 2 indicates a BatchNorm2d layer type, which is spatially
            homogeneous.
        activ_fxn: string
            the name of the activation function to be used at the
            intermediate layers.
        """
        super().__init__()
        self.n_units = n_units
        self.chans = chans
        self.softplus = softplus
        self.infr_exp = inference_exp
        self.bias = bias
        self.img_shape = img_shape
        self.ksizes = ksizes
        self.gc_bias = gc_bias
        self.noise = noise
        self.bn_moment = bn_moment
        self.recurrent = recurrent
        self.kinetic = kinetic
        self.convgc = convgc
        self.centers = centers
        self.bnorm_d = bnorm_d
        assert bnorm_d == 1 or bnorm_d == 2, "Only 1 and 2 dim\
                             batchnorm are currently supported"
        self.activ_fxn = activ_fxn

    def forward(self, x):
        return x

    def extra_repr(self):
        """
        This function is used in the pytorch model printing. Gives
        details about the model's member variables.
        """
        s = ['n_units={}', 'noise={}', 'bias={}', 'gc_bias={}',
             'chans={}', 'bn_moment={}', 'softplus={}',
             'inference_exp={}', 'img_shape={}', 'ksizes={}']
        s = ", ".join(s)
        return s.format(self.n_units, self.noise, self.bias,
                                    self.gc_bias, self.chans,
                                    self.bn_moment, self.softplus,
                                    self.infr_exp,
                                    self.img_shape, self.ksizes)

    def requires_grad(self, state):
        """
        A function to turn on and off all gradient calculations. You
        will most likely want to use `with torch.no_grad():` instead.

        state: bool
            if true, then all parameters' `requires_grad` variable
            will be set to true. Visa-versa with false.
        """
        for p in self.parameters():
            try:
                p.requires_grad = state
            except:
                pass

class BNCNN(TDRModel):
    """
    The batchnorm model from Deep Learning Reveals ...
            (https://www.biorxiv.org/content/10.1101/340943v1)
    """
    def __init__(self, gauss_prior=0, **kwargs):
        """
        gauss_prior: float
            the standard deviation of a 2d gaussian shape applied to
            the initial values of the convolutional filters
        """
        super().__init__(**kwargs)
        self.name = 'McNiruNet'
        self.gauss_prior = gauss_prior
        modules = []
        self.shapes = []
        shape = self.img_shape[1:]
        modules.append(nn.Conv2d(self.img_shape[0],self.chans[0],
                                      kernel_size=self.ksizes[0],
                                      bias=self.bias))
        shape = update_shape(shape, self.ksizes[0])
        self.shapes.append(tuple(shape))
        if self.bnorm_d == 1:
            modules.append(Flatten())
            size = self.chans[0]*shape[0]*shape[1]
            modules.append(nn.BatchNorm1d(size, eps=1e-3,
                                momentum=self.bn_moment))
            modules.append(Reshape((-1,self.chans[0],*shape)))
        else:
            modules.append(nn.BatchNorm2d(self.chans[0], eps=1e-3,
                                         momentum=self.bn_moment))
        modules.append(GaussianNoise(std=self.noise))
        modules.append(getattr(nn,self.activ_fxn)())
        modules.append(nn.Conv2d(self.chans[0],self.chans[1],
                                  kernel_size=self.ksizes[1],
                                  bias=self.bias))
        shape = update_shape(shape, self.ksizes[1])
        self.shapes.append(tuple(shape))
        if self.bnorm_d == 1:
            modules.append(Flatten())
            size = self.chans[1]*shape[0]*shape[1]
            modules.append(nn.BatchNorm1d(size, eps=1e-3,
                                momentum=self.bn_moment))
            tup = (-1, self.chans[1], shape[0], shape[1])
            modules.append(Reshape(tup))
        else:
            modules.append(nn.BatchNorm2d(self.chans[1], eps=1e-3,
                                         momentum=self.bn_moment))
        modules.append(GaussianNoise(std=self.noise))
        modules.append(getattr(nn,self.activ_fxn)())
        if self.convgc:
            modules.append(nn.Conv2d(self.chans[1], self.n_units,
                                      kernel_size=self.ksizes[2],
                                      bias=self.gc_bias))
            shape = update_shape(shape, self.ksizes[2])
            self.shapes.append(tuple(shape))
            modules.append(GrabUnits(self.centers, self.ksizes,
                                               self.img_shape))
        else:
            modules.append(Flatten())
            size = self.chans[1]*shape[0]*shape[1]
            modules.append(nn.Linear(size, self.n_units,
                                     bias=self.gc_bias))
        modules.append(nn.BatchNorm1d(self.n_units, eps=1e-3,
                                    momentum=self.bn_moment))
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
                        prior_x = signal.gaussian(weight.shape[-1],
                                              std=self.gauss_prior)
                        prior_y = signal.gaussian(weight.shape[-2],
                                              std=self.gauss_prior)
                        prior = np.outer(prior_y, prior_x)
                        kernels.append(prior)
                    filters.append(np.asarray(kernels))
                prior = np.asarray(filters)
                denom = np.sqrt(weight.shape[0]+weight.shape[1])
                prior = prior/np.max(prior)/denom
                prior = torch.FloatTensor(prior)
                self.sequential[seq_idx].weight.data = prior

    def forward(self, x):
        if not self.training and self.infr_exp:
            return torch.exp(self.sequential(x))
        return self.sequential(x)

class LinearStackedBNCNN(TDRModel):
    """
    Similar to the BNCNN model, except that the convolutional filters
    are constructed out of s series of linear, smaller convolutions.
    Additionally the batchnorm parameters are forced to be positive.
    This prevents sign inversion.
    """
    def __init__(self, drop_p=0, one2one=False, stack_ksizes=3,
                                                 stack_chans=None,
                                                 paddings=None,
                                                 **kwargs):
        """
        drop_p: float between 0 and 1
            the dropout probability used between the sub-convolutions
            within the linear stacked layers.
        one2one: bool
            if true, prevents cross talk between channels in sub
            convolutions.
        stack_ksizes: list of ints or int
            the kernel sizes of the sub-convolutions. the zeroth index
            is the kernel size for the first LinearStackedConv2d,
            the first index is used for the second
            LinearStackedConv2d, and so on. Defaults to 3
        stack_chans: list of ints or None
            similar to stack_ksizes, but denotes the channel depths
            of each of the sub-convolutions. Defaults to the output
            channel depth is Nones are argued.
        paddings: list of ints
            the paddings for each layer. Defaults
        """
        super().__init__(**kwargs)
        self.name = 'StackedNet'
        self.drop_p = drop_p
        self.one2one = one2one
        if isinstance(stack_ksizes, int):
            stack_ksizes = [stack_ksizes for i in\
                                        range(len(self.ksizes))]
        self.stack_ksizes = stack_ksizes
        if stack_chans is None or isinstance(stack_chans, int):
            stack_chans = [stack_chans for i in\
                                        range(len(self.ksizes))]
        self.stack_chans = stack_chans
        self.paddings = [0 for x in stack_ksizes] if paddings is None\
                                                         else paddings
        shape = self.img_shape[1:] # (H, W)
        self.shapes = []
        modules = []

        ##### First Layer
        # Convolution
        if one2one:
            conv = OneToOneLinearStackedConv2d(self.img_shape[0],
                                        self.chans[0],
                                        kernel_size=self.ksizes[0],
                                        padding=self.paddings[0],
                                        bias=self.bias)

        else:
            conv = LinearStackedConv2d(self.img_shape[0],
                                    self.chans[0],
                                    kernel_size=self.ksizes[0],
                                    abs_bnorm=False,
                                    bias=self.bias,
                                    stack_chan=self.stack_chans[0],
                                    stack_ksize=self.stack_ksizes[0],
                                    drop_p=self.drop_p,
                                    padding=self.paddings[0])
        modules.append(conv)
        shape = update_shape(shape, self.ksizes[0],
                            padding=self.paddings[0])
        self.shapes.append(tuple(shape))
        # BatchNorm
        if self.bnorm_d == 1:
            modules.append(Flatten())
            size = self.chans[0]*shape[0]*shape[1]
            modules.append(AbsBatchNorm1d(size, eps=1e-3,
                                    momentum=self.bn_moment))
            modules.append(Reshape((-1,self.chans[0],shape[0],
                                                   shape[1])))
        else:
            modules.append(AbsBatchNorm2d(self.chans[0], eps=1e-3,
                                        momentum=self.bn_moment))
        # Noise and Activation
        modules.append(GaussianNoise(std=self.noise))
        modules.append(getattr(nn,self.activ_fxn)())

        ##### Second Layer
        # Convolution
        if one2one:
            conv = OneToOneLinearStackedConv2d(self.chans[0],
                                    self.chans[1],
                                    kernel_size=self.ksizes[1],
                                    padding=self.paddings[1],
                                    bias=self.bias)
        else:
            conv = LinearStackedConv2d(self.chans[0],self.chans[1],
                                    kernel_size=self.ksizes[1],
                                    abs_bnorm=False,
                                    bias=self.bias,
                                    stack_chan=self.stack_chans[1],
                                    stack_ksize=self.stack_ksizes[1],
                                    padding=self.paddings[1],
                                    drop_p=self.drop_p)
        modules.append(conv)
        shape = update_shape(shape, self.ksizes[1],
                            padding=self.paddings[1])
        self.shapes.append(tuple(shape))
        # BatchNorm
        if self.bnorm_d == 1:
            modules.append(Flatten())
            size = self.chans[1]*shape[0]*shape[1]
            modules.append(AbsBatchNorm1d(size, eps=1e-3,
                                momentum=self.bn_moment))
            modules.append(Reshape((-1,self.chans[1],shape[0],
                                                   shape[1])))
        else:
            modules.append(AbsBatchNorm2d(self.chans[1], eps=1e-3,
                                         momentum=self.bn_moment))
        # Noise and Activation
        modules.append(GaussianNoise(std=self.noise))
        modules.append(getattr(nn,self.activ_fxn)())

        ##### Final Layer
        if self.convgc:
            modules.append(nn.Conv2d(self.chans[1],self.n_units,
                                     kernel_size=self.ksizes[2],
                                     bias=self.gc_bias))
            shape = update_shape(shape, self.ksizes[2])
            self.shapes.append(tuple(shape))
            modules.append(GrabUnits(self.centers, self.ksizes,
                                               self.img_shape))
        else:
            modules.append(Flatten())
            modules.append(nn.Linear(self.chans[1]*shape[0]*shape[1],
                                    self.n_units, bias=self.gc_bias))
        modules.append(AbsBatchNorm1d(self.n_units, eps=1e-3,
                                        momentum=self.bn_moment))
        if self.softplus:
            modules.append(nn.Softplus())
        else:
            modules.append(Exponential(train_off=True))

        self.sequential = nn.Sequential(*modules)

    def forward(self, x):
        if not self.training and self.infr_exp:
            return torch.exp(self.sequential(x))
        return self.sequential(x)

    def deactivate_grads(self, deactiv=True):
        """
        Turns grad off for all trainable parameters in model

        deactiv: bool
            if true, deactivates gradient calculations for all
            parameters in the model
        """
        for p in self.parameters():
            p.requires_grad = not deactiv

    def tiled_forward(self,x):
        """
        Removes the grab-units layer, providing the full convolutional
        output from the model
        """
        if not self.convgc:
            return self.forward(x)
        fx = self.sequential[:-3](x) # Remove GrabUnits layer
        bnorm = self.sequential[-2]
        # Perform 2d batchnorm using 1d parameters from training
        fx =torch.nn.functional.batch_norm(fx,bnorm.running_mean.data,
                                            bnorm.running_var.data,
                                            weight=bnorm.scale.abs(),
                                            bias=bnorm.shift,
                                            eps=bnorm.eps,
                                            momentum=bnorm.momentum,
                                            training=self.training)
        fx = self.sequential[-1](fx)
        if not self.training and self.infr_exp:
            return torch.exp(fx)
        return fx

class LN(TDRModel):
    """
    Linear non-linear model
    """
    def __init__(self, drop_p=0, **kwargs):
        """
        drop_p: float between 0 and 1
            the dropout probability
        """
        super().__init__(**kwargs)
        modules = []
        self.shapes = []
        shape = self.img_shape
        self.drop_p = drop_p
        modules.append(Flatten())
        modules.append(nn.Dropout(self.drop_p))
        modules.append(nn.Linear(shape[2]*shape[0]*shape[1],
                                   self.n_units, bias=True))
        if self.softplus:
            modules.append(nn.Softplus())
        else:
            modules.append(nn.ELU())
        self.sequential = nn.Sequential(*modules)

    def forward(self, x):
        return self.sequential(x)

class RevCorLN:
    """
    LN model made by reverse correlation with fitted polynomial
    nonlinearity.
    """
    def __init__(self, filt, ln_cutout_size, center, norm_stats=[0,1],
                                            fit=[1,0], cell_file=None,
                                            img_shape=(40,50,50),
                                            cell_idx=None, **kwargs):
        """
        filt: ndarray or torch FloatTensor (C*H*W,) or (C,H,W)
            the filter
        ln_cutout_size: int
            the size of the cutout window centered on the gc receptive
            field.
        center: tuple of ints (row,col)
            the row,col coordinate of the ganglion cell receptive
            field.
        norm_stats: list of floats
            the normalization statistics of the data
        fit: list of floats
            the polynomial fit parameters.
        cell_file: str
            used as a tracking parameter to ensure models are not
            confused with one another.
        img_shape: tuple of ints
            the shape of the data
        cell_idx: int
            used as a tracking parameter to ensure models are not
            confused with one another.
        """
        if type(filt) == type(np.array([])):
            filt = torch.FloatTensor(filt)
        self.filt = filt.reshape(-1)
        self.span = ln_cutout_size
        self.center = center
        self.poly = Poly1d(fit)
        self.norm_stats = norm_stats
        self.cell_file = cell_file
        self.cell_idx = cell_idx
        self.img_shape = img_shape

    def normalize(self, x):
        """
        Normalizes x using the mean and std that were used during
        training of this model.
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
                normed[i:i+step_size] = normed_x.reshape(-1,
                                          *normed.shape[1:])
            return normed.reshape(shape)

    def convolve(self, x):
        if type(x) == type(np.array([])):
            x = torch.FloatTensor(x)
        batch_size = 500
        self.filt = self.filt.to(DEVICE)
        x = x.reshape(len(x), -1)
        outputs = torch.empty(len(x)).float()
        for i in range(0,len(x),batch_size):
            outs =torch.einsum("ij,j->i",x[i:i+batch_size].to(DEVICE),
                                                            self.filt)
            outputs[i:i+len(outs)] = outs.cpu()
        return outputs

    def __call__(self,x):
        fx = self.convolve(x)
        return self.poly(fx)

class VaryModel(TDRModel):
    """
    Built as a modular model that can assume the form of most other
    models in this package.
    """
    def __init__(self, n_layers=3, stackconvs=True, drop_p=0,
                                        one2one=False,
                                        stack_ksizes=[3,3],
                                        stack_chans=[None,None],
                                        paddings=None, **kwargs):
        super().__init__(**kwargs)
        """
        n_layers: int
            number of neural network layers. Includes ganglion cell
            layer
        stackconvs: bool
            if true, the convolutions are all linearstacked
            convolutions
        drop_p: float
            the dropout probability for the linearly stacked
            convolutions
        one2one: bool
            if true and stackconvs is true, then the stacked
            convolutions do not allow crosstalk between the inner
            channels
        stack_ksizes: list of ints
            the kernel size of the stacked convolutions
        stack_chans: list of ints
            the channel size of the stacked convolutions. If none,
            defaults to channel size of main convolution
        paddings: list of ints
            the padding for each conv layer. If none,
            defaults to 0.
        """
        self.name = 'VaryNet'
        self.n_layers = n_layers
        self.stackconvs = stackconvs
        self.drop_p = drop_p
        self.one2one = one2one
        if isinstance(stack_ksizes, int):
            stack_ksizes=[stack_ksizes for i in\
                                        range(len(self.ksizes))]
        self.stack_ksizes = stack_ksizes
        if stack_chans is None or isinstance(stack_chans, int):
            stack_chans = [stack_chans for i in\
                                        range(len(self.ksizes))]
        self.stack_chans = stack_chans
        self.paddings = [0 for x in stack_ksizes] if paddings is None\
                                                         else paddings
        shape = self.img_shape[1:] # (H, W)
        self.shapes = []
        modules = []

        #### Layer Loop
        temp_chans = [self.img_shape[0]]
        if self.n_layers > 1:
            temp_chans = [self.img_shape[0]]+self.chans
        for i in range(self.n_layers-1):
            ## Convolution
            if not self.stackconvs:
                conv = nn.Conv2d(temp_chans[i],temp_chans[i+1],
                                    kernel_size=self.ksizes[i],
                                    padding=self.paddings[i],
                                    bias=self.bias)
            else:
                if self.one2one:
                    conv = OneToOneLinearStackedConv2d(temp_chans[i],
                                          temp_chans[i+1],
                                          kernel_size=self.ksizes[i],
                                          padding=self.paddings[i],
                                          bias=self.bias)
                else:
                    conv = LinearStackedConv2d(temp_chans[i],
                                         temp_chans[i+1],
                                         kernel_size=self.ksizes[i],
                                         abs_bnorm=False,
                                         bias=self.bias,
                                         stack_chan=stack_chans[i],
                                         stack_ksize=stack_ksizes[i],
                                         drop_p=self.drop_p,
                                         padding=self.paddings[i])
            modules.append(conv)
            shape = update_shape(shape, self.ksizes[i],
                                        padding=self.paddings[i])
            self.shapes.append(tuple(shape))

            ## BatchNorm
            if self.bnorm_d == 1:
                modules.append(Flatten())
                size = temp_chans[i+1]*shape[0]*shape[1]
                modules.append(AbsBatchNorm1d(size,eps=1e-3,
                             momentum=self.bn_moment))
                modules.append(Reshape((-1,temp_chans[i+1],*shape)))
            else:
                bnorm = AbsBatchNorm2d(temp_chans[i+1],eps=1e-3,
                                        momentum=self.bn_moment)
                modules.append(bnorm)
            # Noise and Activation
            modules.append(GaussianNoise(std=self.noise))
            modules.append(getattr(nn,self.activ_fxn)())

        ##### Final Layer
        if self.convgc:
            conv = nn.Conv2d(temp_chans[-1],self.n_units,
                             kernel_size=self.ksizes[self.n_layers-1],
                             bias=self.gc_bias)
            modules.append(conv)
            shape = update_shape(shape, self.ksizes[self.n_layers-1])
            self.shapes.append(tuple(shape))
            modules.append(GrabUnits(self.centers, self.ksizes,
                                               self.img_shape))
        else:
            modules.append(Flatten())
            modules.append(nn.Linear(temp_chans[-1]*shape[0]*shape[1],
                                                 self.n_units,
                                                 bias=self.gc_bias))
        modules.append(AbsBatchNorm1d(self.n_units, eps=1e-3,
                                    momentum=self.bn_moment))
        if self.softplus:
            modules.append(nn.Softplus())
        else:
            modules.append(Exponential(train_off=True))

        self.sequential = nn.Sequential(*modules)

    def forward(self, x):
        if not self.training and self.infr_exp:
            return torch.exp(self.sequential(x))
        return self.sequential(x)

    def deactivate_grads(self, deactiv=True):
        """
        Turns grad off for all trainable parameters in model
        """
        for p in self.parameters():
            p.requires_grad = deactiv

    def tiled_forward(self,x):
        """
        Allows for the fully convolutional functionality
        """
        if not self.convgc:
            return self.forward(x)
        # Remove GrabUnits layer
        fx = self.sequential[:-3](x)
        bnorm = self.sequential[-2]
        # Perform 2dbatchnorm using 1d parameters
        fx =torch.nn.functional.batch_norm(fx,bnorm.running_mean.data,
                                            bnorm.running_var.data,
                                            weight=bnorm.scale.abs(),
                                            bias=bnorm.shift,
                                            eps=bnorm.eps,
                                            momentum=bnorm.momentum,
                                            training=self.training)
        fx = self.sequential[-1:](fx)
        if not self.training and self.infr_exp:
            return torch.exp(fx)
        return fx

class RetinotopicModel(TDRModel):
    """
    Built as a modular model that can assume the form of most other
    models in this package.
    """
    def __init__(self, n_layers=3, stackconvs=True, drop_p=0,
                                        one2one=False,
                                        stack_ksizes=[3,3],
                                        stack_chans=[None,None],
                                        paddings=None,
                                        collapsed=False,
                                        collapse_labels=None,
                                        **kwargs):
        super().__init__(**kwargs)
        """
        n_layers: int
            number of neural network layers. Includes ganglion cell
            layer
        stackconvs: bool
            if true, the convolutions are all linearstacked
            convolutions
        drop_p: float
            the dropout probability for the linearly stacked
            convolutions
        one2one: bool
            if true and stackconvs is true, then the stacked
            convolutions do not allow crosstalk between the inner
            channels
        stack_ksizes: list of ints
            the kernel size of the stacked convolutions
        stack_chans: list of ints
            the channel size of the stacked convolutions. If none,
            defaults to channel size of main convolution
        paddings: list of ints
            the padding for each conv layer. If none,
            defaults to 0.
        """
        self.name = 'VaryNet'
        self.n_layers = n_layers
        self.stackconvs = stackconvs
        self.drop_p = drop_p
        self.one2one = one2one
        self.collapsed = collapsed
        self.collapse_labels=collapse_labels



        if isinstance(stack_ksizes, int):
            stack_ksizes=[stack_ksizes for i in\
                                        range(len(self.ksizes))]
        self.stack_ksizes = stack_ksizes
        if stack_chans is None or isinstance(stack_chans, int):
            stack_chans = [stack_chans for i in\
                                        range(len(self.ksizes))]
        self.stack_chans = stack_chans
        self.paddings = [0 for x in stack_ksizes] if paddings is None\
                                                         else paddings
        shape = self.img_shape[1:] # (H, W)
        self.shapes = []
        modules = []

        #### Layer Loop
        temp_chans = [self.img_shape[0]]
        if self.n_layers > 1:
            temp_chans = [self.img_shape[0]]+self.chans
        for i in range(self.n_layers-1):
            ## Convolution
            if not self.stackconvs:
                conv = nn.Conv2d(temp_chans[i],temp_chans[i+1],
                                    kernel_size=self.ksizes[i],
                                    padding=self.paddings[i],
                                    bias=self.bias)
            else:
                conv = LinearStackedConv2d(temp_chans[i],
                                     temp_chans[i+1],
                                     kernel_size=self.ksizes[i],
                                     abs_bnorm=False,
                                     bias=self.bias,
                                     stack_chan=stack_chans[i],
                                     stack_ksize=stack_ksizes[i],
                                     drop_p=self.drop_p,
                                     padding=self.paddings[i])

            modules.append(conv)
            shape = update_shape(shape, self.ksizes[i],
                                        padding=self.paddings[i])
            self.shapes.append(tuple(shape))

            ## BatchNorm
            if self.bnorm_d == 1:
                modules.append(Flatten())
                size = temp_chans[i+1]*shape[0]*shape[1]
                modules.append(AbsBatchNorm1d(size,eps=1e-3,
                             momentum=self.bn_moment))
                modules.append(Reshape((-1,temp_chans[i+1],*shape)))
            else:
                bnorm = AbsBatchNorm2d(temp_chans[i+1],eps=1e-3,
                                        momentum=self.bn_moment)
                modules.append(bnorm)
            # Noise and Activation
            modules.append(GaussianNoise(std=self.noise))
            modules.append(getattr(nn,self.activ_fxn)())

        ##### Final Layer
        if not self.collapsed:
            modules.append(nn.Conv2d(self.chans[-1],
            self.n_units,
            kernel_size=self.ksizes[-1],
            bias=self.bias))

            modules.append(AbsBatchNorm2d(self.n_units, eps=1e-3,
                                        momentum=self.bn_moment))

            shape = update_shape(shape, self.ksizes[-1])
            self.shapes.append(tuple(shape))

            modules.append(nn.Softplus())

            modules.append(OneHot((self.n_units,*shape)))


        elif self.collapsed:
            print('This was collapsed!')
            modules.append(nn.Conv2d(self.chans[-1],
            len(np.unique(self.collapse_labels)),
            self.ksizes[-1],
            self.bias))

            modules.append(AbsBatchNorm2d(len(np.unique(self.collapse_labels)),
                                        eps=1e-3,
                                        momentum=self.bn_moment))

            shape = update_shape(shape, self.ksizes[-1])
            self.shapes.append(tuple(shape))
            modules.append(nn.Softplus())
            print('Collapse_Shape ???{}'.format(shape))
            modules.append(ConsolidatedOneHot([self.n_units,shape[-1],shape[-1]],self.collapse_labels))


        self.sequential = nn.Sequential(*modules)
    def forward(self, x):
        if not self.training and self.infr_exp:
            return torch.exp(self.sequential(x))
        return self.sequential(x)

    def deactivate_grads(self, deactiv=True):
        """
        Turns grad off for all trainable parameters in model
        """
        for p in self.parameters():
            p.requires_grad = deactiv

    def tiled_forward(self,x):
        """
        Allows for the fully convolutional functionality
        """
        if not self.convgc:
            return self.forward(x)
        # Remove One-Hot layer
        fx = self.sequential[:-1](x)
        bnorm = self.sequential[-2]
        # Perform 2dbatchnorm using 1d parameters
        fx =torch.nn.functional.batch_norm(fx,bnorm.running_mean.data,
                                            bnorm.running_var.data,
                                            weight=bnorm.scale.abs(),
                                            bias=bnorm.shift,
                                            eps=bnorm.eps,
                                            momentum=bnorm.momentum,
                                            training=self.training)
        fx = self.sequential[-1:](fx)
        if not self.training and self.infr_exp:
            return torch.exp(fx)
        return fx
