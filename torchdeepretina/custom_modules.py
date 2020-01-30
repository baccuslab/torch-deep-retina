import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

if torch.cuda.is_available():
    DEVICE = torch.device("cuda:0")
else:
    DEVICE = torch.device("cpu")

def update_shape(shape, kernel=3, padding=0, stride=1, op="conv"):
    """
    Calculates the new shape of the tensor following a convolution or
    deconvolution

    shape: list-like or int
        the height/width of the activations
    kernel: int or list-like
        size of the kernel
    padding: list-like or int
    stride: list-like or int
    op: str
        'conv' or 'deconv'
    """
    if type(shape) == type(int()):
        shape = np.asarray([shape])
    if type(kernel) == type(int()):
        kernel = np.asarray([kernel for i in range(len(shape))])
    if type(padding) == type(int()):
        padding = np.asarray([padding for i in range(len(shape))])
    if type(stride) == type(int()):
        stride = np.asarray([stride for i in range(len(shape))])

    if op == "conv":
        shape = (shape - kernel + 2*padding)/stride + 1
    elif op == "deconv" or op == "conv_transpose":
        shape = (shape - 1)*stride + kernel - 2*padding
    if len(shape) == 1:
        return int(shape[0])
    return [int(s) for s in shape]

def diminish_weight_magnitude(params):
    for param in params:
        divisor = float(np.prod(param.data.shape))/2
        if param.data is not None and divisor >= 0:
            param.data = param.data/divisor

class Poly1d:
    """
    Creates a polynomial with the argued fit
    """
    def __init__(self, fit):
        """
        fit: list of float
            the polynomial coefs. the zeroth index of the fit
            corresponds to the largest power of x.
            y = fit[-1] + fit[-2]*x + fig[-3]*x**2 + ...
        """
        self.coefficients = fit
        self.poly = self.get_poly(fit)

    def get_poly(self, fit):
        def poly(x):
            cumu_sum = 0
            for i in range(len(fit)):
                cumu_sum = cumu_sum + fit[i]*(x**(len(fit)-i-1))
            return cumu_sum
        return poly

    def __call__(self, x):
        return self.poly(x)

class GrabUnits(nn.Module):
    """
    A module that returns the model units centered at the argued
    centers (row col coordinates).
    """
    def __init__(self, centers, ksizes, img_shape):
        """
        centers: list of tuples of ints
            this should be a list of (row,col) coordinates of the
            receptive field centers for each of the ganglion cells.
        ksizes: list of ints
            the kernel sizes of each of the layers. len(ksizes)==3
        img_shape: tuple (chan, height, width)
            the shape of the original image
        """
        super().__init__()
        assert len(ksizes) > 2 and centers is not None
        self.ksizes = ksizes
        self.img_shape = img_shape
        self.coords = self.centers2coords(centers,ksizes,img_shape)
        self.chans = torch.arange(len(centers)).long()

    def centers2coords(self, centers, ksizes, img_shape):
        """
        Converts the argued center coordinates into coordinates
        of the output layer. Assumes a stride of 1 with 0 padding
        in each layer.

        centers: list of tuples of ints
            this should be a list of (row,col) coordinates of the
            receptive field centers for each of the ganglion cells.
        ksizes: list of ints
            the kernel sizes of each of the layers. len(ksizes)==3
        img_shape: tuple (chan, height, width)
            the shape of the original image
        """
        # Each quantity is even, thus the final half_effective_ksize is odd
        half_effective_ksize = (ksizes[0]-1) + (ksizes[1]-1) +\
                                            (ksizes[2]//2-1) + 1
        coords = []
        for center in centers:
            row = min(max(0,center[0]-half_effective_ksize),
                        img_shape[1]-2*(half_effective_ksize-1))
            col = min(max(0,center[1]-half_effective_ksize),
                        img_shape[2]-2*(half_effective_ksize-1))
            coords.append([row,col])
        return torch.LongTensor(coords)

    def forward(self, x):
        units = x[...,:,self.chans,self.coords[:,0],self.coords[:,1]]
        return units

class ScaledSoftplus(nn.Module):
    def __init__(self):
        super(ScaledSoftplus, self).__init__()
        self.scale = nn.Parameter(torch.ones(1))
        self.Softplus = nn.Softplus()

    def forward(self, x):
        return self.scale*self.Softplus(x)

class Exponential(nn.Module):
    """
    Simply the exponential function. Option exists
    to only use during inference.
    """
    def __init__(self, train_off=True):
        """
        train_off: bool
            if true, then the module performs the identity
            operation when in training mode.
        """
        super(Exponential, self).__init__()
        self.train_off = train_off

    def forward(self, x):
        if self.training and self.train_off:
            return x
        return torch.exp(x)

    def extra_repr(self):
        return 'train_off={}'.format(self.train_off)

class GaussRegularizer:
    def __init__(self, model, conv_idxs, std=1):
        """
        Regularizes a model such that weights further from the spatial
        center of the model have a greater regularizing effect,
        encouraging a large center with weaker edges.

        model - torch nn Module
        conv_idxs - list of indices of convolutional layers
        std - int
            standard deviation of gaussian in terms of pixels
        """
        modu = model.sequential[conv_idxs[0]]
        # Needs to be Conv2d module
        assert "sequential" in dir(model) and type(modu) == nn.Conv2d
        self.weights = [model.sequential[i].weight for i in conv_idxs]
        self.std = std
        self.gaussians = []
        for i,weight in enumerate(self.weights):
            shape = weight.shape[1:]
            half_width = shape[1]//2
            sq = 1/np.sqrt(2*np.pi*self.std**2)
            ar = np.arange(-half_width,half_width+1)**2
            ex = np.exp(-(ar/(2*self.std**2)))
            pdf = sq * ex
            gauss = np.outer(pdf, pdf)
            inverted = 1/(gauss+1e-5)
            inverted = (inverted-np.min(inverted))/np.max(inverted)
            full_gauss = np.asarray([gauss for i in range(shape[0])])
            self.gaussians.append(torch.FloatTensor(full_gauss))

    def get_loss(self):
        """
        Calculates the loss associated with the regularizer.
        """
        if self.weights[0].data.is_cuda and not\
                                        self.gaussians[0].is_cuda:
            self.gaussians = [g.to(DEVICE) for g in self.gaussians]
        loss = 0
        for weight,gauss in zip(self.weights,self.gaussians):
            loss += (weight*gauss).mean()
        return loss

class GaussianNoise(nn.Module):
    def __init__(self, std=0.1, trainable=False, adapt=False,
                                               momentum=.95):
        """
        std - float
            the standard deviation of the noise to add to the layer.
            if adapt is true, this is used as the proportional value to
            set the std to based of the std of the activations.
            gauss_std = activ_std*std
        trainable - bool
            If trainable is set to True, then the std is turned into
            a learned parameter. Cannot be set to true if adapt is True
        adapt - bool
            adapts the gaussian std to a proportion of the
            std of the received activations. Cannot be set to True if
            trainable is True
        momentum - float (0 <= momentum < 1)
            this is the exponentially moving average factor for
            updating the activ_std. 0 uses the std of the current
            activations.
        """
        super(GaussianNoise, self).__init__()
        self.trainable = trainable
        self.adapt = adapt
        assert not (self.trainable and self.adapt)
        self.std = std
        self.sigma = nn.Parameter(torch.ones(1)*std,
                            requires_grad=trainable)
        self.running_std = 1
        self.momentum = momentum if adapt else None

    def forward(self, x):
        if not self.training or self.std == 0:
            return x
        if self.adapt:
            xstd = x.std().item()
            self.running_std = self.momentum*self.running_std +\
                                          (1-self.momentum)*xstd
            self.sigma.data[0] = self.std*self.running_std
        noise = self.sigma*torch.randn_like(x)
        return x + noise

    def extra_repr(self):
        s = 'std={}, trainable={}, adapt={}, momentum={}'
        return s.format(self.std, self.trainable,
                        self.adapt, self.momentum)

class ScaleShift(nn.Module):
    """
    Scales and shifts the activations by a learnable amount.
    """
    def __init__(self, shape, scale=True, shift=True):
        """
        shape: tuple (depth, height, width) or (length,)
            shape of the incoming activations discluding the
            batch dimension
        scale: bool
            include multiplicative parameter
        shift: bool
            include summing parameter
        """
        super(ScaleShift, self).__init__()
        self.shape = shape
        self.scale = scale
        self.shift = shift
        self.scale_param = nn.Parameter(torch.ones(shape).float(),
                                              requires_grad=scale)
        self.shift_param= nn.Parameter(torch.zeros(shape).float(),
                                              requires_grad=shift)

    def forward(self, x):
        return x*self.scale_param + self.shift_param

    def extra_repr(self):
        s = 'shape={}, scale={}, shift={}'
        return s.format(self.shape, self.scale, self.shift)

class AbsScaleShift(nn.Module):
    """
    Performs a learned scaling and shifting, but the parameters
    are constrained to be positive.
    """
    def __init__(self, shape, scale=True, shift=True,
                                    abs_shift=False):
        """
        shape: tuple (depth, height, width) or (length,)
            shape of the incoming activations discluding the
            batch dimension
        scale: bool
            include multiplicative parameter
        shift: bool
            include summing parameter
        abs_shift: bool
            if true, the shifting parameters are contrained
            to be positive. if false, no constraints are set.
        """
        super(AbsScaleShift, self).__init__()
        self.shape = shape
        self.scale = scale
        self.shift = shift
        self.abs_shift = abs_shift
        self.scale_param = nn.Parameter(torch.ones(shape).float(),
                                              requires_grad=scale)
        self.shift_param= nn.Parameter(torch.zeros(shape).float(),
                                              requires_grad=shift)

    def forward(self, x):
        if self.abs_shift:
            shift = self.shift_param.abs()
        return x*self.scale_param.abs() + self.shift_param

    def extra_repr(self):
        s = 'shape={}, scale={}, shift={}, abs_shift={}'
        return s.format(self.shape, self.scale, self.shift,
                                            self.abs_shift)

class DaleActivations(nn.Module):
    """
    Constrains the activations so that a portion of the
    channels are positive and a portion of the channels are
    negative.

    For the full Dale effect, will also need to use AbsConv2d
    and AbsLinear layers.
    """
    def __init__(self, n_chan, neg_p=.33):
        """
        n_chan: int
            number of channels in the layer
        neg_p: float
            portion of channels to be made negative
        """
        super(DaleActivations, self).__init__()
        self.n_chan = n_chan
        self.neg_p = neg_p
        self.n_neg_chan = int(neg_p * n_chan)
        mask = torch.ones(n_chan).float()
        mask[:self.n_neg_chan] = mask[:self.n_neg_chan]*-1
        self.mask = nn.Parameter(mask, requires_grad=False)

    def forward(self, x):
        x = x.permute(0,2,3,1).abs()
        x = x*self.mask
        return x.permute(0,3,1,2).contiguous()

    def extra_repr(self):
        s = 'n_chan={}, neg_p={}, n_neg_chan={}'
        return s.format(self.n_chan, self.neg_p, self.n_neg_chan)

class BatchNorm1d(nn.Module):
    def __init__(self, n_units, bias=True, momentum=.1, eps=1e-5):
        super(BatchNorm1d, self).__init__()
        self.n_units = n_units
        self.momentum = momentum
        self.eps = eps
        self.running_mean = nn.Parameter(torch.zeros(n_units))
        self.running_var = nn.Parameter(torch.ones(n_units))
        self.scale = nn.Parameter(torch.ones(n_units).float())
        self.bias = bias
        self.shift = nn.Parameter(torch.zeros(n_units).float())

    def forward(self, x):
        self.shift.requires_grad = self.bias
        return torch.nn.functional.batch_norm(x,
                                self.running_mean.data,
                                self.running_var.data,
                                weight=self.scale,
                                bias=self.shift,
                                eps=self.eps,
                                momentum=self.momentum,
                                training=self.training)

    def extra_repr(self):
        return 'bias={}, momentum={}, eps={}'.format(self.bias,
                                       self.momentum, self.eps)

class AbsBatchNorm1d(nn.Module):
    """
    BatchNorm1d module in which the scaling parameters are
    constrained to be positive
    """
    def __init__(self, n_units, bias=True, abs_bias=False,
                                   momentum=.1, eps=1e-5):
        """
        n_units: int
            the number of activations that will be operated on
        bias: bool
            include learnable shifting parameter
        abs_bias: bool
            restrict bias to only be positive
        momentum: float
            the amount to update the running mean and running
            variance by after each batch
        eps: float
            a value to prevent division by zero
        """
        super(AbsBatchNorm1d, self).__init__()
        self.n_units = n_units
        self.momentum = momentum
        self.eps = eps
        # Running mean and var are parameters so that they get
        # saved in the state dict
        self.running_mean = nn.Parameter(torch.zeros(n_units))
        self.running_var = nn.Parameter(torch.ones(n_units))
        self.scale = nn.Parameter(torch.ones(n_units).float())
        self.bias = bias
        self.abs_bias = abs_bias
        self.shift = nn.Parameter(torch.zeros(n_units).float())

    def forward(self, x):
        assert len(x.shape) == 2
        self.shift.requires_grad = self.bias
        shift = self.shift
        if self.abs_bias:
            shift = shift.abs()
        return torch.nn.functional.batch_norm(x,
                                    self.running_mean.data,
                                    self.running_var.data,
                                    weight=self.scale.abs(),
                                    bias=shift, eps=self.eps,
                                    momentum=self.momentum,
                                    training=self.training)

    def extra_repr(self):
        s = 'bias={}, abs_bias={}, momentum={}, eps={}'
        return s.format(self.bias, self.abs_bias, self.momentum,
                                                       self.eps)

class AbsBatchNorm2d(nn.Module):
    """
    BatchNorm2d module in which the scaling parameters are
    constrained to be positive
    """
    def __init__(self, n_units, bias=True, abs_bias=False,
                                    momentum=.1, eps=1e-5):
        """
        n_units: int
            the number of activations that will be operated on
        bias: bool
            include learnable shifting parameter
        abs_bias: bool
            restrict bias to only be positive
        momentum: float
            the amount to update the running mean and running
            variance by after each batch
        eps: float
            a value to prevent division by zero
        """
        super(AbsBatchNorm2d, self).__init__()
        self.n_units = n_units
        self.momentum = momentum
        self.eps = eps
        self.running_mean = nn.Parameter(torch.zeros(n_units))
        self.running_var = nn.Parameter(torch.ones(n_units))
        self.scale = nn.Parameter(torch.ones(n_units).float())
        self.bias = bias
        self.abs_bias = abs_bias
        self.shift = nn.Parameter(torch.zeros(n_units).float())

    def forward(self, x):
        assert len(x.shape) == 4
        self.shift.requires_grad = self.bias
        shift = self.shift
        if self.abs_bias:
            shift = shift.abs()
        return torch.nn.functional.batch_norm(x,
                                    self.running_mean.data,
                                    self.running_var.data,
                                    weight=self.scale.abs(),
                                    bias=shift, eps=self.eps,
                                    momentum=self.momentum,
                                    training=self.training)

    def extra_repr(self):
        s = 'bias={}, abs_bias={}, momentum={}, eps={}'
        return s.format(self.bias, self.abs_bias, self.momentum,
                                                       self.eps)

class AbsConvTranspose2d(nn.Module):
    """
    A convolution transpose that restricts it's parameters to
    be positive.
    """
    def __init__(self, in_channels, out_channels, kernel_size,
                stride=1, padding=0, bias=True, abs_bias=False):
        """
        same as nn.Conv2d
        """
        super().__init__()
        self.abs_bias = abs_bias
        self.bias = bias
        self.conv = nn.ConvTranspose2d(in_channels, out_channels,
                                                kernel_size, stride,
                                                padding, bias=bias)

    def forward(self, x):
        weight = self.conv.weight.abs()
        bias = None
        if self.bias and self.abs_bias:
            bias = self.conv.bias.abs()
        elif self.bias:
            bias = self.conv.bias
        return nn.functional.conv_transpose2d(x, weight, bias,
                                              self.conv.stride,
                                              self.conv.padding)

    def extra_repr(self):
        return 'bias={}, abs_bias={}'.format(self.bias, self.abs_bias)

class AbsConv2d(nn.Module):
    """
    A convolution module that contrains it's weights to be positive.
    The bias can be negative if abs_bias is set to False.
    """
    def __init__(self, in_channels, out_channels, kernel_size,
                              stride=1, padding=0, dilation=1,
                              groups=1, bias=True, abs_bias=False):
        super(AbsConv2d, self).__init__()
        self.abs_bias = abs_bias
        self.bias = bias
        self.conv = nn.Conv2d(in_channels, out_channels,
                           kernel_size, stride, padding,
                           dilation, groups, bias)

    def forward(self, x):
        bias = None
        if self.bias:
            bias = self.conv.bias
            if self.abs_bias:
                bias = bias.abs()
        weight = self.conv.weight.abs()
        return nn.functional.conv2d(x, weight, bias,
                                    self.conv.stride,
                                    self.conv.padding,
                                    self.conv.dilation,
                                    self.conv.groups)

    def extra_repr(self):
        return 'bias={}, abs_bias={}'.format(self.bias, self.abs_bias)

class AbsLinear(nn.Module):
    """
    Performs a fully connected operation in which the weights
    are all positive.
    """
    def __init__(self, in_features, out_features, bias=True,
                                            abs_bias=False):
        super(AbsLinear, self).__init__()
        self.bias = bias
        self.abs_bias = abs_bias
        self.linear = nn.Linear(in_features, out_features, bias=bias)

    def forward(self, x):
        bias = None
        weight = self.linear.weight.abs()
        if self.bias:
            bias = self.linear.bias
            if self.abs_bias:
                bias = bias.abs()
        return nn.functional.linear(x, weight, bias)

    def extra_repr(self):
        return 'bias={}, abs_bias={}'.format(self.bias, self.abs_bias)

class AbsLinearStackedConv2d(nn.Module):
    '''
    Builds argued kernel out of multiple 3x3 kernels without
    added nonlinearities. All weights are forced to be positive.
    '''
    def __init__(self, in_channels, out_channels, kernel_size,
                                   bias=True, abs_bnorm=False,
                                   conv_bias=False, drop_p=0,
                                   padding=0):
        """
        in_channels: int
        out_channels: int
        kernel_size: int
        bias: bool
        abs_bnorm: bool
            if true, AbsBatchnorm2d modules are placed between
            each of the convolutions.
        conv_bias: bool
            if true, each of the stacked convolutions have a bias
            term
        drop_p: float between 0 and 1
            adds dropout between each of the stacked convolutions
            with a drop probability equal to drop_p
        padding: int
        """
        super().__init__()
        assert kernel_size % 2 == 1 # kernel must be odd
        self.ksize = kernel_size
        self.bias = bias
        self.conv_bias = conv_bias
        self.abs_bnorm = abs_bnorm
        self.padding = padding
        self.drop_p = drop_p
        n_filters = int((kernel_size-1)/2)
        if n_filters > 1:
            convs = [AbsConv2d(in_channels, out_channels, 3,
                                            bias=conv_bias)]
            if abs_bnorm:
                convs.append(AbsBatchNorm2d(out_channels))
            if drop_p > 0:
                convs.append(nn.Dropout(drop_p))
            for i in range(n_filters-1):
                if i == n_filters-2:
                    convs.append(AbsConv2d(out_channels, out_channels,
                                                        3, bias=bias))
                else:
                    convs.append(AbsConv2d(out_channels, out_channels,
                                                    3, bias=conv_bias))
                    if abs_bnorm:
                        convs.append(AbsBatchNorm2d(out_channels))
                    if drop_p > 0:
                        convs.append(nn.Dropout(drop_p))
        else:
            convs = [AbsConv2d(in_channels, out_channels, 3, bias=bias)]
        self.convs = nn.Sequential(*convs)

    def forward(self, x):
        x = F.pad(x, (self.padding, self.padding, self.padding,
                                                 self.padding))
        return self.convs(x)

    def extra_repr(self):
        return 'bias={}, abs_bnorm={}'.format(self.bias, self.abs_bnorm)

class LinearStackedConv2d(nn.Module):
    '''
    Builds argued kernel out of multiple KxK kernels with no
    nonlinearities between the kernels. There is cross talk
    between each of the intermediate channels.
    '''
    def __init__(self, in_channels, out_channels, kernel_size,
                                   bias=True, conv_bias=False,
                                   stack_ksize=3, stack_chan=None,
                                   abs_bnorm=False, bnorm=False,
                                   drop_p=0, padding=0, stride=(1,1)):
        """
        in_channels: int
        out_channels: int
        kernel_size: int
            the size of the effective convolution kernel
        bias: bool
            bias refers to the final layer only
        conv_bias: bool
            conv_bias refers to the inner stacked convolution biases
        stack_ksize: int
            the size of the stacked filters (must be odd integers)
        stack_chan: int
            the number of channels in the inner stacked convolutions
        bnorm: bool
            if true, inserts batchnorm layers between each stacked
            convolution
        abs_bnorm: bool
            if true, inserts absolute value batchnorm layers between
            each stacked convolution. takes precedence over bnorm
        drop_p: float
            the amount of dropout used between stacked convolutions
        padding: int
        stride: int or tuple
        """
        super(LinearStackedConv2d, self).__init__()
        assert kernel_size % 2 == 1 # kernel must be odd
        assert kernel_size > 1 # kernel must be greater than 1
        self.ksize = kernel_size
        self.stack_ksize = stack_ksize
        assert self.stack_ksize <= self.ksize
        self.bias = bias
        self.conv_bias = conv_bias
        self.abs_bnorm = abs_bnorm
        self.padding = 0 if padding is None else padding
        self.drop_p = drop_p
        self.stack_chan = out_channels if stack_chan is None else stack_chan

        n_filters = (kernel_size-self.stack_ksize)/(self.stack_ksize-1)+1
        if n_filters - int(n_filters) > 0:
            effective=self.stack_ksize+int(n_filters-1)*(self.stack_ksize-1)
            remaining = (kernel_size-effective)
            self.last_ksize = remaining+1
            n_filters += 1
        else:
            self.last_ksize = self.stack_ksize
        n_filters = int(n_filters)

        if n_filters > 1:
            pad_inc = int(self.stack_ksize//2)
            pad = min(pad_inc,padding) if padding > 0 else 0
            padding -= pad

            convs = [nn.Conv2d(in_channels, self.stack_chan,
                            self.stack_ksize, bias=conv_bias,
                            padding=pad)]
            if abs_bnorm:
                convs.append(AbsBatchNorm2d(self.stack_chan))
            if drop_p > 0:
                convs.append(nn.Dropout(drop_p))
            for i in range(n_filters-1):
                if i < n_filters-2:
                    pad = min(pad_inc,padding) if padding > 0 else 0
                    padding -= pad
                    convs.append(nn.Conv2d(self.stack_chan,
                                    self.stack_chan, self.stack_ksize,
                                    bias=conv_bias, padding=pad))
                    if abs_bnorm:
                        convs.append(AbsBatchNorm2d(self.stack_ksize))
                    elif bnorm:
                        convs.append(nn.BatchNorm2d(self.stack_ksize))
                    if drop_p > 0:
                        convs.append(nn.Dropout(drop_p))
                # Last Convolution
                else:
                    pad = padding if padding > 0 else 0
                    convs.append(nn.Conv2d(self.stack_chan,
                                        out_channels, self.last_ksize,
                                        bias=bias, padding=pad,
                                        stride=stride))
        else:
            convs = [nn.Conv2d(in_channels, out_channels,
                                self.stack_ksize, bias=bias,
                                padding=padding)]
        self.convs = nn.Sequential(*convs)

    def forward(self, x):
        return self.convs(x)

    def extra_repr(self):
        s = 'bias={}, abs_bnorm={}, padding={}'
        return s.format(self.bias, self.abs_bnorm, self.padding)

class OneToOneLinearStackedConv2d(nn.Module):
    '''
    Builds argued kernel out of multiple 3x3 kernels without added
    nonlinearities. No crosstalk between intermediate channels.
    '''
    def __init__(self, in_channels, out_channels, kernel_size,
                        bias=True, conv_bias=False, padding=0):
        super().__init__()
        assert kernel_size % 2 == 1 # kernel must be odd
        self.ksize = kernel_size
        self.bias = bias
        self.conv_bias = conv_bias
        self.padding = padding
        n_filters = int((kernel_size-1)/2)
        assert n_filters > 1
        self.first_conv = nn.Conv2d(in_channels, out_channels, 3,
                                                  bias=conv_bias)
        convs = []
        self.seqs = nn.ModuleList([])
        for c in range(out_channels):
            convs.append([])
            for i in range(n_filters-1):
                if i == n_filters-2:
                    convs[c].append(nn.Conv2d(1, 1, 3, bias=bias))
                else:
                    convs[c].append(nn.Conv2d(1, 1, 3, bias=conv_bias))
            self.seqs.append(nn.Sequential(*convs[c]))

    def forward(self, x):
        x = F.pad(x, (self.padding, self.padding, self.padding,
                                                 self.padding))
        fx = self.first_conv(x)
        outs = []
        for chan,seq in enumerate(self.seqs):
            outs.append(seq(fx[:,chan:chan+1]))
        fx = torch.cat(outs,dim=1)
        return fx

    def extra_repr(self):
        s = 'bias={}, conv_bias={}, padding={}'
        return s.format(self.bias, self.conv_bias, self.padding)

class ShakeShakeModule(nn.Module):
    """
    Employs the shake shake regularization described in
    Shake-Shake regularization (https://arxiv.org/abs/1705.07485)
    """
    def __init__(self, module, n_shakes=2, batch_size=1000):
        super().__init__()
        """
        module: torch.nn.Module
            The module should contain at least one Conv2d module
        n_shakes: int
            number of parallel shakes
        """
        assert n_shakes > 1, 'Number of shakes must be greater than 1'
        self.n_shakes = n_shakes
        self.modu_list = nn.ModuleList([module])
        for i in range(n_shakes-1):
            new_module = copy.deepcopy(self.modu_list[-1])
            for name,modu in new_module.named_modules():
                if isinstance(modu, nn.Conv2d) or\
                                isinstance(modu, nn.Linear):
                    nn.init.xavier_uniform(modu.weight)
            self.modu_list.append(new_module)
        self.alphas = nn.Parameter(torch.zeros(batch_size, n_shakes),
                                                 requires_grad=False)

    def update_alphas(self, is_training, batch_size):
        """
        Updates the alphas to each be in range [0-1) and sum to about 1

        is_training: bool
            if true, alphas will be random values. If false, alphas
            will default to equal values (all summing to 1)
        batch_size: int
            number of samples in the batch
        """
        if is_training:
            remainder = torch.ones(batch_size)
            rands = torch.rand(batch_size, self.n_shakes)
            if self.alphas.is_cuda:
                self.alphas.data = torch.zeros(batch_size,
                                        self.n_shakes).to(DEVICE)
                rands = rands.to(DEVICE)
                remainder = remainder.to(DEVICE)
            else:
                self.alphas.data = torch.zeros(batch_size, self.n_shakes)

            for i in range(self.n_shakes-1):
                self.alphas.data[:,i] = rands[:,i]*remainder
                remainder -= self.alphas.data[:,i]
            self.alphas.data[:,-1] = remainder
        else:
            ones = torch.ones(batch_size, self.n_shakes)
            uniform = ones/float(self.n_shakes)
            if self.alphas.is_cuda:
                uniform = uniform.to(DEVICE)
            self.alphas.data = uniform
        ones = torch.ones(batch_size)
        if self.alphas.is_cuda:
            ones = ones.to(DEVICE)
        assert ((self.alphas.data.sum(-1)-ones)**2).mean() < 0.01

    def forward(self, x):
        self.update_alphas(is_training=self.train, batch_size=len(x))
        fx = 0
        einstring = "ij,i->ij" if len(x.shape) == 2 else "ijkl,i->ijkl"
        for i in range(self.n_shakes):
            output = self.modu_list[i](x)
            output = torch.einsum(einstring, output, self.alphas[:,i])
            fx += output
        # Post update is used to randomize gradients
        self.update_alphas(is_training=self.train, batch_size=len(x))
        return fx

class ConvGRUCell(nn.Module):
    def __init__(self, in_channels, out_channels, rnn_channels,
                                        kernel_size, stride=1,
                                        padding=0, dilation=1,
                                        groups=1, bias=True):
        super().__init__()
        self.in_chans = in_channels
        self.out_chans = out_channels
        self.rnn_chans = rnn_channels
        self.ksize = kernel_size
        self.stride = stride
        self.padding = padding
        self.bias = bias
        self.conv = nn.Conv2d(in_channels+rnn_channels, out_channels,
                                        kernel_size, stride, padding,
                                        dilation, groups, bias)
        assert kernel_size % 2 == 1 # Must have odd kernel size
        self.rnn_padding = (kernel_size-1)//2
        temp_conv = nn.Conv2d(in_channels+rnn_channels,
                                                rnn_channels*2,
                                                kernel_size, stride,
                                                self.rnn_padding,
                                                bias=True)
        self.rnn_conv = nn.Sequential(temp_conv, nn.Sigmoid())
        temp_conv = nn.Conv2d(in_channels+rnn_channels,
                              rnn_channels, kernel_size,
                              stride, self.rnn_padding,
                              bias=True)
        self.tan_conv = nn.Sequential(temp_conv, nn.Tanh())

    def forward(self, x, h):
        """
        x: torch tensor (B, IN_CHAN, H, W)
            the new input
        h: torch tensor (B, RNN_CHAN, H, W)
            the recurrent input
        """
        ins = torch.cat([x,h], dim=1)
        outs = self.conv(ins)
        temp = self.rnn_conv(ins)
        z,r = (temp[:,:self.rnn_chans], temp[:,self.rnn_chans:])
        tanin = torch.cat(x,h*r, dim=1)
        tanout = self.tan_conv(tanin)
        h_new = z*h + (1-z)*tanout
        return outs, h_new

class Abs(nn.Module):
    """
    Performs an absolute value operation on incoming activations
    """
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.abs(x)

class InvertSign(nn.Module):
    """
    Inverts the sign of incoming activations
    """
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return -x

class Add(nn.Module):
    """
    Adds a single value to the activations
    """
    def __init__(self, additive, trainable=False):
        """
        additive: float
            the amount to be added to the activations
        trainable: bool
            if true, the additive can be trained
        """
        super().__init__()
        self.trainable = trainable
        self.additive = nn.Parameter(torch.ones(1)*additive,
                                    requires_grad=trainable)

    def forward(self, x):
        if not self.trainable:
            self.additive.requires_grad = False
        return x+self.additive

class Clamp(nn.Module):
    """
    Clips the activations between low and high
    """
    def __init__(self, low, high):
        super().__init__()
        self.low = low
        self.high = high

    def forward(self, x):
        return torch.clamp(x, self.low, self.high)

class Multiply(nn.Module):
    """
    Multiplies a single value on the activations
    """
    def __init__(self, multiplier, trainable=False):
        """
        multiplier: float
            the amount to be multiplied on the activations
        trainable: bool
            if true, the multiplier can be trained
        """
        super().__init__()
        self.trainable = trainable
        self.multiplier = nn.Parameter(torch.ones(1)*multiplier,
                                        requires_grad=trainable)

    def forward(self, x):
        if not self.trainable:
            self.multiplier.requires_grad = False
        return x+self.multiplier

class Flatten(nn.Module):
    """
    Reshapes the activations to be of shape (B,-1) where B
    is the batch size
    """
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        return x.view(x.shape[0], -1)

class Reshape(nn.Module):
    """
    Reshapes the activations to be of shape (B, *shape) where B
    is the batch size.
    """
    def __init__(self, shape):
        super(Reshape, self).__init__()
        self.shape = shape

    def forward(self, x):
        return x.view(self.shape)

    def extra_repr(self):
        return "shape={}".format(self.shape)

class SplitConv2d(nn.Module):
    """
    Performs parallel convolutional operations on the input.
    Each convolution layer must return the same shaped
    output.
    """
    def __init__(self, conv_param_tuples, ret_stacked=True):
        super(SplitConv2d,self).__init__()
        self.convs = nn.ModuleList([])
        self.ret_stacked = ret_stacked
        for tup in conv_param_tuples:
            convs.append(nn.Conv2d(*tup))

    def forward(self, x):
        fxs = []
        for conv in self.convs:
            fxs.append(conv(x))
        if self.ret_stacked:
            return torch.cat(fxs, dim=1) # Concat on channel axis
        else:
            cumu_sum = fxs[0]
            for fx in fxs[1:]:
                cumu_sum = cumu_sum + fx
            return cumu_sum

    def extra_repr(self):
        return 'ret_stacked={}'.format(self.ret_stacked)

class SkipConnection(nn.Module):
    """
    Performs a conv2d and returns the output stacked with
    the original input.
    """
    def __init__(self, in_channels, out_channels, kernel_size,
                                                   bias=True):
        super(SkipConnection,self).__init__()
        padding = kernel_size//2
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                                stride=1, padding=padding, bias=bias)
        self.relu = nn.ReLU()

    def forward(self, x):
        fx = self.conv(x)
        fx = self.relu(fx)
        return torch.cat([x,fx], dim=1)

class SkipConnectionBN(nn.Module):
    """
    Performs a conv2d and returns the output stacked with
    the original input.
    """
    def __init__(self, in_channels, out_channels, kernel_size,
                                x_shape=(40,50,50), bias=True,
                                noise=.05):
        super(SkipConnectionBN,self).__init__()
        padding = kernel_size//2
        modules = []
        modules.append(nn.Conv2d(in_channels, out_channels,
                                     kernel_size, stride=1,
                                     padding=padding, bias=bias))
        modules.append(Flatten())
        modules.append(nn.BatchNorm1d(out_channels*x_shape[-2]*x_shape[-1]))
        modules.append(GaussianNoise(noise))
        modules.append(nn.ReLU())
        modules.append(Reshape((out_channels,x_shape[-2],x_shape[-1])))
        self.sequential = nn.Sequential(*modules)

    def forward(self, x):
        fx = self.sequential(x)
        return torch.cat([x,fx], dim=1)

class Kinetics(nn.Module):
    def __init__(self, dt=0.001):
        super().__init__()
        self.ka = nn.Parameter(torch.rand(1).abs()/10)
        self.kfi = nn.Parameter(torch.rand(1).abs()/10)
        self.kfr = nn.Parameter(torch.rand(1).abs()/10)
        self.ksi = nn.Parameter(torch.rand(1).abs()/10)
        self.ksr = nn.Parameter(torch.rand(1).abs()/10)
        self.dt = 0.001

    def extra_repr(self):
        return "dt={}".format(self.dt)

    def clamp_params(self, low, high):
        self.ka.data  = torch.clamp(self.ka.data,  low, high)
        self.kfi.data = torch.clamp(self.kfi.data, low, high)
        self.kfr.data = torch.clamp(self.kfr.data, low, high)
        self.ksi.data = torch.clamp(self.ksi.data, low, high)
        self.ksr.data = torch.clamp(self.ksr.data, low, high)

    def forward(self, rate, pop):
        """
        rate - FloatTensor (B, N)
            firing rates
        pop - FloatTensor (B, S, N)
            populations should have 4 states for each neuron.
            States should be:
                0: R
                1: A
                2: I1
                3: I2
        """
        self.clamp_params(-.99999,.99999)
        dt = self.dt
        ka  = self.ka.abs()*rate*pop[:,0]
        kfi = self.kfi.abs()*pop[:,1]
        kfr = self.kfr.abs()*pop[:,2]
        ksi = self.ksi.abs()*pop[:,2]
        ksr = self.ksr.abs()*rate*pop[:,3]
        new_pop = torch.zeros_like(pop)
        new_pop[:,0] = pop[:,0] + dt*(-ka + kfr)
        new_pop[:,1] = pop[:,1] + dt*(-kfi + ka)
        new_pop[:,2] = pop[:,2] + dt*(-kfr - ksi + kfi + ksr)
        new_pop[:,3] = pop[:,3] + dt*(-ksr + ksi)
        return new_pop[:,1], new_pop

# For retinotopic models
class OneHot(nn.Module):
    """
    Layer to choose a single output activation as a cell's response

    shape: list-like or int (num channels, hieght, width)
        the shape of the incoming activations without the batch size.

    Attributes:
        w: nn Parameter (C,L)
            actual weights for each channel
        prob: FloatTensor (C,L)
            probability form of w (normed before every forward pass)
    """
    def __init__(self,shape, rand_init=False):
        super(OneHot, self).__init__()
        assert len(shape) == 3, "Shape must have 3 dimensions"
        self.shape = shape
        l = shape[1]*shape[2]
        if rand_init:
            tensor = torch.rand(shape[0],l)
        else:
            tensor = torch.ones(shape[0],l)/l
        self.w = nn.Parameter(tensor)
        self.prob = None


    def forward(self, x):
        """
        x: FloatTensor (B,C,H,W)
        """
        mins = torch.min(self.w,dim=-1)[0][:,None] # (C,1)
        positive = self.w - mins # all vals are positive (L,C)
        # Create probabilities for each channel (C,L)
        self.prob = torch.einsum("cl,c->cl",positive,positive.sum(-1))

        x = x.reshape(*x.shape[:2],-1)
        # Multiply by probabilities and sum accross channels
        out = torch.einsum("bcl,cl->bc",x,self.prob)

        return out

class ConsolidatedOneHot(nn.Module):
    """
    Generates a new one-hot layer that works that works for collapsed filters
        Example: If you trained a model with 4 cells, then look at similarites,
        and find that 2 are of cell-type A, and 2 are of cell-type B, you would
        'collapse' those filters into 2, one for A, one for B, but there still
        needs to be 4 one-hot layers that point to their respective filters

    IMPORTANT: This implementation initializes one-hot layers as one-hots, as
        opposed to randomly initializing like we did in the beginnign.

    shape: list-like or int
        the height/width of the activations

    locations: locations that previous epoch has 'chosen' to be 1.0,
        this is only one way to initialize, could also randomize, or just take
        the previous one-hot weights and probabilities

    labels: cluster assignments (i.e.) which filters should each one-hot be
        corresponded too


    """
    def __init__(self,shape,locations,labels):
        super().__init__()
        self.shape=shape
        self.labels = labels
        self.w = nn.Parameter(torch.zeros(shape[0],shape[1]*shape[2]))
        for i,loc in enumerate(locations):
            self.w[i,loc] = 1.0
        self.prob = None

    def forward(self,x):
        positive = self.w - torch.min(self.w,1)[0][:,None]
        normed = positive.permute(1,0)/positive.sum(-1)
        self.prob = normed.permute(1,0)

        x = x.reshape(*x.shape[:2],-1)
        resps = []

        for i,l in enumerate(self.labels):
            out = torch.sum(self.prob[i]*x[:,l,:],dim=-1)
            resps.append(out)
        out = torch.stack(resps)
        return out.transpose(1,0)

def semantic_loss(prob):
    """
    Loss fxn that encourages one-hot vectorization

    prob: from OneHot.prob - will be of shape
        [NUMBER_OF_CHANNELS x KERNEL_SIZE^2] (because its flattened)

    I tried many implementations and this is the fastest

    """
    wmc_tmp = torch.zeros_like(prob)

    for i in range(prob.shape[1]):
        one_situation = torch.ones_like(prob).scatter_(1,torch.zeros_like(prob[:,0]).fill_(i).unsqueeze(-1).long(),0)
        wmc_tmp[:,i] = torch.abs((one_situation - prob).prod(dim=1))

    wmc_tmp = -1.0*torch.log(wmc_tmp.sum(dim=1))
    total_loss = torch.sum(wmc_tmp)
    return total_loss


