import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

DEVICE = torch.device("cuda:0")

def update_shape(shape, kernel=3, padding=0, stride=1, op="conv"):
    """
    Calculates the new shape of the tensor following a convolution or deconvolution

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

class GrabUnits(nn.Module):
    def __init__(self, centers, ksizes, img_shape, n_units):
        super().__init__()
        assert len(ksizes) > 2 and centers is not None
        self.ksizes = ksizes
        self.img_shape = img_shape
        self.coords = self.centers2coords(centers,ksizes,img_shape)
        self.chans = torch.arange(n_units).long()

    def centers2coords(self, centers, ksizes, img_shape):
        """
        Assumes a stride of 1 with 0 padding in each layer.
        """
        # Each quantity is even, thus the final half_effective_ksize is odd
        half_effective_ksize = (ksizes[0]-1) + (ksizes[1]-1) + (ksizes[2]//2-1) + 1
        coords = []
        for center in centers:
            row = min(max(0,center[0]-half_effective_ksize), img_shape[1]-2*(half_effective_ksize-1))
            col = min(max(0,center[1]-half_effective_ksize), img_shape[2]-2*(half_effective_ksize-1))
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
    def __init__(self, train_off=True):
        super(Exponential, self).__init__()
        self.train_off = train_off # When train_off is true, exponential is not used during training
    
    def forward(self, x):
        if self.training and self.train_off: # Only used in eval mode when train_off is true
            return x
        return torch.exp(x)

    def extra_repr(self):
        return 'train_off={}'.format(self.train_off)

class GaussianNoise(nn.Module):
    def __init__(self, std=0.1, trainable=False, adapt=False, momentum=.95):
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
            this is the exponentially moving average factor for updating the
            activ_std. 0 uses the std of the current activations.
        """
        super(GaussianNoise, self).__init__()
        self.trainable = trainable
        self.adapt = adapt
        assert not (self.trainable and self.adapt)
        self.std = std
        self.sigma = nn.Parameter(torch.ones(1)*std, requires_grad=trainable)
        self.running_std = 1
        self.momentum = momentum
    
    def forward(self, x):
        if not self.training or self.std == 0: # No noise during evaluation
            return x
        if self.adapt:
            xstd = x.std().item()
            self.running_std = self.momentum*self.running_std + (1-self.momentum)*xstd
            self.sigma.data[0] = self.std*self.running_std
        if self.sigma.is_cuda:
            noise = self.sigma * torch.randn(x.size()).to(self.sigma.get_device())
        else:
            noise = self.sigma * torch.randn(x.size())
        return x + noise

    def extra_repr(self):
        try:
            return 'std={}, trainable={}, adapt={}, momentum={}'.format(self.std, self.trainable, self.adapt, self.momentum)
        except:
            try:
                return 'std={}, trainable={}, adapt={}'.format(self.std, self.trainable, self.adapt)
            except:
                try:
                    return 'std={}, trainable={}'.format(self.std, self.trainable)
                except:
                    return 'std={}'.format(self.std)
            
class GaussianNoise1d(nn.Module):
    def __init__(self, shape, noise=0.1, momentum=.95):
        """
        adds noise to each activation based off the batch statistics. Very similar to
        BatchNorm1d but function is adding gaussian noise.

        shape - list like, sequence (..., C, H, W) or (..., L)
            this is the shape of the incoming activations 
        noise - float
            used as the proportional value to set the std of the gaussian noise. 
            proportion is based of the std of the activations.
            gauss_std = activ_std*noise
        momentum - float (0 <= momentum < 1)
            this is the exponentially moving average factor for updating the
            activ_std. 0 uses the std of the current activations.
        """
        super(GaussianNoise1d, self).__init__()
        if type(shape) == type(int()):
            shape = [shape]
        self.noise = noise
        self.shape = shape
        self.momentum = momentum
        self.running_std = nn.Parameter(torch.ones(shape[1:]), requires_grad=False)
        self.sigma = nn.Parameter(torch.ones(shape[1:])*self.noise, requires_grad=False)
    
    def forward(self, x):
        if not self.training: # No noise during evaluation
            return x
        xstd = x.std(0)
        self.running_std.data = self.momentum*self.running_std.data + (1-self.momentum)*xstd
        self.sigma.data = self.running_std.data * self.noise
        gauss = torch.randn(x.size())
        if self.sigma.is_cuda:
            gauss = gauss.to(self.sigma.get_device())
        return x + gauss*self.sigma

    def extra_repr(self):
        return 'shape={}, noise={}, momentum={}'.format(self.shape, self.noise, self.momentum)

class GaussianNoise2d(nn.Module):
    def __init__(self, n_chans, noise=0.1, momentum=.95):
        """
        adds noise to each activation based off the batch statistics. Very similar to
        BatchNorm2d but function is adding gaussian noise.

        n_chans - float
            the number of channels in the activations 
        noise - float
            used as the proportional value to set the std of the gaussian noise. 
            proportion is based of the std of the activations.
            gauss_std = activ_std*noise
        momentum - float (0 <= momentum < 1)
            this is the exponentially moving average factor for updating the
            activ_std. 0 uses the std of the current activations.
        """
        super(GaussianNoise2d, self).__init__()
        self.noise = noise
        self.n_chans = n_chans
        self.momentum = momentum
        self.running_std = nn.Parameter(torch.ones(n_chans), requires_grad=False)
        self.sigma = nn.Parameter(torch.ones(n_chans)*self.noise, requires_grad=False)
    
    def forward(self, x):
        if not self.training: # No noise during evaluation
            return x
        # x.shape = (B, C, H, W)
        x_perm = x.permute(0,2,3,1).contiguous() # Now (B, H, W, C)
        xstd = x.view(-1, self.shape).std(0) # (C,)
        self.running_std = self.momentum*self.running_std + (1-self.momentum)*xstd
        self.sigma.data = self.running_std * self.noise
        gauss = torch.randn(x.size())
        if self.sigma.is_cuda:
            gauss = gauss.to(self.sigma.get_device())
        return (x_perm + gauss*self.sigma).permute(0,3,1,2).contiguous() # Return x with shape (B, C, H, W)

    def extra_repr(self):
        return 'n_chans={}, noise={}, momentum={}'.format(self.n_chans, self.noise, self.momentum)

class ScaleShift(nn.Module):
    def __init__(self, shape, scale=True, shift=True):
        super(ScaleShift, self).__init__()
        self.shape = shape
        self.scale = scale
        self.shift = shift
        self.scale_param = nn.Parameter(torch.ones(shape).float(), requires_grad=scale)
        self.shift_param= nn.Parameter(torch.zeros(shape).float(), requires_grad=shift)

    def forward(self, x):
        return x*self.scale_param + self.shift_param

    def extra_repr(self):
        return 'shape={}, scale={}, shift={}'.format(self.shape, self.scale, self.shift)

class AbsScaleShift(nn.Module):
    def __init__(self, shape, scale=True, shift=True, abs_shift=False):
        super(AbsScaleShift, self).__init__()
        self.shape = shape
        self.scale = scale
        self.shift = shift
        self.abs_shift = abs_shift
        self.scale_param = nn.Parameter(torch.ones(shape).float(), requires_grad=scale)
        self.shift_param= nn.Parameter(torch.zeros(shape).float(), requires_grad=shift)

    def forward(self, x):
        if self.abs_shift:
            shift = self.shift_param.abs()
        return x*self.scale_param.abs() + self.shift_param

    def extra_repr(self):
        return 'shape={}, scale={}, shift={}, abs_shift={}'.format(self.shape, self.scale, self.shift, self.abs_shift)

class DaleActivations(nn.Module):
    """
    For the full Dale effect, will also need to use AbsConv2d and AbsLinear layers.
    """
    def __init__(self, n_chan, neg_p=.33):
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
        return 'n_chan={}, neg_p={}, n_neg_chan={}'.format(self.n_chan, self.neg_p, self.n_neg_chan)

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
        return torch.nn.functional.batch_norm(x, self.running_mean.data, self.running_var.data,
                                            weight=self.scale, bias=self.shift, eps=self.eps, 
                                            momentum=self.momentum, training=self.training)

    def extra_repr(self):
        return 'bias={}, momentum={}, eps={}'.format(self.bias, self.momentum, self.eps)

class AbsBatchNorm1d(nn.Module):
    def __init__(self, n_units, bias=True, abs_bias=False, momentum=.1, eps=1e-5):
        super(AbsBatchNorm1d, self).__init__()
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
        assert len(x.shape) == 2
        self.shift.requires_grad = self.bias
        if self.abs_bias:
            return torch.nn.functional.batch_norm(x, self.running_mean.data, self.running_var.data,
                                            weight=self.scale.abs(), bias=self.shift.abs(), eps=self.eps, 
                                            momentum=self.momentum, training=self.training)
        return torch.nn.functional.batch_norm(x, self.running_mean.data, self.running_var.data,
                                            weight=self.scale.abs(), bias=self.shift, eps=self.eps, 
                                            momentum=self.momentum, training=self.training)

    def extra_repr(self):
        return 'bias={}, abs_bias={}, momentum={}, eps={}'.format(self.bias, self.abs_bias, self.momentum, self.eps)
                                            
class AbsBatchNorm2d(nn.Module):
    def __init__(self, n_units, bias=True, abs_bias=False, momentum=.1, eps=1e-5):
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
        if self.abs_bias:
            return torch.nn.functional.batch_norm(x, self.running_mean.data, self.running_var.data,
                                            weight=self.scale.abs(), bias=self.shift.abs(), eps=self.eps, 
                                            momentum=self.momentum, training=self.training)
        return torch.nn.functional.batch_norm(x, self.running_mean.data, self.running_var.data,
                                            weight=self.scale.abs(), bias=self.shift, 
                                            eps=self.eps, momentum=self.momentum, 
                                            training=self.training)

    def extra_repr(self):
        return 'bias={}, abs_bias={}, momentum={}, eps={}'.format(self.bias, self.abs_bias, self.momentum, self.eps)
                                            
class AbsConvTranspose2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True, abs_bias=False):
        super().__init__()
        self.abs_bias = abs_bias
        self.bias = bias
        self.conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias)

    def forward(self, x):
        bias = None
        if self.bias and self.abs_bias:
            bias = self.conv.bias.abs()
        elif self.bias:
            bias = self.conv.bias
        return nn.functional.conv_transpose2d(x, self.conv.weight.abs(), bias,
                                           self.conv.stride, self.conv.padding)

    def extra_repr(self):
        try:
            return 'bias={}, abs_bias={}'.format(self.bias, self.abs_bias)
        except:
            return "abs_bias={}".format(True)

class AbsConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, abs_bias=False):
        super(AbsConv2d, self).__init__()
        self.abs_bias = abs_bias
        self.bias = bias
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)

    def forward(self, x):
        if self.bias:
            try:
                if not self.abs_bias:
                    return nn.functional.conv2d(x, self.conv.weight.abs(), self.conv.bias, 
                                                            self.conv.stride, self.conv.padding, 
                                                            self.conv.dilation, self.conv.groups)
            except:
                pass
            return nn.functional.conv2d(x, self.conv.weight.abs(), self.conv.bias.abs(), 
                                                    self.conv.stride, self.conv.padding, 
                                                    self.conv.dilation, self.conv.groups)
        else:
            return nn.functional.conv2d(x, self.conv.weight.abs(), None, 
                                                self.conv.stride, self.conv.padding, 
                                                self.conv.dilation, self.conv.groups)
    def extra_repr(self):
        try:
            return 'bias={}, abs_bias={}'.format(self.bias, self.abs_bias)
        except:
            return "abs_bias={}".format(True)

class SqrConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, abs_bias=False):
        super().__init__()
        self.abs_bias = abs_bias
        self.bias = bias
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)

    def forward(self, x):
        if self.bias:
            try:
                if not self.abs_bias:
                    return nn.functional.conv2d(x, self.conv.weight**2, self.conv.bias, 
                                                            self.conv.stride, self.conv.padding, 
                                                            self.conv.dilation, self.conv.groups)
            except:
                pass
            return nn.functional.conv2d(x, self.conv.weight**2, self.conv.bias.abs(), 
                                                    self.conv.stride, self.conv.padding, 
                                                    self.conv.dilation, self.conv.groups)
        else:
            return nn.functional.conv2d(x, self.conv.weight**2, None, 
                                                self.conv.stride, self.conv.padding, 
                                                self.conv.dilation, self.conv.groups)
    def extra_repr(self):
        try:
            return 'bias={}, abs_bias={}'.format(self.bias, self.abs_bias)
        except:
            return "abs_bias={}".format(True)

class DecoupledLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True, eps=1e-5):
        super(DecoupledLinear, self).__init__()
        self.eps = eps
        linear = nn.Linear(in_features, out_features, bias=bias)
        self.weight = nn.Parameter(linear.weight.transpose(1,0))
        self.bias = nn.Parameter(linear.bias)
        self.scale = nn.Parameter(torch.ones(out_features))

    def forward(self, x):
        norms = torch.norm(self.weight, 2, dim=0)
        normed_weight = (self.weight / (norms + self.eps)).transpose(1,0)
        fx = nn.functional.linear(x, normed_weight, self.bias)
        return fx*self.scale
    
    def extra_repr(self):
        return "bias={}".format(self.bias is not None)

class AbsLinear(nn.Module):
    """
    Performs a fully connected operation in which the weights are all positive.
    """
    def __init__(self, in_features, out_features, bias=True, abs_bias=False):
        super(AbsLinear, self).__init__()
        self.bias = bias
        self.abs_bias = abs_bias
        self.linear = nn.Linear(in_features, out_features, bias=bias)

    def forward(self, x):
        if self.abs_bias:
            return nn.functional.linear(x, self.linear.weight.abs(), 
                                            self.linear.bias.abs())
        else:
            return nn.functional.linear(x, self.linear.weight.abs(), 
                                                    self.linear.bias)
    
    def extra_repr(self):
        try:
            return 'bias={}, abs_bias={}'.format(self.bias, self.abs_bias)
        except:
            return "bias={}, abs_bias={}".format(self.bias, True)

class AbsLinearStackedConv2d(nn.Module):
    '''
    Builds argued kernel out of multiple 3x3 kernels without added nonlinearities.
    '''
    def __init__(self, in_channels, out_channels, kernel_size, bias=True, abs_bnorm=False, conv_bias=False, drop_p=0, padding=0):
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
            convs = [AbsConv2d(in_channels, out_channels, 3, bias=conv_bias)]
            if abs_bnorm:
                convs.append(AbsBatchNorm2d(out_channels))
            if drop_p > 0:
                convs.append(nn.Dropout(drop_p))
            for i in range(n_filters-1):
                if i == n_filters-2:
                    convs.append(AbsConv2d(out_channels, out_channels, 3, bias=bias))
                else:
                    convs.append(AbsConv2d(out_channels, out_channels, 3, bias=conv_bias))
                    if abs_bnorm:
                        convs.append(AbsBatchNorm2d(out_channels))
                    if drop_p > 0:
                        convs.append(nn.Dropout(drop_p))
        else:
            convs = [AbsConv2d(in_channels, out_channels, 3, bias=bias)]
        self.convs = nn.Sequential(*convs)

    def forward(self, x):
        x = F.pad(x, (self.padding, self.padding, self.padding, self.padding))
        return self.convs(x)

    def extra_repr(self):
        try:
            return 'bias={}, abs_bnorm={}'.format(self.bias, self.abs_bnorm)
        except:
            return "bias={}, abs_bnorm={}".format(self.bias, True)

class SqrLinearStackedConv2d(nn.Module):
    '''
    Builds argued kernel out of multiple 3x3 kernels without added nonlinearities.
    '''
    def __init__(self, in_channels, out_channels, kernel_size, bias=True, abs_bnorm=False, conv_bias=False, drop_p=0, padding=0):
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
            convs = [SqrConv2d(in_channels, out_channels, 3, bias=conv_bias)]
            if abs_bnorm:
                convs.append(AbsBatchNorm2d(out_channels))
            if drop_p > 0:
                convs.append(nn.Dropout(drop_p))
            for i in range(n_filters-1):
                if i == n_filters-2:
                    convs.append(SqrConv2d(out_channels, out_channels, 3, bias=bias))
                else:
                    convs.append(SqrConv2d(out_channels, out_channels, 3, bias=conv_bias))
                    if abs_bnorm:
                        convs.append(AbsBatchNorm2d(out_channels))
                    if drop_p > 0:
                        convs.append(nn.Dropout(drop_p))
        else:
            convs = [SqrConv2d(in_channels, out_channels, 3, bias=bias)]
        self.convs = nn.Sequential(*convs)

    def forward(self, x):
        x = F.pad(x, (self.padding, self.padding, self.padding, self.padding))
        return self.convs(x)

    def extra_repr(self):
        try:
            return 'bias={}, abs_bnorm={}'.format(self.bias, self.abs_bnorm)
        except:
            return "bias={}, abs_bnorm={}".format(self.bias, True)

class LinearStackedConv2d(nn.Module):
    '''
    Builds argued kernel out of multiple KxK kernels without added nonlinearities.
    '''
    def __init__(self, in_channels, out_channels, kernel_size, bias=True, stack_ksize=3, stack_chan=None, abs_bnorm=False, conv_bias=False, drop_p=0, padding=0):
        super(LinearStackedConv2d, self).__init__()
        assert kernel_size % 2 == 1 # kernel must be odd
        assert kernel_size > 1 # kernel must be greater than 1
        self.ksize = kernel_size
        self.stack_ksize = stack_ksize
        assert self.stack_ksize <= self.ksize
        self.bias = bias
        self.conv_bias = conv_bias
        self.abs_bnorm = abs_bnorm
        self.padding = padding
        self.drop_p = drop_p
        self.stack_chan = out_channels if stack_chan is None else stack_chan

        n_filters = (kernel_size-self.stack_ksize)/(self.stack_ksize-1)+1
        if n_filters - int(n_filters) > 0:
            effective = self.stack_ksize+int(n_filters-1)*(self.stack_ksize-1)
            remaining = (kernel_size-effective)
            self.last_ksize = remaining+1
            n_filters += 1
        else:
            self.last_ksize = self.stack_ksize
        n_filters = int(n_filters)

        if n_filters > 1:
            convs = [nn.Conv2d(in_channels, self.stack_chan, self.stack_ksize, bias=conv_bias)]
            if abs_bnorm:
                convs.append(AbsBatchNorm2d(self.stack_chan))
            if drop_p > 0:
                convs.append(nn.Dropout(drop_p))
            for i in range(n_filters-1):
                if i == n_filters-2:
                    convs.append(nn.Conv2d(self.stack_chan, out_channels, self.last_ksize, bias=bias))
                else:
                    convs.append(nn.Conv2d(self.stack_chan, self.stack_chan, self.stack_ksize, bias=conv_bias))
                    if abs_bnorm:
                        convs.append(AbsBatchNorm2d(out_channels))
                    if drop_p > 0:
                        convs.append(nn.Dropout(drop_p))
        else:
            convs = [nn.Conv2d(in_channels, out_channels, self.stack_ksize, bias=bias)]
        self.convs = nn.Sequential(*convs)

    def forward(self, x):
        x = F.pad(x, (self.padding, self.padding, self.padding, self.padding))
        return self.convs(x)

    def extra_repr(self):
        try:
            return 'bias={}, abs_bnorm={}'.format(self.bias, self.abs_bnorm)
        except:
            return "bias={}, abs_bnorm={}".format(self.bias, True)

class OneToOneLinearStackedConv2d(nn.Module):
    '''
    Builds argued kernel out of multiple 3x3 kernels without added nonlinearities. No crosstalk between intermediate 
    channels.
    '''
    def __init__(self, in_channels, out_channels, kernel_size, bias=True, conv_bias=False, padding=0):
        super().__init__()
        assert kernel_size % 2 == 1 # kernel must be odd
        self.ksize = kernel_size
        self.bias = bias
        self.conv_bias = conv_bias
        self.padding = padding
        n_filters = int((kernel_size-1)/2)
        assert n_filters > 1
        self.first_conv = nn.Conv2d(in_channels, out_channels, 3, bias=conv_bias)
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
        x = F.pad(x, (self.padding, self.padding, self.padding, self.padding))
        fx = self.first_conv(x)
        outs = []
        for chan,seq in enumerate(self.seqs):
            outs.append(seq(fx[:,chan:chan+1]))
        fx = torch.cat(outs,dim=1)
        return fx

    def extra_repr(self):
        return 'bias={}, conv_bias={}, padding={}'.format(self.bias, self.conv_bias, self.padding)

class StackedConv2d(nn.Module):
    '''
    Builds argued kernel out of multiple 3x3 kernels.
    '''
    def __init__(self, in_channels, out_channels, kernel_size, bias=True, abs_bnorm=False, conv_bias=False, drop_p=0, legacy=False):
        super(StackedConv2d, self).__init__()
        self.bias = bias
        n_filters = int((kernel_size-1)/2)
        if n_filters > 1:
            if not abs_bnorm or legacy:
                convs = [nn.Conv2d(in_channels, out_channels, 3, bias=conv_bias), nn.BatchNorm2d(out_channels), nn.ReLU()]
            else:
                convs = [nn.Conv2d(in_channels, out_channels, 3, bias=conv_bias), AbsBatchNorm2d(out_channels), nn.ReLU()]
            for i in range(n_filters-1):
                if i == n_filters-2:
                    convs.append(nn.Conv2d(out_channels, out_channels, 3, bias=bias))
                else:
                    convs.append(nn.Conv2d(out_channels, out_channels, 3, bias=conv_bias))
                convs.append(nn.ReLU())
                if abs_bnorm:
                    convs.append(AbsBatchNorm2d(out_channels))
                else:
                    convs.append(nn.BatchNorm2d(out_channels))
                if drop_p > 0:
                    convs.append(nn.Dropout(drop_p))
        else:
            convs = [nn.Conv2d(in_channels, out_channels, 3, bias=bias), nn.BatchNorm2d(out_channels), nn.ReLU()]
        self.convs = nn.Sequential(*convs)

    def forward(self, x):
        return self.convs(x)

class ConvRNNCell(nn.Module):
    def __init__(self, in_channels, out_channels, rnn_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True):
        super(ConvRNNCell, self).__init__()
        self.in_chans = in_channels
        self.out_chans = out_channels
        self.rnn_chans = rnn_channels
        self.ksize = kernel_size
        self.stride = stride
        self.padding = padding
        self.bias = bias
        self.conv = nn.Conv2d(in_channels+rnn_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        assert kernel_size % 2 == 1 # Must have odd kernel size
        self.rnn_padding = (kernel_size-1)//2
        self.rnn_conv = nn.Conv2d(in_channels+rnn_channels, rnn_channels, kernel_size, stride, self.rnn_padding, bias)
    
    def forward(self, x, h):
        """
        x: torch tensor (B, IN_CHAN, H, W)
            the new input
        h: torch tensor (B, RNN_CHAN, H, W) 
            the recurrent input
        """
        ins = torch.cat([x,h], dim=1)
        outs = self.conv(ins)
        h_new = self.rnn_conv(ins)
        return outs, h_new

class DalesAmacRNNSimple(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding=0, bias=True, stackconvs=True):
        super().__init__()
        self.in_chans = in_channels
        self.out_chans = out_channels
        self.ksize = kernel_size
        self.padding = padding
        self.bias = bias
        self.stackconvs = stackconvs
        self.rnn_padding = (kernel_size-1)//2
        if stackconvs:
            self.amacrine = AbsLinearStackedConv2d(in_channels, out_channels, kernel_size=kernel_size,
                                                                                    padding=self.rnn_padding,
                                                                                    bias=bias, drop_p=0)
            self.conv = AbsLinearStackedConv2d(in_channels, out_channels, kernel_size=kernel_size, 
                                                                               bias=bias, drop_p=0)
        else:
            self.amacrine = AbsConv2d(in_channels, out_channels, kernel_size, padding=self.rnn_padding, bias=bias)
            self.conv = AbsConv2d(in_channels, out_channels, kernel_size, bias=bias)
        assert kernel_size % 2 == 1 # Must have odd kernel size
        self.bipolar = AbsConvTranspose2d(out_channels, in_channels, kernel_size, padding=0, bias=True)
    
    def forward(self, x, h):
        """
        Creates a unique h for both self feedback and bipolar feedback.

        x: torch tensor (B, IN_CHAN, H, W)
            the new input
        h: torch tensor (B, OUT_CHAN, H1, W1) 
            the recurrent input for the amacrine cells
        """
        outs = self.conv(x)
        outs = outs + h
        bipolar_feedback = -self.bipolar(outs).abs()
        h_new = -self.amacrine(outs).abs()
        return outs, bipolar_feedback, h_new

class DalesAmacRNN(nn.Module):
    def __init__(self, in_channels, out_channels, rnn_channels, kernel_size, padding=0, bias=True, stackconvs=True):
        super().__init__()
        self.in_chans = in_channels
        self.out_chans = out_channels
        self.rnn_chans = rnn_channels
        self.ksize = kernel_size
        self.padding = padding
        self.bias = bias
        self.stackconvs = stackconvs
        if stackconvs:
            print("Stride ignored in AmacRNN convolution due to linear stacked conv choice")
            self.conv = AbsLinearStackedConv2d(in_channels+rnn_channels, out_channels, kernel_size=kernel_size, 
                                                                                        bias=bias, drop_p=0)
        else:
            self.conv = AbsConv2d(in_channels+rnn_channels, out_channels, kernel_size, bias=bias)
        assert kernel_size % 2 == 1 # Must have odd kernel size

        self.rnn_padding = (kernel_size-1)//2
        self.bipolar = AbsConvTranspose2d(out_channels, in_channels, kernel_size, padding=0, bias=True)
        if stackconvs:
            self.rnn_conv = AbsLinearStackedConv2d(in_channels+rnn_channels, rnn_channels*2, kernel_size=kernel_size, 
                                                                                        padding=self.rnn_padding,
                                                                                        bias=True)
            self.rnn_conv = nn.Sequential(self.rnn_conv, nn.Sigmoid())
            self.tan_conv = AbsLinearStackedConv2d(in_channels+rnn_channels, rnn_channels, kernel_size=kernel_size, 
                                                                                        padding=self.rnn_padding,
                                                                                        bias=True)
            self.tan_conv = nn.Sequential(self.tan_conv, nn.Tanh())
        else:
            self.rnn_conv = nn.Sequential(AbsConv2d(in_channels+rnn_channels, rnn_channels*2, kernel_size=kernel_size, 
                                                                                    padding=self.rnn_padding, bias=True),
                                                                                                            nn.Sigmoid())
            self.tan_conv = nn.Sequential(AbsConv2d(in_channels+rnn_channels, rnn_channels, kernel_size=kernel_size, 
                                                                                padding=self.rnn_padding, bias=True),
                                                                                                          nn.Tanh())
    
    def forward(self, x, h):
        """
        Creates a unique h for both self feedback and bipolar feedback.

        x: torch tensor (B, IN_CHAN, H, W)
            the new input
        h: torch tensor (B, RNN_CHAN, H, W) 
            the recurrent input for the amacrine cells
        """
        ins = torch.cat([x,h], dim=1)
        outs = self.conv(ins)
        bipolar_feedback = self.bipolar(-outs.abs()) # Need negative outs here for inhibitory effect
        temp = self.rnn_conv(ins)
        z,r = (temp[:,:self.rnn_chans], temp[:,self.rnn_chans:])
        tanin = torch.cat([x,h*r], dim=1)
        tanout = self.tan_conv(tanin)
        h_new = -(z*h + (1-z)*tanout).abs()
        return outs, bipolar_feedback, h_new

class AmacRNNFull(nn.Module):
    def __init__(self, in_channels, out_channels, rnn_channels, kernel_size, padding=0, bias=True, stackconvs=True):
        super().__init__()
        self.in_chans = in_channels
        self.out_chans = out_channels
        self.rnn_chans = rnn_channels
        self.ksize = kernel_size
        self.padding = padding
        self.bias = bias
        self.stackconvs = stackconvs
        if stackconvs:
            print("Stride ignored in AmacRNN convolution due to linear stacked conv choice")
            self.conv = LinearStackedConv2d(in_channels+rnn_channels, out_channels, kernel_size=kernel_size, 
                                                                                        bias=bias, drop_p=0)
        else:
            self.conv = nn.Conv2d(in_channels+rnn_channels, out_channels, kernel_size, bias=bias)
        assert kernel_size % 2 == 1 # Must have odd kernel size
        self.rnn_padding = (kernel_size-1)//2
        self.bipolar = nn.ConvTranspose2d(out_channels, in_channels, kernel_size, padding=0, bias=True)
        if stackconvs:
            self.rnn_conv = LinearStackedConv2d(in_channels+rnn_channels, rnn_channels*2, kernel_size=kernel_size, 
                                                                                        padding=self.rnn_padding,
                                                                                        bias=True)
            self.rnn_conv = nn.Sequential(self.rnn_conv, nn.Sigmoid())
            self.tan_conv = LinearStackedConv2d(in_channels+rnn_channels, rnn_channels, kernel_size=kernel_size, 
                                                                                        padding=self.rnn_padding,
                                                                                        bias=True)
            self.tan_conv = nn.Sequential(self.tan_conv, nn.Tanh())
        else:
            self.rnn_conv = nn.Sequential(nn.Conv2d(in_channels+rnn_channels, rnn_channels*2, kernel_size=kernel_size, 
                                                                                    padding=self.rnn_padding, bias=True),
                                                                                                            nn.Sigmoid())
            self.tan_conv = nn.Sequential(nn.Conv2d(in_channels+rnn_channels, rnn_channels, kernel_size=kernel_size, 
                                                                                padding=self.rnn_padding, bias=True),
                                                                                                          nn.Tanh())
    
    def forward(self, x, h):
        """
        Creates a unique h for both self feedback and bipolar feedback.

        x: torch tensor (B, IN_CHAN, H, W)
            the new input
        h: torch tensor (B, RNN_CHAN, H, W) 
            the recurrent input for the amacrine cells
        """
        ins = torch.cat([x,h], dim=1)
        outs = self.conv(ins)
        bipolar_feedback = -self.bipolar(outs).abs()
        temp = self.rnn_conv(ins)
        z,r = (temp[:,:self.rnn_chans], temp[:,self.rnn_chans:])
        tanin = torch.cat([x,h*r], dim=1)
        tanout = self.tan_conv(tanin)
        h_new = z*h + (1-z)*tanout
        return outs, bipolar_feedback, h_new

class ConvGRUCell(nn.Module):
    def __init__(self, in_channels, out_channels, rnn_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True):
        super().__init__()
        self.in_chans = in_channels
        self.out_chans = out_channels
        self.rnn_chans = rnn_channels
        self.ksize = kernel_size
        self.stride = stride
        self.padding = padding
        self.bias = bias
        self.conv = nn.Conv2d(in_channels+rnn_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        assert kernel_size % 2 == 1 # Must have odd kernel size
        self.rnn_padding = (kernel_size-1)//2
        self.rnn_conv = nn.Sequential(nn.Conv2d(in_channels+rnn_channels, rnn_channels*2, kernel_size, stride, self.rnn_padding, bias=True),
                                                                                                                          nn.Sigmoid())
        self.tan_conv = nn.Sequential(nn.Conv2d(in_channels+rnn_channels, rnn_channels, kernel_size, stride, self.rnn_padding, bias=True),
                                                                                                                          nn.Tanh())
    
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
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.abs(x)

class InvertSign(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return -x

class Add(nn.Module):
    def __init__(self, additive, trainable=False):
        super().__init__()
        self.trainable = trainable
        self.additive = nn.Parameter(torch.ones(1)*additive, requires_grad=trainable)

    def forward(self, x):
        if not self.trainable and self.additive.requires_grad:
            self.additive.requires_grad = False
        return x+self.additive

class Clamp(nn.Module):
    def __init__(self, low, high):
        super().__init__()
        self.low = low
        self.high = high

    def forward(self, x):
        return torch.clamp(x, self.low, self.high)

class Multiply(nn.Module):
    def __init__(self, multiplier, trainable=False):
        super().__init__()
        self.trainable = trainable
        self.multiplier = nn.Parameter(torch.ones(1)*multiplier, requires_grad=trainable)

    def forward(self, x):
        if not self.trainable and self.multiplier.requires_grad:
            self.multiplier.requires_grad = False
        return x+self.multiplier

class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        return x.view(x.shape[0], -1)

class Reshape(nn.Module):
    def __init__(self, shape):
        super(Reshape, self).__init__()
        self.shape = shape

    def forward(self, x):
        return x.view(self.shape)

    def extra_repr(self):
        return "shape={}".format(self.shape)

class WeightNorm(nn.Module):
    def __init__(self, torch_module):
        super(WeightNorm, self).__init__()
        self.torch_module = torch_module
        torch.nn.utils.weight_norm(self.torch_module, 'weight')

    def forward(self, x):
        return self.torch_module(x)

class MeanOnlyBatchNorm1d(nn.Module):
    def __init__(self, n_units, momentum=.1, eps=1e-5, scale=True, shift=True):
        super(MeanOnlyBatchNorm1d, self).__init__()
        self.n_units = n_units
        self.use_scale = scale
        self.use_shift = shift
        self.running_mean = nn.Parameter(torch.zeros(n_units), requires_grad=False)
        self.running_var = nn.Parameter(torch.ones(n_units), requires_grad=False)
        self.momentum = momentum
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(n_units).float())
        self.shift = nn.Parameter(torch.zeros(n_units).float())

    def forward(self, x):
        self.scale.requires_grad = self.scale.requires_grad and self.use_scale
        self.shift.requires_grad = self.shift.requires_grad and self.use_shift
        if self.training:
            # the argument training=False forces the use of the argued statistics in the batch_norm func
            xmean = x.mean(0)
            self.running_mean.data = self.running_mean.data*(1-self.momentum) + xmean*self.momentum
            return nn.functional.batch_norm(x, x.mean(0).data, self.running_var.data, 
                                            weight=self.scale, bias=self.shift, eps=0, training=False) 
        else:
            return nn.functional.batch_norm(x, self.running_mean.data, self.running_var.data, 
                        weight=self.scale, bias=self.shift, training=False)

class SplitConv2d(nn.Module):
    """
    Performs parallel convolutional operations on the input.
    Must have each convolution layer return the same shaped
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
    def __init__(self, in_channels, out_channels, kernel_size, bias=True):
        super(SkipConnection,self).__init__()
        padding = kernel_size//2
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=padding, bias=bias)
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
    def __init__(self, in_channels, out_channels, kernel_size, x_shape=(40,50,50), bias=True, noise=.05):
        super(SkipConnectionBN,self).__init__()
        padding = kernel_size//2
        modules = []
        modules.append(nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=padding, bias=bias))
        modules.append(Flatten())
        modules.append(nn.BatchNorm1d(out_channels*x_shape[-2]*x_shape[-1]))
        modules.append(GaussianNoise(noise))
        modules.append(nn.ReLU())
        modules.append(Reshape((out_channels,x_shape[-2],x_shape[-1])))
        self.sequential = nn.Sequential(*modules)

    def forward(self, x):
        fx = self.sequential(x)
        return torch.cat([x,fx], dim=1)

class DalesSkipConnection(nn.Module):
    """
    Performs an absolute value conv2d and returns the output stacked with  
    the original input.
    """
    def __init__(self, in_channels, out_channels, kernel_size, x_shape=(40,50,50), bias=True, noise=.05, neg_p=1, bn_moment=0.1):
        super(DalesSkipConnection,self).__init__()
        padding = kernel_size//2
        modules = []
        modules.append(AbsConv2d(in_channels, out_channels, kernel_size, stride=1, padding=padding, bias=bias))
        diminish_weight_magnitude(modules[-1].parameters())
        modules.append(Flatten())
        modules.append(nn.BatchNorm1d(out_channels*x_shape[-2]*x_shape[-1], momentum=bn_moment))
        modules.append(GaussianNoise(noise))
        modules.append(nn.ReLU())
        modules.append(Reshape((-1, out_channels,x_shape[-2],x_shape[-1])))
        modules.append(DaleActivations(out_channels, neg_p=neg_p))
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

class RunningNorm1d(nn.Module):
    pass
    #def __init__(self, n_units, momentum=.1):
    #    super().__init__()
    #    self.n_units = n_units
    #    self.momentum = momentum
    #    self.running_mean = nn.Parameter(torch.zeros(n_units))
    #    self.running_var = nn.Parameter(torch.ones(n_units))
    #    #self.weight = nn.Parameter(torch.ones(n_units).float())
    #    #self.bias = nn.Parameter(torch.zeros(n_units).float())

    #def forward(self, x):
    #    self.running_mean = self.running_mean
    #    return x

    #def extra_repr(self):
    #    return 'n_units={}, momentum={}'.format(self.n_units, self.momentum)

class Kinetics_channel(nn.Module):
    def __init__(self, chan=8, dt=0.001):
        super().__init__()
        self.ka = nn.Parameter(torch.rand(chan, 1).abs()/10)
        self.kfi = nn.Parameter(torch.rand(chan, 1).abs()/10)
        self.kfr = nn.Parameter(torch.rand(chan, 1).abs()/10)
        self.ksi = nn.Parameter(torch.rand(chan, 1).abs()/10)
        self.ksr = nn.Parameter(torch.rand(chan, 1).abs()/10)
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
        rate - FloatTensor (B, C, N)
            firing rates
        pop - FloatTensor (B, S, C, N)
            populations should have 4 states for each neuron.
            States should be:
                0: R
                1: A
                2: I1
                3: I2
        """
        #self.clamp_params(-.99999, .99999)
        dt = self.dt
        ka  = self.ka.abs() * rate * pop[:, 0]
        kfi = self.kfi.abs() * pop[:,1]
        kfr = self.kfr.abs() * pop[:,2]
        ksi = self.ksi.abs() * pop[:,2]
        ksr = self.ksr.abs() * rate*pop[:, 3]
        new_pop = torch.zeros_like(pop)
        new_pop[:, 0] = pop[:, 0] + dt * (- ka + kfr)
        new_pop[:, 1] = pop[:, 1] + dt * (- kfi + ka)
        new_pop[:, 2] = pop[:, 2] + dt * (- kfr - ksi + kfi + ksr)
        new_pop[:, 3] = pop[:, 3] + dt * (- ksr + ksi)
        return new_pop[:, 1], new_pop
    
class Temperal_Filter(nn.Module):
    def __init__(self, tem_len, spatial):
        super().__init__()
        self.spatial = spatial
        spatial_dims = np.ones(spatial).astype(np.int32).tolist()
        self.filter = nn.Parameter(torch.rand(tem_len, *spatial_dims))

    def forward(self, x):
        out = (x * self.filter).sum(axis=-self.spatial-1)
        return out
    
class Chan_Temperal_Filter(nn.Module):
    def __init__(self, chan, tem_len, spatial):
        super().__init__()
        self.spatial = spatial
        spatial_dims = np.ones(spatial).astype(np.int32).tolist()
        self.filter = nn.Parameter(torch.rand(chan, tem_len, *spatial_dims))

    def forward(self, x):
        out = (x * self.filter).sum(axis=-self.spatial-1)
        return out

class LinearStackedConv3d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stack_ksize=3, stack_chan=None, 
                 conv_bias=True, drop_p=0):
        super(LinearStackedConv3d, self).__init__()
        assert kernel_size[1] % 2 == 1 # kernel must be odd
        assert kernel_size[1] > 1 # kernel must be greater than 1
        self.ksize = kernel_size
        self.stack_ksize = stack_ksize
        self.conv_bias = conv_bias
        self.drop_p = drop_p
        self.stack_chan = out_channels if stack_chan is None else stack_chan

        n_filters = (kernel_size[1]-self.stack_ksize)/(self.stack_ksize-1)+1
        assert n_filters == int(n_filters)
        n_filters = int(n_filters)

        convs = [nn.Conv3d(in_channels, self.stack_chan, 
                           [kernel_size[0], self.stack_ksize, self.stack_ksize], 
                           bias=self.conv_bias)]
        if drop_p > 0:
            convs.append(nn.Dropout(drop_p))
        for i in range(n_filters-1):
            convs.append(nn.Conv3d(self.stack_chan, self.stack_chan, 
                                   [1, self.stack_ksize, self.stack_ksize], bias=self.conv_bias))
            if drop_p > 0:
                convs.append(nn.Dropout(drop_p))
                
        self.convs = nn.Sequential(*convs)

    def forward(self, x):
        return self.convs(x)
    
class OuterProduct3DFilter(nn.Module):
    def __init__(self, shape, chan, n_units):
        super().__init__()
        modules = []
        modules.append(Reshape((-1, chan, shape[0], shape[1], shape[2])))
        modules.append(Temperal_Filter(shape[0], 2))
        modules.append(Flatten())
        modules.append(nn.Linear(chan*shape[1]*shape[2], 
                                 n_units, bias=True))
        self.sequential = nn.Sequential(*modules)
        
    def forward(self, x):
        return self.sequential(x)
    
class MultiOuterProduct3DFilter(nn.Module):
    def __init__(self, shape, chan, n_units, n_filters):
        super().__init__()
        assert n_filters == 2 or n_filters == 3
        self.n_filters = n_filters
        self.filter1 = OuterProduct3DFilter(shape, chan, n_units)
        self.filter2 = OuterProduct3DFilter(shape, chan, n_units)
        if self.n_filters == 3:
            self.filter3 = OuterProduct3DFilter(shape, chan, n_units)
            
    def forward(self, x):
        out = self.filter1(x)
        out = out + self.filter2(x)
        if self.n_filters == 3:
            out = out + self.filter3(x)
        return out
    
class OuterProduct3DFilterEachChannel(nn.Module):
    def __init__(self, shape, chan, n_units):
        super().__init__()
        modules = []
        modules.append(Reshape((-1, chan, shape[0], shape[1], shape[2])))
        modules.append(Chan_Temperal_Filter(chan, shape[0], 2))
        modules.append(Flatten())
        modules.append(nn.Linear(chan*shape[1]*shape[2], 
                                 n_units, bias=True))
        self.sequential = nn.Sequential(*modules)
        
    def forward(self, x):
        return self.sequential(x)
                                  