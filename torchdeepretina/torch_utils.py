import torch
import torch.nn as nn
import numpy as np

DEVICE = torch.device("cuda:0")


def diminish_weight_magnitude(params):
    for param in params:
        divisor = float(np.prod(param.data.shape))/2
        if param.data is not None and divisor >= 0:
            param.data = param.data/divisor

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
        self.cuda_param = nn.Parameter(torch.ones(1)*std, requires_grad=trainable)
        self.running_std = 1
        self.momentum = momentum
    
    def forward(self, x):
        if not self.training: # No noise during evaluation
            return x
        if self.adapt:
            xstd = x.std().item()
            self.running_std = self.momentum*self.running_std + (1-self.momentum)*xstd
            self.cuda_param.data[0] = self.std*self.running_std
        if self.cuda_param.is_cuda:
            noise = self.cuda_param * torch.randn(x.size()).to(self.cuda_param.get_device())
        else:
            noise = self.cuda_param * torch.randn(x.size())
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
    def __init__(self, shape, scale=True, shift=True):
        super(AbsScaleShift, self).__init__()
        self.shape = shape
        self.scale = scale
        self.shift = shift
        self.scale_param = nn.Parameter(torch.ones(shape).float(), requires_grad=scale)
        self.shift_param= nn.Parameter(torch.zeros(shape).float(), requires_grad=shift)

    def forward(self, x):
        return x*self.scale_param.abs() + self.shift_param

    def extra_repr(self):
        return 'shape={}, scale={}, shift={}'.format(self.shape, self.scale, self.shift)

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
            return 'abs_bias={}'.format(self.abs_bias)
        except:
            return "abs_bias={}".format(True)

class AbsLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True, abs_bias=False):
        super(AbsLinear, self).__init__()
        self.bias = bias
        self.abs_bias = bias
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

class StackedConv2d(nn.Module):
    '''
    Builds argued kernel out of multiple 3x3 kernels.
    '''
    def __init__(self, in_channels, out_channels, kernel_size, bias=True):
        super(StackedConv2d, self).__init__()
        self.bias = bias
        n_filters = int((kernel_size-1)/2)
        if n_filters > 1:
            convs = [nn.Conv2d(in_channels, out_channels, 3, bias=False), nn.BatchNorm2d(out_channels), nn.ReLU()]
            for i in range(n_filters-1):
                if i == n_filters-2:
                    convs.append(nn.Conv2d(out_channels, out_channels, 3, bias=bias))
                else:
                    convs.append(nn.Conv2d(out_channels, out_channels, 3, bias=False))
                convs.append(nn.ReLU())
                convs.append(nn.BatchNorm2d(out_channels))
        else:
            convs = [nn.Conv2d(in_channels, out_channels, 3, bias=bias), nn.BatchNorm2d(out_channels), nn.ReLU()]
        self.convs = nn.Sequential(*convs)

    def forward(self, x):
        return self.convs(x)

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

class WeightNorm(nn.Module):
    def __init__(self, torch_module):
        super(WeightNorm, self).__init__()
        self.torch_module = torch_module
        torch.nn.utils.weight_norm(self.torch_module, 'weight')

    def forward(self, x):
        return self.torch_module(x)

class MeanOnlyBatchNorm1d(nn.Module):
    def __init__(self, n_units, momentum=.1, eps=1e-5):
        super(MeanOnlyBatchNorm1d, self).__init__()
        self.n_units = n_units
        self.running_mean = torch.zeros(n_units)
        self.running_var = None
        self.momentum = momentum
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(n_units).float())
        self.shift = nn.Parameter(torch.zeros(n_units).float())

    def forward(self, x):
        if self.training:
            # the argument training=False forces the use of the argued statistics in the batch_norm func
            xmean = x.mean(0)
            self.running_mean = self.running_mean*(1-self.momentum) + xmean*self.momentum
            return nn.functional.batch_norm(x, x.mean(0), torch.ones(self.running_mean.shape), 
                                            weight=self.scale, bias=self.shift, eps=0, training=False) 
        else:
            return nn.functional.batch_norm(x, self.running_mean, torch.ones(self.running_mean.shape), 
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
    def __init__(self, in_channels, out_channels, kernel_size, x_shape=(40,50,50), bias=True, noise=.05, neg_p=1):
        super(DalesSkipConnection,self).__init__()
        padding = kernel_size//2
        modules = []
        modules.append(AbsConv2d(in_channels, out_channels, kernel_size, stride=1, padding=padding, bias=bias))
        diminish_weight_magnitude(modules[-1].parameters())
        modules.append(Flatten())
        modules.append(nn.BatchNorm1d(out_channels*x_shape[-2]*x_shape[-1]))
        modules.append(GaussianNoise(noise))
        modules.append(nn.ReLU())
        modules.append(Reshape((-1, out_channels,x_shape[-2],x_shape[-1])))
        modules.append(DaleActivations(out_channels, neg_p=neg_p))
        self.sequential = nn.Sequential(*modules)

    def forward(self, x):
        fx = self.sequential(x)
        return torch.cat([x,fx], dim=1)






