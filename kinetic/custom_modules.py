import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

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

class Kinetics(nn.Module):
    def __init__(self, dt=0.01, chan=8, ka_offset=False, ksr_gain=False, k_chan=True,
                 ka=None, ka_2=None, kfi=None, kfr=None, ksi=None, ksr=None, ksr_2=None):
        super().__init__()
        if not k_chan:
            chan = 1
        else:
            pass
        self.ka_offset = ka_offset
        self.ksr_gain = ksr_gain
        self.ka = nn.Parameter(torch.rand(chan, 1).abs()/10)
        if self.ka_offset:
            self.ka_2 = nn.Parameter(torch.rand(chan, 1).abs()/10)
        self.kfi = nn.Parameter(torch.rand(chan, 1).abs()/10)
        self.kfr = nn.Parameter(torch.rand(chan, 1).abs()/10)
        self.ksi = nn.Parameter(torch.rand(chan, 1).abs()/10)
        self.ksr = nn.Parameter(torch.rand(chan, 1).abs()/10)
        if self.ksr_gain:
            self.ksr_2 = nn.Parameter(torch.rand(chan, 1).abs()/10)
        self.dt = dt
        
        if ka != None:
            self.ka.data = ka * torch.ones(chan, 1)
        if ka_2 != None and self.ka_offset:
            self.ka_2.data = ka_2 * torch.ones(chan, 1)
        if kfi != None:
            self.kfi.data = kfi * torch.ones(chan, 1)
        if kfr != None:
            self.kfr.data = kfr * torch.ones(chan, 1)
        if ksi != None:
            self.ksi.data = ksi * torch.ones(chan, 1)
        if ksr != None:
            self.ksr.data = ksr * torch.ones(chan, 1)
        if ksr_2 != None and self.ksr_gain:
            self.ksr_2.data = ksr_2 * torch.ones(chan, 1)

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
        dt = self.dt
        ka  = self.ka.abs() * rate * pop[:, 0]
        if self.ka_offset:
            ka += self.ka_2.abs() * pop[:, 0]
        kfi = self.kfi.abs() * pop[:,1]
        kfr = self.kfr.abs() * pop[:,2]
        ksi = self.ksi.abs() * pop[:,2]
        ksr = self.ksr.abs() * pop[:, 3]
        if self.ksr_gain:
            ksr += self.ksr_2.abs() * rate * pop[:, 3]
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
                                  