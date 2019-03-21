import torch
import torch.nn as nn

class GaussianNoise(nn.Module):
    def __init__(self, std=0.1, set_at_runtime=False):
        super(GaussianNoise, self).__init__()
        self.cuda_param = nn.Parameter(torch.zeros(1), requires_grad=False)
        self.std = std
        self.set_at_runtime = set_at_runtime
    
    def forward(self, x):
        if not self.training:
            return x
        if self.set_at_runtime is True:
            self.set_at_runtime = False
            self.std = x.std()*self.std
        noise = self.std * torch.randn(x.size())
        if next(self.parameters()).is_cuda:
            noise = noise.cuda()
        return x + noise

    def extra_repr(self):
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

class AbsConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True):
        super(AbsConv2d, self).__init__()
        self.bias = bias
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)

    def forward(self, x):
        if self.bias:
            return nn.functional.conv2d(x, self.conv.weight.abs(), self.conv.bias.abs(), 
                                                    self.conv.stride, self.conv.padding, 
                                                    self.conv.dilation, self.conv.groups)
        else:
            return nn.functional.conv2d(x, self.conv.weight.abs(), None, 
                                                self.conv.stride, self.conv.padding, 
                                                self.conv.dilation, self.conv.groups)

class AbsLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(AbsLinear, self).__init__()
        self.bias = bias
        self.linear = nn.Linear(in_features, out_features, bias=bias)

    def forward(self, x):
        if self.bias:
            return nn.functional.linear(x, self.linear.weight.abs(), 
                                            self.linear.bias.abs())
        else:
            return nn.functional.linear(x, self.linear.weight.abs(), 
                                                    self.linear.bias)

class StackedConv2d(nn.Module):
    '''
    Builds the main kernel out of multiple 3x3 kernels
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

class MeanOnlyBatchNorm(nn.Module):
    def __init__(self, shape, momentum=.1):
        super(MeanOnlyBatchNorm, self).__init__()
        self.running_mean = 0
        self.momentum = momentum
        self.scale = nn.Parameter(torch.ones(shape).float())
        self.shift = nn.Parameter(torch.zeros(shape).float())

    def forward(self, x):
        mean = x.mean(0)
        x = x - mean
        self.running_mean = (1-self.momentum)*self.running_mean + self.momentum*mean
        return x*self.scale + self.shift
