import torch
import torch.nn as nn

class GaussianNoise(nn.Module):
    def __init__(self, std=0.05):
        super(GaussianNoise, self).__init__()
        self.cuda_param = nn.Parameter(torch.zeros(1), requires_grad=False)
        self.std = std
    
    def forward(self, x):
        if not self.training:
            return x
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
