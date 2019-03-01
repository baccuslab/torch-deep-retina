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

class ScaleShift(nn.Module):
    def __init__(self, x_shape, scale=True, shift=True):
        super(ScaleShift, self).__init__()
        self.scale = nn.Parameter(torch.ones(x_shape).float(), requires_grad=scale)
        self.shift = nn.Parameter(torch.zeros(x_shape).float(), requires_grad=shift)

    def forward(self, x):
        return x*self.scale + self.shift
