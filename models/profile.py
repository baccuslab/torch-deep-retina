import torch
import torch.nn as nn
from torch.distributions import normal
from torch_utils import GaussianNoise
import time

def exec_time(args, fxn, iters=10000):
    startt = time.time()
    for i in range(iters):
        fxn(*args)
    return time.time() - startt

def gaussian(x, sigma=.05):
    noise = normal.Normal(torch.zeros(x.size()), sigma*torch.ones(x.size()))
    if x.is_cuda:
        return x + noise.sample().cuda()
    return x + noise.sample()

x = torch.ones(100,100).cuda()
relu = nn.ReLU().cuda()

print("Functional:", exec_time([x], nn.functional.relu))
print("Module:", exec_time([x], relu))
