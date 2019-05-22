import torch
import torch.nn as nn
from .torch_utils import GaussianNoise
import numpy as np

def deconv_block(in_depth, out_depth, ksize=3, stride=1, padding=1, bnorm=False, activation='relu', noise=0.05):
    block = []
    block.append(nn.ConvTranspose2d(in_depth, out_depth, ksize, stride=stride, padding=padding))
    block.append(GaussianNoise(std=noise))
    if activation is None:
        pass
    elif activation.lower() == 'relu':
        block.append(nn.ReLU())
    elif activation.lower() == 'tanh':
        block.append(nn.Tanh())
    if bnorm:
        block.append(nn.BatchNorm2d(out_depth))
    return nn.Sequential(*block)
    
def conv_block(in_depth,out_depth,ksize=3,stride=1,padding=1,activation='leaky',bnorm=False, noise=0.05):
    block = []
    block.append(nn.Conv2d(in_depth, out_depth, ksize, stride=stride, padding=padding))
    block.append(GaussianNoise(std=noise))
    if activation is None:
        pass
    elif activation.lower() == 'relu':
        block.append(nn.ReLU())
    elif activation.lower() == 'tanh':
        block.append(nn.Tanh())
    elif activation.lower() == 'leaky':
        block.append(nn.LeakyReLU(negative_slope=0.05))
    if bnorm:
        block.append(nn.BatchNorm2d(out_depth))
    return nn.Sequential(*block)

class StimGenerator(nn.Module):
    def __init__(self, z_size=100, trainable_z=False, bnorm=False, out_depths=None, noise=0.1):
        """
        Assumes the desired image should be (40,50,50)
        """
        super(StimGenerator, self).__init__()
        self.img_shape = [40,50,50]
        self.z_size = z_size
        self.trainable_z = trainable_z
        self.bnorm = bnorm
        self.noise = noise

        # Trainable Mean and STD
        if trainable_z:
            self.mu = nn.Parameter(torch.zeros(1,z_size))
            self.std = nn.Parameter(torch.ones(1,z_size))
            self.z_stats = [self.mu, self.std]
        else:
            self.mu = None
            self.std = None
            self.z_stats = None

        
        ## General img size eqn
        ## img_new = (img_old - 1)*stride + ksize - 2*padding

        if out_depths is None:
            out_depths = [100, 200, 400, 256, 256, 128, 64, 64, 64]
        self.out_depths = out_depths

        # Convolution Transpose Portion
        self.deconvs = []

        # Conv transposes
        ksize=4; padding=0; stride=2; in_depth = z_size # Starts width and height of output at 4x4
        out_depth = out_depths[0]
        self.deconvs.append(deconv_block(in_depth,out_depth,
                                            ksize=ksize,padding=padding,
                                            stride=stride,bnorm=self.bnorm,
                                            noise=noise))
        size = ksize
        n_deconvs = 4
        # Each loop doubles hieght and width
        for i in range(1,n_deconvs):
            ksize=4; padding=1; stride=2 # Doubles the width and height of outputs
            self.deconvs.append(deconv_block(out_depths[i-1], out_depths[i], 
                                                ksize=ksize, padding=padding, 
                                                stride=stride, bnorm=self.bnorm,
                                                noise=noise))
            size *= 2
        print("After doubling loop:", size) # size should be 32
        # This leaves 18 for goal of 50

        for i in range(n_deconvs, n_deconvs+5): 
            ksize=5; padding=0; stride=1
            in_depth = out_depths[i-1]
            out_depth = out_depths[i]
            self.deconvs.append(deconv_block(in_depth, out_depth, ksize=ksize, 
                                                    padding=padding, stride=stride, 
                                                    bnorm=self.bnorm, noise=noise))
            size += 4

        print("After add 4 loop:", size) # size should be 52

        # Reduce from 52x52 to 50x50
        ksize=3; padding=0; stride=1
        in_depth = out_depths[-1]
        out_depth = self.img_shape[-3]
        self.deconvs.append(conv_block(in_depth, out_depth, ksize=ksize, 
                                                padding=padding, stride=stride, 
                                                bnorm=self.bnorm, noise=noise))

        # Final layer
        ksize=3; padding=1; stride=1; in_depth = out_depth
        out_depth = self.img_shape[-3]
        self.deconvs.append(conv_block(in_depth, out_depth, ksize=ksize, 
                                                padding=padding, stride=stride, 
                                                activation=None, bnorm=True, noise=0))

        self.generator = nn.Sequential(*self.deconvs)

    def generate_img(self, n_samples=2):
        zs = self.get_zs(n_samples)
        if self.trainable_z:
            zs = self.std * zs + self.mu
        samples = zs.view(-1, self.z_size, 1, 1)
        return self.generator(samples)

    def get_zs(self, n_samples):
        z = torch.randn(n_samples, self.z_size)
        cuda_p = next(self.parameters())
        if cuda_p is not None and cuda_p.is_cuda:
            return z.to(cuda_p.get_device())
        return z

    def forward(self, n_samples):
        return self.generate_img(n_samples)

    def extra_repr(self):
        return "z_size={}, trainable_z={}, bnorm={}, out_depths={}, noise={}".format(self.z_size, self.trainable_z, self.bnorm, self.out_depths, self.noise)

