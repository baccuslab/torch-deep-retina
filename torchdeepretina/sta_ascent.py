import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import json
import pickle
import numpy as np
from .miscellaneous import freeze_weights

def load_json(json_file):
    with open(json_file) as f:
        s = f.read()
        j = json.loads(s)
    return j

def cuda_if(x):
    if torch.cuda.is_available:
        return x.cuda()
    return x

class STAAscent:
    def __init__(self):
        pass

    def remove_hook(self):
        self.hook_handle.remove()

    def attach_hook(self, module):
        def fwd_hook(module, inp, output):
            self.activs = output[0]
        self.hook_handle = module.register_forward_hook(fwd_hook)
    
    def sta_ascent(self, model, layer, units, lr=.01, n_epochs=5000, sta_shape=(1,40,50,50), constraint=1):
        """
        Performs gradient ascent on a randomly initialized image with dimensions
        the same shape as was trained on.
    
        hyps - dict of hyperparameters
            keys:
                "lr"
                "n_epochs"
        model - pytorch module
        layer - string
            name of module of interest
        units - sequence of coordinate tuples (c,h,w)
            if none, all units will be examined, otherwise only units with the
            specified idxs will be evaluated
        constraint - float
            the coefficient of the norm of the sta image added to the loss
        """
        model.eval()
        freeze_weights(model)
        cuda_if(model)
        vis_module = None
        for name, module in model.named_modules():
            if name == layer:
                vis_module = module
                break
        self.attach_hook(vis_module)
        sta_image = cuda_if(torch.randn(sta_shape)*2/(np.prod(sta_shape)))
        x = model(sta_image)
        x.detach()
        activs_shape = self.activs.view(-1).shape
        # ravel the indices
        try:
            if len(units[0]) > 1:
                c,h,w = units[0]
                shape = self.activs.shape
                units = [int(c*shape[1]*shape[2]+h*shape[2]+w) for c,h,w in units]
        except:
            pass
        sta_image = cuda_if(torch.randn(sta_shape)*2/(np.prod(sta_shape)))
        sta_image.requires_grad = True
        optim = torch.optim.Adam([sta_image], lr=lr)
        for i in range(n_epochs):
            optim.zero_grad()
            model(sta_image)
            loss = -self.activs.view(-1)[units].mean()
            if constraint > 0:
                loss += sta_image.norm(2)*constraint
            loss.backward()
            optim.step()
            print("Loss:", "{:.5f}".format(loss.item()), "--", i/n_epochs, "% done", end='\r')
        sta_image = sta_image.detach().cpu().numpy().astype(np.float)
        self.remove_hook()
        return sta_image

