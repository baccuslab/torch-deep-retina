import numpy as np
import scipy
import copy
from collections import deque
import torch
import torch.nn as nn
import subprocess
import json
import os
import torchdeepretina.stimuli as tdrstim
from torchdeepretina.physiology import Physio
from tqdm import tqdm
import pyret.filtertools as ft
from kinetic.utils import get_hs

def rolling_window(array, window, time_axis=0):
    """
    Make an ndarray with a rolling window of the last dimension

    Parameters
    ----------
    array : array_like
        Array to add rolling window to

    window : int
        Size of rolling window

    time_axis : int, optional
        The axis of the temporal dimension, either 0 or -1 (Default: 0)

    Returns
    -------
    Array that is a view of the original array with a added dimension
    of size w.

    Examples
    --------
    >>> x=np.arange(10).reshape((2,5))
    >>> rolling_window(x, 3)
    array([[[0, 1, 2], [1, 2, 3], [2, 3, 4]],
           [[5, 6, 7], [6, 7, 8], [7, 8, 9]]])

    Calculate rolling mean of last dimension:

    >>> np.mean(rolling_window(x, 3), -1)
    array([[ 1.,  2.,  3.],
           [ 6.,  7.,  8.]])
    """
    if time_axis == 0:
        array = array.T

    elif time_axis == -1:
        pass

    else:
        raise ValueError('Time axis must be 0 (first dimension) or -1 (last)')

    assert window >= 1, "`window` must be at least 1."
    assert window < array.shape[-1], "`window` is too long."

    # with strides
    shape = array.shape[:-1] + (array.shape[-1] - window, window)
    strides = array.strides + (array.strides[-1],)
    arr = np.lib.stride_tricks.as_strided(array, shape=shape, strides=strides)

    if time_axis == 0:
        return np.rollaxis(arr.T, 1, 0)
    else:
        return arr


def get_hook(layer_dict, key, to_numpy=True, to_cpu=False):
    if to_numpy:
        def hook(module, inp, out):
            if torch.is_tensor(out):
                layer_dict[key] = out.detach().cpu().numpy()
            else:
                layer_dict[key] = out
    elif to_cpu:
        def hook(module, inp, out):
            if torch.is_tensor(out):
                layer_dict[key] = out.cpu()
            else:
                layer_dict[key] = out
    else:
        def hook(module, inp, out):
            layer_dict[key] = out
    return hook

def integrated_gradient(model, X, layer='sequential.2', gc_idx=None, alpha_steps=5,
                                                    batch_size=500, y=None, lossfxn=None,
                                                    to_numpy=False, verbose=False):
    """
    Returns the integrated gradient for a particular stimulus at the arged layer.
    Inputs:
        model: PyTorch Deep Retina models
        X: Input stimuli ndarray or torch FloatTensor (T,D,H,W)
        layer: str layer name
        gc_idx: ganglion cell of interest
            if None, uses all cells
        alpha_steps: int, integration steps
        batch_size: step size when performing computations on GPU
        y: torch FloatTensor or ndarray (T,N)
            if None, ignored
        lossfxn: some differentiable function
            if None, ignored
    Outputs:
        intg_grad: Integrated Gradients ndarray or FloatTensor (T, C, H1, W1)
        gc_activs: Activation of the final layer ndarray or FloatTensor (T,N)
    """
    # Handle Gradient Settings
    requires_grad(model, False) # Model gradient unnecessary for integrated gradient
    prev_grad_state = torch.is_grad_enabled() # Save current grad calculation state
    torch.set_grad_enabled(True) # Enable grad calculations

    layer_idx = 0
    for i,seq in enumerate(model.sequential):
        if layer == "sequential."+str(i):
            break
        if isinstance(seq, nn.ReLU):
            layer_idx += 1
    intg_grad = torch.zeros(len(X), model.chans[layer_idx], *model.shapes[layer_idx])
    gc_activs = None
    model.to(DEVICE)
    if gc_idx is None:
        gc_idx = list(range(model.n_units))
    if batch_size is None:
        batch_size = len(X)
    X = torch.FloatTensor(X)
    X.requires_grad = True
    idxs = torch.arange(len(X)).long()
    for batch in range(0, len(X), batch_size):
        prev_response = None
        linspace = torch.linspace(0,1,alpha_steps)
        if verbose:
            print("Calculating for batch {}/{}".format(batch, len(X)))
            linspace = tqdm(linspace)
        idx = idxs[batch:batch+batch_size]
        for alpha in linspace:
            x = alpha*X[idx]
            # Response is dict of activations. response[layer] has shape intg_grad.shape
            response = inspect(model, x, insp_keys=[layer], batch_size=None,
                                                    to_numpy=False, to_cpu=False,
                                                    verbose=False)
            if prev_response is not None:
                ins = response[layer]
                outs = response['outputs'][:,gc_idx]
                if lossfxn is not None and y is not None:
                    truth = y[idx,gc_idx]
                    outs = lossfxn(outs,truth)
                grad = torch.autograd.grad(outs.sum(), ins)[0]
                grad = grad.detach().cpu().reshape(len(grad), *intg_grad.shape[1:])
                act = (response[layer].data.cpu()-prev_response[layer]).reshape(grad.shape)
                intg_grad[idx] += grad*act
                if alpha == 1:
                    if gc_activs is None:
                        if isinstance(gc_idx, int):
                            gc_activs = torch.zeros(len(X))
                        else:
                            gc_activs = torch.zeros(len(X), len(gc_idx))
                    gc_activs[idx] = response['outputs'][:,gc_idx].detach().cpu()
            prev_response = {k:v.data.cpu() for k,v in response.items()}
    del response
    del grad
    if len(gc_activs.shape) == 1:
        gc_activs = gc_activs.unsqueeze(1) # Create new axis

    # Return to previous gradient calculation state
    requires_grad(model, True)
    torch.set_grad_enabled(prev_grad_state) # return to previous grad calculation state
    if to_numpy:
        return intg_grad.data.cpu().numpy(), gc_activs.data.cpu().numpy()
    return intg_grad, gc_activs

def stimulus_importance(model, X, gc_idx=None, alpha_steps=5, batch_size=500, 
                        to_numpy=False, verbose=False, device=torch.device('cuda:1')):
    # Handle Gradient Settings
    requires_grad(model, False) # Model gradient unnecessary for integrated gradient
    prev_grad_state = torch.is_grad_enabled() # Save current grad calculation state
    torch.set_grad_enabled(True) # Enable grad calculations

    intg_grad = torch.zeros(len(X), *model.image_shape)
    if gc_idx is None:
        gc_idx = list(range(model.n_units))
    if batch_size is None:
        batch_size = len(X)
    X = torch.FloatTensor(X)
    X.requires_grad = True
    idxs = torch.arange(len(X)).long()
    for batch in range(0, len(X), batch_size):
        linspace = torch.linspace(0,1,alpha_steps)
        if verbose:
            print("Calculating for batch {}/{}".format(batch, len(X)))
            linspace = tqdm(linspace)
        idx = idxs[batch:batch+batch_size]
        for alpha in linspace:
            x = alpha*X[idx]
            response = inspect(model, x, insp_keys=[], batch_size=None, to_numpy=False, device=device)
            outs = response['outputs'][:,gc_idx]
            grad = torch.autograd.grad(outs.sum(), x)[0]
            grad = grad.detach().cpu().reshape(len(grad), *intg_grad.shape[1:])
            act = X[idx].detach().cpu()
            intg_grad[idx] += grad*act
    del response
    del grad

    requires_grad(model, True)
    torch.set_grad_enabled(prev_grad_state) # return to previous grad calculation state
    if to_numpy:
        return intg_grad.data.cpu().numpy()
    return intg_grad

def inspect(model, X, insp_keys={}, batch_size=None, to_numpy=True, device=torch.device('cuda:1')):
    """
    Get the response from the argued layers in the model as np arrays. If model is on cpu,
    operations are performed on cpu. Put model on gpu if you desire operations to be
    performed on gpu.

    model - torch Module or torch gpu Module
    X - ndarray (T,C,H,W)
    insp_keys - set of str
        name of layers activations to collect
    to_numpy - bool
        if true, activations will all be ndarrays. Otherwise torch tensors
    to_cpu - bool
        if true, torch tensors will be on the cpu.
        only effective if to_numpy is false.

    returns dict of np arrays or torch cpu tensors
    """
    layer_outs = dict()
    handles = []
    if "all" in insp_keys:
        for i in range(len(model.sequential)):
            key = "sequential."+str(i)
            hook = get_hook(layer_outs, key, to_numpy=to_numpy, to_cpu=True)
            handle = model.sequential[i].register_forward_hook(hook)
            handles.append(handle)
    else:
        for key, mod in model.named_modules():
            if key in insp_keys:
                hook = get_hook(layer_outs, key, to_numpy=to_numpy, to_cpu=True)
                handle = mod.register_forward_hook(hook)
                handles.append(handle)
    X = torch.FloatTensor(X)
    if batch_size is None:
        if next(model.parameters()).is_cuda:
            X = X.to(device)
        preds = model(X)
        if to_numpy:
            layer_outs['outputs'] = preds.detach().cpu().numpy()
        else:
            layer_outs['outputs'] = preds.cpu()
    else:
        use_cuda = next(model.parameters()).is_cuda
        batched_outs = {key:[] for key in insp_keys}
        outputs = []
        for batch in range(0,len(X), batch_size):
            x = X[batch:batch+batch_size]
            if use_cuda:
                x = x.to(device)
            preds = model(x).cpu()
            if to_numpy:
                preds = preds.detach().numpy()
            outputs.append(preds)
            for k in layer_outs.keys():
                batched_outs[k].append(layer_outs[k])
        batched_outs['outputs'] = outputs
        if to_numpy:
            layer_outs = {k:np.concatenate(v,axis=0) for k,v in batched_outs.items()}
        else:
            layer_outs = {k:torch.cat(v,dim=0) for k,v in batched_outs.items()}
    for i in range(len(handles)):
        handles[i].remove()
    del handles
    return layer_outs

def inspect_rnn(model, X, hs, insp_keys=[]):

    layer_outs = dict()
    handles = []
    for key, mod in model.named_modules():
        if key in insp_keys:
            hook = get_hook(layer_outs, key, to_numpy=True)
            handle = mod.register_forward_hook(hook)
            handles.append(handle)

    resps = []
    layer_outs_list = {key:[] for key in insp_keys}
    with torch.no_grad():
        for i in range(X.shape[0]):
            resp, hs = model(X[i:i+1], hs)
            resps.append(resp)
            for k in layer_outs.keys():
                layer_outs_list[k].append(layer_outs[k])
    
    layer_outs = {k:np.concatenate(v, axis=0) if isinstance(v[0], np.ndarray) else v for k,v in layer_outs_list.items()}
    resp = torch.cat(resps, dim=0)
    layer_outs['outputs'] = resp.detach().cpu().numpy()
    
    for i in range(len(handles)):
        handles[i].remove()
    del handles
    
    return layer_outs

def batch_compute_model_response(stimulus, model, batch_size=500, recurrent=False, 
                                insp_keys={'all'}, cust_h_init=False, verbose=False):
    '''
    Computes a model response in batches in pytorch. Returns a dict of lists 
    where each list is the sequence of batch responses.     
    Args:
        stimulus: 3-d checkerboard stimulus in (time, space, space)
        model: the model
        batch_size: the size of the batch
        recurrent: bool
            use recurrent approach to calculating model response
        insp_keys: set (or dict) with keys of layers to be inspected in Physio
    '''
    phys = Physio(model)
    model_response = None
    batch_size = 1 if recurrent else batch_size
    n_loops, leftover = divmod(stimulus.shape[0], batch_size)
    hs = None
    if recurrent:
        hs = [torch.zeros(1, *h_shape).to(DEVICE) for h_shape in model.h_shapes]
        if cust_h_init:
            hs[0][:,0] = 1
    with torch.no_grad():
        responses = []
        rng = range(n_loops)
        if verbose:
            rng = tqdm(rng)
        for i in rng:
            stim = torch.FloatTensor(stimulus[i*batch_size:(i+1)*batch_size])
            outs = phys.inspect(stim.to(DEVICE), hs=hs, insp_keys=insp_keys)
            if type(outs) == type(tuple()):
                outs, hs = outs[0].copy(), outs[1]
            else:
                outs = outs.copy()
            for k in outs.keys():
                if type(outs[k]) != type(np.array([])):
                    outs[k] = outs[k].detach().cpu().numpy()
            responses.append(outs)
        if leftover > 0 and hs is None:
            stim = torch.FloatTensor(stimulus[-leftover:])
            outs = phys.inspect(stim.to(DEVICE), hs=hs, insp_keys=insp_keys).copy()
            for k in outs.keys():
                if type(outs[k]) != type(np.array([])):
                    outs[k] = outs[k].detach().cpu().numpy()
            responses.append(outs)
        model_response = {}
        for key in responses[0].keys():
             model_response[key] = np.concatenate([x[key] for x in responses], axis=0)
        del outs

    # Get the last few samples
    phys.remove_hooks()
    phys.remove_refs()
    del phys
    return model_response

def get_stim_grad(model, X, layer, cell_idx, batch_size=500, layer_shape=None, verbose=True, I20=None):
    """
    Gets the gradient of the model output at the specified layer and cell idx with respect
    to the inputs (X). Returns a gradient array with the same shape as X.
    """
    if verbose:
        print("layer:", layer)
    requires_grad(model, False)
    device = next(model.parameters()).get_device()

    if model.kinetic:
        hs = get_hs(model, batch_size, device, I20)
    elif model.recurrent:
        batch_size = 1
        hs = [torch.zeros(batch_size, *h_shape).to(device) for h_shape in model.h_shapes]

    hook_outs = dict()
    module = None
    for name, modu in model.named_modules():
        if name == layer:
            if verbose:
                print("hook attached to " + name)
            module = modu
            hook = get_hook(hook_outs,key=layer,to_numpy=False)
            hook_handle = module.register_forward_hook(hook)

    # Get gradient with respect to activations
    model.eval()
    X.requires_grad = True
    n_loops = X.shape[0]//batch_size
    rng = range(n_loops)
    if verbose:
        rng = tqdm(rng)
    for i in rng:
        idx = i*batch_size
        x = X[idx:idx+batch_size]
        x = x.to(device)
        if model.kinetic:
            _, hs = model(x, hs)
            hs[0] = hs[0].detach()
            hs[1] = deque([h.detach() for h in hs[1]], maxlen=model.seq_len)
        elif model.recurrent:
            _, hs = model(x, hs)
            hs = [h.data for h in hs]
        else:
            _ = model(x)
        # Outs are the activations at the argued layer and cell idx accross the batch
        if type(cell_idx) == type(int()):
            fx = hook_outs[layer][:,cell_idx]
        elif len(cell_idx) == 1:
            fx = hook_outs[layer][:,cell_idx[0]]
        else:
            fx = hook_outs[layer][:, cell_idx[0], cell_idx[1], cell_idx[2]]
        fx = fx.mean()
        fx.backward()
    hook_handle.remove()
    requires_grad(model, True)
    return X.grad.data.cpu().numpy()

def compute_sta(model, contrast, layer, cell_index, layer_shape=None, verbose=True, I20=None):
    """
    Computes the STA using the average of instantaneous receptive 
    fields (gradient of output with respect to input)
    """
    # generate some white noise
    #X = stim.concat(white(1040, contrast=contrast)).copy()
    X = tdrstim.concat(contrast*np.random.randn(10000,50,50),nh=model.img_shape[0])
    X = torch.FloatTensor(X)
    X.requires_grad = True

    # compute the gradient of the model with respect to the stimulus
    drdx = get_stim_grad(model, X, layer, cell_index, layer_shape=layer_shape, verbose=verbose, I20=I20)
    sta = drdx.mean(0)

    del X
    return sta

def revcor_sta(model, layers=['sequential.0','sequential.6'], chans=[8,8], verbose=True, device=torch.device('cuda:1')):
    """
    Computes the sta using reverse correlation. Uses the central unit for computation

    model - torch Module

    returns:
        dict of sta lists for each channel in each layer
        keys: layer names
            vals: lists of stas for each channel in the layer
    """
    noise = np.random.randn(10000,50,50)
    try:
        filter_size = model.img_shape[0]
    except:
        filter_size = model.image_shape[0]
    X = tdrstim.concat(noise, nh=filter_size)
    noise = noise[filter_size:]
    response = inspect(model, X, insp_keys=set(layers), batch_size=500, to_numpy=True, device=device)
    stas = {layer:[] for layer in layers}
    for layer,chan in zip(layers,chans):
        resp = response[layer]
        if len(resp.shape) == 2:
            if layer == "sequential.2":
                resp = resp.reshape(len(resp), len(chan), *model.shapes[0])
            else:
                resp = resp.reshape(len(resp), len(chan), *model.shapes[1])
        centers = np.array(resp.shape[2:])//2
        for c in range(chan):
            if len(centers) == 2:
                sta,_ = ft.revcorr(noise, scipy.stats.zscore(resp[:, c, centers[0], centers[1]]),
                                                             0, filter_size)
            if len(centers) == 3:
                sta,_ = ft.revcorr(noise, scipy.stats.zscore(resp[:, c, centers[0], centers[1], centers[2]]),
                                                             0, filter_size)
            stas[layer].append(sta)
    return stas

def revcor_sta_rnn(model, layers, device, I20=None):
    
    chans = model.chans
    noise = np.random.randn(10000,50,50)
    try:
        filter_size = model.img_shape[0]
    except:
        filter_size = model.image_shape[0]
    X = tdrstim.concat(noise, nh=filter_size)
    X = torch.FloatTensor(X).to(device)
    noise = noise[filter_size:]
    hs = get_hs(model, 1, device, I20)
    
    response = inspect_rnn(model, X, hs, insp_keys=list(layers))
    stas = {layer:[] for layer in layers}
    for layer,chan in zip(layers,chans):
        resp = response[layer]
        if len(resp.shape) == 2:
            if layer == "bipolar.0":
                resp = resp.reshape(len(resp), chan, *model.shapes[0])
            else:
                resp = resp.reshape(len(resp), chan, *model.shapes[1])
        centers = np.array(resp.shape[2:])//2
        for c in range(chan):
            if len(centers) == 2:
                sta,_ = ft.revcorr(noise, scipy.stats.zscore(resp[:, c, centers[0], centers[1]]),
                                                             0, filter_size)
            if len(centers) == 3:
                sta,_ = ft.revcorr(noise, scipy.stats.zscore(resp[:, c, centers[0], centers[1], centers[2]]),
                                                             0, filter_size)
            stas[layer].append(sta)
    return stas

def revcor_sta_ganglion(model, layers=['ganglion.0'], n_units=5,  verbose=True, device=torch.device('cuda:1')):
    """
    Computes the sta using reverse correlation. Uses the central unit for computation

    model - torch Module

    returns:
        dict of sta lists for each channel in each layer
        keys: layer names
            vals: lists of stas for each channel in the layer
    """
    noise = np.random.randn(10000,50,50)
    try:
        filter_size = model.img_shape[0]
    except:
        filter_size = model.image_shape[0]
    X = tdrstim.concat(noise, nh=filter_size)
    noise = noise[filter_size:]
    response = inspect(model, X, insp_keys=set(layers), batch_size=500, to_numpy=True, device=device)
    stas = {layer:[] for layer in layers}
    for layer in layers:
        for cell in range(n_units):
            resp = response[layer]
            sta,_ = ft.revcorr(noise, scipy.stats.zscore(resp[:, cell]), 0, filter_size)
            stas[layer].append(sta)
    return stas