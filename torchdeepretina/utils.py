import numpy as np
import copy
import torch
import torch.nn as nn
import json
import os
import torchdeepretina.stimuli as tdrstim
from torchdeepretina.models import LinearStackedConv2d
from tqdm import tqdm
import pyret.filtertools as ft

DEVICE = torch.device("cuda:0")

def try_key(dict_, key, default):
    """
    If the key is in the dict, then the corresponding value is
    returned. If the key is not in the dict, then the default value
    is returned.

    dict_: dict
    key: any hashable object
    default: anything
    """
    if key in dict_:
        return dict_[key]
    return default

def load_json(file_name):
    """
    Loads a json file as a python dict

    file_name: str
        the path of the json file
    """
    file_name = os.path.expanduser(file_name)
    with open(file_name) as f:
        s = f.read()
        j = json.loads(s)
    return j

def get_conv_layer_names(model, conv_types=None):
    """
    Finds the layer names of convolutions in the model. Does not
    return names of sublayers.

    inputs:
        model: torch nn Module object
        conv_types: set of classes
            the classes that constitute a convolutional layer type

    returns:
        conv_names: list of str
            list of all the conv names in the model
    """
    if conv_types is None:
        conv_types = set()
        conv_types.add(nn.Conv2d)
        conv_types.add(LinearStackedConv2d)
    conv_names = []
    for i,(name,modu) in enumerate(model.named_modules()):
        if len(name.split(".")) == 2 and type(modu) in conv_types:
            conv_names.append(name)
    return conv_names

def get_layer_name_sets(model, delimeters=[nn.ReLU,nn.Softplus,
                                                     nn.Tanh]):
    """
    Creates a set of the module names for each layer. Delineates
    layers based on the argued layer types. 

    model: torch nn Module object
    delimeters: list of classes to delineate the start of a new layer

    returns:
        layer_names: list of sets of str
            list of sets of all the layer names in each layer
    """
    layer_names = []
    layer_set = set()
    for i,(name,modu) in enumerate(model.named_modules()):
        layer_set.add(name)
        if i > 0 and type(modu) in delimeters:
            layer_names.append(layer_set)
            layer_set = set()
    if len(layer_set) > 0:
        layer_names.append(layer_set)
    return layer_names

def get_layer_idx(model, layer, delimeters=[nn.ReLU, nn.Tanh,
                                               nn.Softplus]):
    """
    Finds the layer index of the layer name. Returns -1 if the layer
    does not exist in the argued model.

    model: torch nn Module object
    layer: str
        name of the layer in question
    delimeters: list of classes 
        used to delineate the start of a new layer
    """
    layer_names = get_layer_name_sets(model, delimeters=delimeters)
    for i,lnames in enumerate(layer_names):
        if layer in lnames:
            return i
    return -1

def get_hook(layer_dict, key, to_numpy=True, to_cpu=False):
    """
    Returns a hook function that can be used to collect gradients
    or activations in the backward or forward pass respectively of
    a torch Module.

    layer_dict: dict
        Can be empty

        keys: str
            names of model layers of interest
        vals: NA
    key: str
        name of layer of interest
    to_numpy: bool
        if true, the gradients/activations are returned as ndarrays.
        otherwise they are returned as torch tensors
    """
    if to_numpy:
        def hook(module, inp, out):
            layer_dict[key] = out.detach().cpu().numpy()
    elif to_cpu:
        def hook(module, inp, out):
            layer_dict[key] = out.cpu()
    else:
        def hook(module, inp, out):
            layer_dict[key] = out
    return hook

def linear_response(filt, stim, batch_size=1000, to_numpy=True):
    """
    Runs a filter as a convolution over the stimulus.

    filt: torch tensor or ndarray (C,) or (C,H,W)
        the filter to be run over the stimulus
    stim: torch tensor or ndarray (T,C) or (T,C,H,W)
        the stimulus to be filtered
    batch_size: int
        the size of the batching during filtering. Used to reduce
        memory consumption.
    to_numpy: bool
        if true, the response is returned as a ndarray
        if false, the resp is returned as a torch tensor
    """
    if type(filt) == type(np.array([])):
        filt = torch.FloatTensor(filt)
    if type(stim) == type(np.array([])):
        stim = torch.FloatTensor(stim)
    filt = filt.to(DEVICE)
    filt = filt.reshape(-1)
    stim = stim.reshape(len(stim), -1)
    # Filter must match spatiotemporal dims of stimulus
    assert filt.shape[0] == stim.shape[1]
    if batch_size is None:
        stim = stim.to(DEVICE)
        resp = torch.einsum("ij,j->i", stim, filt).cpu()
    else:
        resps = []
        for i in range(0, len(stim), batch_size):
            temp = stim[i:i+batch_size].to(DEVICE)
            temp = torch.einsum("ij,j->i", temp, filt)
            resps.append(temp.cpu())
        resp = torch.cat(resps, dim=0)
    if to_numpy:
        resp = resp.detach().numpy()
    return resp

def inspect(model, X, insp_keys={}, batch_size=500, to_numpy=True,
                                                      to_cpu=True,
                                                      verbose=False):
    """
    Get the response from the argued layers in the model as np arrays.
    If model is on cpu, operations are performed on cpu. Put model on
    gpu if you desire operations to be performed on gpu.

    model - torch Module or torch gpu Module
    X - ndarray or FloatTensor (T,C,H,W)
    insp_keys - set of str
        name of layers activations to collect
    to_numpy - bool
        if true, activations will all be ndarrays. Otherwise torch
        tensors
    to_cpu - bool
        if true, torch tensors will be on the cpu.
        only effective if to_numpy is false.

    returns: 
        layer_outs: dict of np arrays or torch cpu tensors
            "outputs": default key for output layer
    """
    layer_outs = dict()
    handles = []
    if "all" in insp_keys:
        for i in range(len(model.sequential)):
            key = "sequential."+str(i)
            hook = get_hook(layer_outs, key, to_numpy=to_numpy,
                                                 to_cpu=to_cpu)
            handle = model.sequential[i].register_forward_hook(hook)
            handles.append(handle)
    else:
        for key, mod in model.named_modules():
            if key in insp_keys:
                hook = get_hook(layer_outs, key, to_numpy=to_numpy,
                                                     to_cpu=to_cpu)
                handle = mod.register_forward_hook(hook)
                handles.append(handle)
    X = torch.FloatTensor(X)

    # prev_grad_state is used to ensure we do not mess with an outer
    # "with torch.no_grad():" statement
    prev_grad_state = torch.is_grad_enabled() 
    if to_numpy:
        # Turns off all gradient calculations. When returning numpy
        # arrays, the computation graph is inaccessible, as such we
        # do not need to calculate it.
        torch.set_grad_enabled(False)

    if batch_size is None or batch_size > len(X):
        if next(model.parameters()).is_cuda:
            X = X.to(DEVICE)
        preds = model(X)
        if to_numpy:
            layer_outs['outputs'] = preds.detach().cpu().numpy()
        else:
            layer_outs['outputs'] = preds.cpu()
    else:
        use_cuda = next(model.parameters()).is_cuda
        batched_outs = {key:[] for key in insp_keys}
        outputs = []
        rnge = range(0,len(X), batch_size)
        if verbose:
            rnge = tqdm(rnge)
        for batch in rnge:
            x = X[batch:batch+batch_size]
            if use_cuda:
                x = x.to(DEVICE)
            preds = model(x).cpu()
            if to_numpy:
                preds = preds.detach().numpy()
            outputs.append(preds)
            for k in layer_outs.keys():
                batched_outs[k].append(layer_outs[k])
        batched_outs['outputs'] = outputs
        if to_numpy:
            layer_outs = {k:np.concatenate(v,axis=0) for k,v in\
                                           batched_outs.items()}
        else:
            layer_outs = {k:torch.cat(v,dim=0) for k,v in\
                                     batched_outs.items()}
    
    # If we turned off the grad state, this will turn it back on.
    # Otherwise leaves it the same.
    torch.set_grad_enabled(prev_grad_state) 
    
    # This for loop ensures we do not create a memory leak when
    # using hooks
    for i in range(len(handles)):
        handles[i].remove()
    del handles

    return layer_outs


def get_stim_grad(model, X, layer, cell_idx, batch_size=500,
                                           layer_shape=None,
                                           to_numpy=True,
                                           verbose=True):
    """
    Gets the gradient of the model output at the specified layer and
    cell idx with respect to the inputs (X). Returns a gradient array
    with the same shape as X.

    model: nn.Module
    X: torch FloatTensor
    layer: str
    cell_idx: int or tuple (chan, row, col)
        idx of cell of interest
    batch_size: int
        size of batching for calculations
    layer_shape: tuple
        changes the shape of the argued layer to this shape if tuple
    to_numpy: bool
        returns the gradient vector as a numpy array if true
    """
    if verbose:
        print("layer:", layer)
    requires_grad(model, False)
    cud = next(model.parameters()).is_cuda
    device = torch.device('cuda:0') if cud else torch.device('cpu')
    prev_grad_state = torch.is_grad_enabled() 
    torch.set_grad_enabled(True)

    if model.recurrent:
        batch_size = 1
        hs = [torch.zeros(batch_size, *h_shape).to(device) for\
                                     h_shape in model.h_shapes]

    if layer == 'output' or layer=='outputs':
        layer = "sequential."+str(len(model.sequential)-1)
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
    if type(X) == type(np.array([])):
        X = torch.FloatTensor(X)
    X.requires_grad = True
    n_loops = X.shape[0]//batch_size
    rng = range(n_loops)
    if verbose:
        rng = tqdm(rng)
    for i in rng:
        idx = i*batch_size
        x = X[idx:idx+batch_size].to(device)
        if model.recurrent:
            _, hs = model(x, hs)
            hs = [h.data for h in hs]
        else:
            _ = model(x)
        if layer_shape is not None:
            hook_outs[layer] = hook_outs[layer].reshape(-1,
                                              *layer_shape)
        # Outs are the activations at the argued layer and cell idx
        # for the batch
        if type(cell_idx) == type(int()):
            fx = hook_outs[layer][:,cell_idx]
        elif len(cell_idx) == 1:
            fx = hook_outs[layer][:,cell_idx[0]]
        else:
            fx = hook_outs[layer][:, cell_idx[0], cell_idx[1],
                                                  cell_idx[2]]
        fx = fx.sum()
        fx.backward()
    hook_handle.remove()
    requires_grad(model, True)
    torch.set_grad_enabled(prev_grad_state) 
    if to_numpy:
        return X.grad.data.cpu().numpy()
    else:
        return X.grad.data.cpu()

def integrated_gradient(model, X, layer='sequential.2', gc_idx=None,
                                                    alpha_steps=5,
                                                    batch_size=500,
                                                    y=None,
                                                    lossfxn=None,
                                                    to_numpy=False,
                                                    verbose=False):
    """
    Returns the integrated gradient for a particular stimulus at the
    arged layer.

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
        intg_grad: ndarray or FloatTensor (T, C, H1, W1)
            integrated gradient
        gc_activs: ndarray or FloatTensor (T,N)
            activation of the final layer of the model
    """
    # Handle Gradient Settings
    # Model gradient unnecessary for integrated gradient
    requires_grad(model, False)
    # Save current grad calculation state
    prev_grad_state = torch.is_grad_enabled()
    torch.set_grad_enabled(True) # Enable grad calculations

    layer_idx = get_layer_idx(model, layer=layer)
    intg_grad = torch.zeros(len(X), model.chans[layer_idx],
                                  *model.shapes[layer_idx])
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
            # Response is dict of activations. response[layer] has
            # shape intg_grad.shape
            response = inspect(model, x, insp_keys=[layer],
                                           batch_size=None,
                                           to_numpy=False,
                                           to_cpu=False,
                                           verbose=False)
            if prev_response is not None:
                ins = response[layer]
                outs = response['outputs'][:,gc_idx]
                if lossfxn is not None and y is not None:
                    truth = y[idx,gc_idx]
                    outs = lossfxn(outs,truth)
                grad = torch.autograd.grad(outs.sum(), ins)[0]
                grad = grad.detach().cpu().reshape(len(grad),
                                        *intg_grad.shape[1:])
                l = layer
                act = (response[l].data.cpu()-prev_response[l])
                act = act.reshape(grad.shape)
                intg_grad[idx] += grad*act
                if alpha == 1:
                    if gc_activs is None:
                        if isinstance(gc_idx, int):
                            gc_activs = torch.zeros(len(X))
                        else:
                            gc_activs =torch.zeros(len(X),len(gc_idx))
                    outs = response['outputs'][:,gc_idx]
                    gc_activs[idx] = outs.detach().cpu()
            prev_response={k:v.data.cpu() for k,v in response.items()}
    del response
    del grad
    if len(gc_activs.shape) == 1:
        gc_activs = gc_activs.unsqueeze(1) # Create new axis

    # Return to previous gradient calculation state
    requires_grad(model, True)
    # return to previous grad calculation state
    torch.set_grad_enabled(prev_grad_state)
    if to_numpy:
        ndgrad = intg_grad.data.cpu().numpy()
        ndactivs = gc_activs.data.cpu().numpy()
        return ndgrad, ndactivs
    return intg_grad, gc_activs

def compute_sta(model, layer, cell_index, layer_shape=None,
                                            batch_size=500,
                                            contrast=1,
                                            n_samples=10000,
                                            to_numpy=True,
                                            verbose=True,
                                            X=None):
    """
    Computes the STA using the average of instantaneous receptive 
    fields (gradient of output with respect to input)

    model: torch Module
    contrast: float
        contrast of whitenoise to calculate the sta
    layer: str
    cell_index: int or tuple (chan, row, col)
        idx of cell of interest
    batch_size: int
        size of batching for calculations
    contrast: int
        the std of the noise used for the stimulus
    n_samples: int
        length of the stimulus
    """
    # generate some white noise
    if X is None:
        X = tdrstim.concat(contrast*np.random.randn(n_samples,
                                        *model.img_shape[1:]))
    X = torch.FloatTensor(X)
    X.requires_grad = True

    # compute the gradient of the model with respect to the stimulus
    drdx = get_stim_grad(model, X, layer, cell_index,
                             layer_shape=layer_shape,
                             batch_size=batch_size,
                             to_numpy=to_numpy,
                             verbose=verbose)
    sta = drdx.mean(0)
    return sta


def get_mean(x, axis=None, batch_size=1000):
    """
    Returns mean of x along argued axis. Used for reducing memory
    footprint on large datasets.

    x: ndarray or torch tensor
    axis: int
    batch_size: int
        size of increment when calculating mean
    """
    cumu_sum = 0
    if axis is None:
        for i in range(0,len(x), batch_size):
            cumu_sum = cumu_sum + x[i:i+batch_size].sum()
        return cumu_sum/x.numel()
    else:
        for i in range(0,len(x), batch_size):
            cumu_sum = cumu_sum + x[i:i+batch_size].sum(axis)
        return cumu_sum/x.shape[axis]

def get_std(x, axis=None, batch_size=1000, mean=None):
    """
    Returns std of x along argued axis. Used for reducing memory
    footprint on large datasets. Does not use n-1 correction.

    x: ndarray or torch tensor
    axis: int
    batch_size: int
        size of increment when calculating mean
    mean: int or ndarray or torch tensor
        The mean to be used in calculating the std. If None, mean is
        automatically calculated. If ndarray or torch tensor, must
        match datatype of x.
    """
    if type(x) == type(np.array([])):
        sqrt = np.sqrt
    else:
        sqrt = torch.sqrt
    if mean is None:
        mean = get_mean(x,axis,batch_size)
    cumu_sum = 0
    if axis is None:
        for i in range(0,len(x), batch_size):
            cumu_sum = cumu_sum + ((x[i:i+batch_size]-mean)**2).sum()
        return sqrt(cumu_sum/x.numel())
    else:
        for i in range(0,len(x), batch_size):
            cumu_sum=cumu_sum+((x[i:i+batch_size]-mean)**2).sum(axis)
        return sqrt(cumu_sum/x.shape[axis])

def pearsonr(x,y):
    """
    Calculates the pearson correlation coefficient. This gives same
    results as scipy's version but allows you to calculate the
    coefficient over much larger data sizes. Additionally allows
    calculation for torch tensors.

    Inputs:
        x: ndarray or torch tensor (T, ...)
            the dimension that will be averaged must be the first.
            dimensionality and type must match that of y
        y: ndarray or torch tensor (T, ...)
            the dimension that will be averaged must be the first.
            dimensionality and type must match that of x

    Returns:
        pearsonr: ndarray or torch tensor (...)
            shape will be the 

    """
    shape = None if len(x.shape) == 1 else x.shape[1:]
    assert type(x) == type(y)
    if type(x) == type(np.array([])):
        sqrt = np.sqrt
    else:
        sqrt = torch.sqrt
    x = x.reshape(len(x), -1)
    y = y.reshape(len(y), -1)
    try:
        mux = x.mean(0)
        muy = y.mean(0)
        # STD calculation ensures same calculation is performed for
        # ndarrays and torch tensors. Torch tensor .std() uses n-1 
        # correction
        sigx = sqrt((x**2).mean(0)-mux**2)
        sigy = sqrt((y**2).mean(0)-muy**2)
    except MemoryError as e:
        mux = get_mean(x,axis=0)
        muy = get_mean(y,axis=0)
        sigx = get_std(x,mean=mux,axis=0)
        sigy = get_std(y,mean=muy,axis=0)
    x = x-mux
    y = y-muy
    numer = (x*y).mean(0)
    denom = sigx*sigy
    r = numer/denom
    if shape is not None:
        r = r.reshape(shape)
    return r

def mtx_cor(X, Y, batch_size=500, to_numpy=False):
    """
    Creates a correlation matrix for X and Y using the GPU

    X: torch tensor or ndarray (T, C) or (T, C, H, W)
    Y: torch tensor or ndarray (T, K) or (T, K, H1, W1)
    batch_size: int
        batches the calculation if this is not None
    to_numpy: bool
        if true, returns matrix as ndarray

    Returns:
        cor_mtx: (C,K)
            the correlation matrix
    """
    if len(X.shape) > 2:
        X = X.reshape(len(X), -1)
    if len(Y.shape) > 2:
        Y = Y.reshape(len(Y), -1)
    to_numpy = type(X) == type(np.array([])) or to_numpy
    X = torch.FloatTensor(X)
    Y = torch.FloatTensor(Y)
    xmean = X.mean(0)
    xstd = torch.sqrt(((X-xmean)**2).mean(0))
    ymean = Y.mean(0)
    ystd = torch.sqrt(((Y-ymean)**2).mean(0))
    X = ((X-xmean)/(xstd+1e-5)).permute(1,0)
    Y = (Y-ymean)/(ystd+1e-5)

    with torch.no_grad():
        if batch_size is None:
            X = X.to(DEVICE)
            Y = Y.to(DEVICE)
            cor_mtx = torch.einsum("it,tj->ij", X, Y).detach().cpu()
        else:
            cor_mtx = []
            for i in range(0,len(X),batch_size): # loop over x neurons
                sub_mtx = []
                x = X[i:i+batch_size].to(DEVICE)

                # Loop over y neurons
                for j in range(0,Y.shape[1], batch_size):
                    y = Y[:,j:j+batch_size].to(DEVICE)
                    cor_block = torch.einsum("it,tj->ij",x,y)
                    cor_block = cor_block.detach().cpu()
                    sub_mtx.append(cor_block)
                cor_mtx.append(torch.cat(sub_mtx,dim=1))
            cor_mtx = torch.cat(cor_mtx, dim=0)
    cor_mtx = cor_mtx/len(Y)
    if to_numpy:
        return cor_mtx.numpy()
    return cor_mtx

def revcor(X, y, batch_size=500, to_numpy=False, ret_norm_stats=False,
                                                       verbose=False):
    """
    Reverse correlates X and y using the GPU

    X: torch tensor (T, C) or (T, C, H, W)
    y: torch tensor (T,)
    batch_size: int
    ret_norm_stats: bool
        if true, returns the normalization statistics of both the X
        and y tensors

    returns:
        
    """
    if type(X) == type(np.array([])):
        X = torch.FloatTensor(X)
        y = torch.FloatTensor(y)
    Xshape = X.shape[1:]
    X = X.reshape(len(X), -1)
    xmean = get_mean(X)
    xstd = get_std(X, mean=xmean)
    xnorm_stats = [xmean, xstd]
    ymean = get_mean(y)
    ystd = get_std(y, mean=ymean)
    ynorm_stats = [ymean, ystd]
    with torch.no_grad():
        if batch_size is None:
            X = (X-xmean)/(xstd+1e-5)
            y = (y-ymean)/(ystd+1e-5)
            matmul = torch.einsum("ij,i->j",X.to(DEVICE),y.to(DEVICE))
            sta = (matmul/len(X)).detach().cpu()
        else:
            n_samples = 0
            cumu_sum = 0
            rng = range(0,len(X),batch_size)
            if verbose:
                rng = tqdm(rng)
            for i in rng:
                x = X[i:i+batch_size]
                truth = y[i:i+batch_size].squeeze()
                x = (x-xmean)/(xstd+1e-5)
                s = "ij,i->j"
                matmul = torch.einsum(s,x.to(DEVICE),truth.to(DEVICE))
                cumu_sum = cumu_sum + matmul.cpu().detach()
                n_samples += len(x)
            sta = cumu_sum/n_samples
    sta = sta.reshape(Xshape)
    if to_numpy:
        sta = sta.numpy()
    if ret_norm_stats:
        return sta, xnorm_stats, ynorm_stats
    return sta

def revcor_sta(model, layer, cell_index, layer_shape=None,
                                          n_samples=20000,
                                          batch_size=500,
                                          contrast=1,
                                          to_numpy=False,
                                          verbose=True):
    """
    Calculates the STA using the reverse correlation method.

    model: torch Module
    layer: str
        name of layer in model
    cell_index: int or list-like (idx,) or (chan, row, col)
    layer_shape: list-like (n_chan, n_row, n_col)
        desired shape of layer. useful when layer is flat, but
        desired shape is not
    n_samples: int
        number of whitenoise samples to use in calculation
    batch_size: int
        size of batching for calculations performed on GPU
    contrast: float
        contrast of whitenoise used to calculate STA
    to_numpy: bool
        returns values as numpy arrays if true, else as torch tensors
    """
    noise = contrast*np.random.randn(n_samples,*model.img_shape[1:])
    nh = model.img_shape[0]
    nx = model.img_shape[1]
    X = tdrstim.concat(noise, nh=nh, nx=nx)
    with torch.no_grad():
        response = inspect(model, X, insp_keys=set([layer]),
                                      batch_size=batch_size,
                                      to_numpy=False)
    resp = response[layer]
    if layer_shape is not None:
        resp = resp.reshape(-1,*layer_shape)
    if type(cell_index) == type(int()):
        resp = resp[:,cell_index]
    elif len(cell_index) == 2:
        resp = resp[:,cell_index[0]]
    else:
        resp = resp[:,cell_index[0], cell_index[1], cell_index[2]]
    sta = revcor(X, resp, batch_size=batch_size, to_numpy=to_numpy)
    return sta.reshape(model.img_shape)

def requires_grad(model, state):
    """
    Turns grad calculations on and off for all parameters in the model

    model: torch Module
    state: bool
        if true, gradient calculations are performed
        if false, gradient calculations are not
    """
    for p in model.parameters():
        try:
            p.requires_grad = state
        except:
            pass

def find_peaks(array):
    """
    Helpful function for finding the peaks of the array

    array: list of integers or floats

    returns:
        maxima: list of integers or floats
            a list of the values that are larger than both of their
            immediate neighbors
    """
    if len(array) == 2 and array[0] > array[1]:
        return [0]
    if len(array) == 2 and array[0] < array[1]:
        return [1]
    if len(array) <= 2:
        return [0]

    maxima = []
    if array[0] > array[1]:
        maxima.append(0)
    for i in range(1,len(array)-1):
        if array[i-1] < array[i] and array[i+1] < array[i]:
            maxima.append(i)
    if array[-2] < array[-1]:
        maxima.append(len(array)-1)

    return maxima

def parallel_shuffle(arrays, set_seed=-1):
    """
    shuffles multiple arrays using the same shuffle permutation.
    shuffles in place using numpy's RandomState shuffle func.

    Parameters:
    -----------
    arrays : List of NumPy arrays.
    set_seed : Seed value if int >= 0, else seed is random.
    """
    assert all(len(arr) == len(arrays[0]) for arr in arrays)
    seed = np.random.randint(0, 2**(32 - 1) - 1) if set_seed < 0\
                                                    else set_seed
    for arr in arrays:
        rstate = np.random.RandomState(seed)
        rstate.shuffle(arr)

def multi_shuffle(arrays):
    """
    shuffles multiple arrays using the same shuffle permutation.
    shuffles in place using custom permutation loop.

    arrays: list of equal length sequences
        this is a list of sequences that will be shuffled in parallel.
    """
    for i in reversed(range(len(arrays[0]))):
        idx = np.random.randint(0, i+1)
        for j in range(len(arrays)):
            temp = copy.deepcopy(arrays[j][i:i+1])
            arrays[j][i:i+1] = copy.deepcopy(arrays[j][idx:idx+1])
            arrays[j][idx:idx+1] = temp
            del temp
    return arrays

def stackedconv2d_to_conv2d(stackedconv2d):
    """
    Takes the whole LinearStackedConv2d module and converts it to a
    single Conv2d

    stackedconv2d - torch LinearStacked2d module
    """
    convs = stackedconv2d.convs
    filters = []
    for conv in convs:
        if "weight" in dir(conv):
            filters.append(conv.weight)
    stacked_filt = stack_filters(filters)
    out_chan, in_chan, k_size, _ = stacked_filt.shape
    conv2d = nn.Conv2d(in_chan, out_chan, k_size)
    conv2d.weight.data = stacked_filt
    try:
        conv2d.bias.data = convs[-1].bias
    except Exception as e:
        print("Bias transfer failed..")
    return conv2d

def stack_filters(filt_list):
    """
    Combines the list of filters into a single stacked filter.

    filt_list - list of torch FloatTensors with shape (Q, R, K, K)
        the first filter in the conv sequence.
    """
    stacked_filt = filt_list[0]
    for i in range(1,len(filt_list)):
        stacked_filt = stack_filter(stacked_filt, filt_list[i])
    return stacked_filt

def stack_filter(base_filt, stack_filt):
    """
    Combines two convolutional filters in a mathematically equal way
    to performing the convolutions one after the other. Forgive the
    quadruple for-loop... There's probably a way to parallelize but
    this was much easier to implement.

    base_filt - torch FloatTensor (Q, R, K1, K1)
        the first filter in the conv sequence.
    stack_filt - torch FloatTensor (S, Q, K2, K2)
        the filter following base_filt in the conv sequence.
    """
    device = torch.device("cuda:0") if base_filt.is_cuda else\
                                           torch.device("cpu")
    kb = base_filt.shape[-1]
    ks = stack_filt.shape[-1]
    new_filt = torch.zeros(stack_filt.shape[0], base_filt.shape[1],
                                         base_filt.shape[2]+(ks-1),
                                         base_filt.shape[3]+(ks-1))
    new_filt = new_filt.to(device)
    for out_chan in range(stack_filt.shape[0]):
        # same as out_chan in base_filt/new_filt
        for in_chan in range(stack_filt.shape[1]):
            for row in range(stack_filt.shape[2]):
                for col in range(stack_filt.shape[3]):
                    oc = out_chan
                    ic = in_chan
                    temp = base_filt[ic]*stack_filt[oc, ic, row, col]
                    new_filt[oc:oc+1,:,row:row+kb,col:col+kb] += temp
    return new_filt

def get_grad(model, X, cell_idxs=None):
    """
    Gets the gradient of the model output with respect to the stimulus
    X

    model - torch module
    X - numpy array or torch float tensor (B,C,H,W)
    layer_idx - None or int
        model layer to use for grads with respect
    cell_idxs - None or list-like (N)
    """
    tensor = torch.FloatTensor(X)
    tensor.requires_grad = True
    back_to_train = model.training
    model.eval()
    back_to_cpu = next(model.parameters()).is_cuda
    model.to(DEVICE)
    outs = model(tensor.to(DEVICE))
    if cell_idxs is not None:
        outs = outs[:,cell_idxs]
    outs.sum().backward()
    grad = tensor.grad.data.detach().cpu().numpy()
    if back_to_train:
        model.train()
    if back_to_cpu:
        model.detach().cpu()
    return grad


def conv_backwards(z, filt, xshape):
    """
    Used for gradient calculations specific to a single convolutional
    filter. '_out' in the dims refers to the output of the forward
    pass of the convolutional layer. '_in' in the dims refers to the
    input of the forward pass of the convolutional layer.

    z - torch FloatTensor (Batch, C_out, W_out, H_out)
        the accumulated activation gradient up to this point
    filt - torch FloatTensor (C_in, k_w, k_h)
        a single convolutional filter from the convolutional layer
        note that this is taken from the greater layer that has
        dims (C_out, C_in
    xshape - list like 
        the shape of the activations of interest. the shape should
        be (Batch, C_in, W_in, H_in)
    """
    dx = torch.zeros(xshape)
    if filt.is_cuda:
        dx = dx.to(filt.get_device())
    filt_temp = filt.view(-1)[:,None]
    for chan in range(z.shape[1]):
        for row in range(z.shape[2]):
            for col in range(z.shape[3]):
                ztemp = z[:,chan,row,col]
                matmul = torch.mm(filt_temp, ztemp[None])
                fs2 = filt.shape[-2]
                fs1 = filt.shape[-1]
                matmul = matmul.permute(1,0).view(dx.shape[0],
                                                  dx.shape[1],
                                                  fs2,
                                                  fs1)
                dx[:,:,row:row+fs2, col:col+fs1] += matmul
    return dx
