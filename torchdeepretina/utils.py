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

DEVICE = torch.device("cuda:0")

def load_json(file_name):
    with open(file_name) as f:
        s = f.read()
        j = json.loads(s)
    return j

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

def get_cuda_info():
    """
    Get the current gpu usage. 

    Partial credit to https://discuss.pytorch.org/t/access-gpu-memory-usage-in-pytorch/3192/3

    Returns
    -------
    usage: dict
        Keys are device ids as integers.
        Values are memory usage as integers in MB.
    """
    result = subprocess.check_output(
        [
            'nvidia-smi', '--query-gpu=memory.used',
            '--format=csv,nounits,noheader'
        ], encoding='utf-8')
    # Convert lines into a dictionary
    gpu_memory = [int(x) for x in result.strip().split('\n')]
    cuda_info = []
    for i,used_mem in zip(range(len(gpu_memory)), gpu_memory):
        info = dict()
        tot_mem = torch.cuda.get_device_properties(i).total_memory/1028**2
        info['total_mem'] = tot_mem
        info['used_mem'] = used_mem
        info['remaining_mem'] = tot_mem-used_mem
        cuda_info.append(info)
    return cuda_info

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

def freeze_weights(model, unfreeze=False):
    for p in model.parameters():
        try:
            p.requires_grad = unfreeze
        except:
            pass

def requires_grad(model, state):
    for p in model.parameters():
        try:
            p.requires_grad = state
        except:
            pass

def find_local_maxima(array):
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
    Parameters:
    -----------
    arrays : List of NumPy arrays.
    set_seed : Seed value if int >= 0, else seed is random.
    """
    assert all(len(arr) == len(arrays[0]) for arr in arrays)
    seed = np.random.randint(0, 2**(32 - 1) - 1) if set_seed < 0 else set_seed
    for arr in arrays:
        rstate = np.random.RandomState(seed)
        rstate.shuffle(arr)

def multi_shuffle(arrays):
    for i in reversed(range(len(arrays[0]))):
        idx = np.random.randint(0, i+1)
        for j in range(len(arrays)):
            temp = copy.deepcopy(arrays[j][i:i+1])
            arrays[j][i:i+1] = copy.deepcopy(arrays[j][idx:idx+1])
            arrays[j][idx:idx+1] = temp
            del temp
    return arrays

def save_checkpoint(save_dict, folder, exp_id, del_prev=False):
    """
    save_dict: dict
        all things to save to file
    folder: str
        path of folder to be saved to
    exp_id: str
        additional name to be prepended to path file string
    del_prev: bool
        if true, deletes the model_state_dict and optim_state_dict of the save of the
        previous file (saves space)
    """
    if del_prev:
        prev_path = os.path.join(folder, exp_id + "_epoch_" + str(save_dict['epoch']-1) + '.pth')
        if os.path.exists(prev_path):
            device = torch.device("cpu")
            data = torch.load(prev_path, map_location=device)
            keys = list(data.keys())
            for key in keys:
                if "state_dict" in key:
                    del data[key]
            torch.save(data, prev_path)
        elif save_dict['epoch'] != 0:
            print("Failed to find previous checkpoint", prev_path)
    path = os.path.join(folder, exp_id + '_epoch_' + str(save_dict['epoch'])) + '.pth'
    path = os.path.abspath(os.path.expanduser(path))
    torch.save(save_dict, path)

def load_checkpoint(checkpt_path):
    """
    Can load a specific model file both architecture and state_dict if the file 
    contains a model_state_dict key, or can just load the architecture.

    checkpt_path: str
        path to checkpoint file
    """
    data = torch.load(folder, map_location=torch.device("cpu"))
    try:
        model = globals()[data['model_type']](**data['model_hyps'])
    except Exception as e:
        print(e)
        print("Likely the checkpoint you are using is deprecated. Try using analysis.load_model()")
    try:
        model.load_state_dict(data['model_state_dict'])
    except KeyError as e:
        print("Failed to load state_dict. This checkpoint does not contain a model state_dict!")
    return model

def stackedconv2d_to_conv2d(stackedconv2d):
    """
    Takes the whole LinearStackedConv2d module and converts it to a single Conv2d

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

def get_grad(model, X, layer_idx=None, cell_idxs=None):
    """
    Gets the gradient of the model output with respect to the stimulus X

    model - torch module
    X - numpy array or torch float tensor (B,C,H,W)
    layer_idx - None or int
        model layer to use for grads with respec
    cell_idxs - None or list-like (N)
    """
    tensor = torch.FloatTensor(X)
    tensor.requires_grad = True
    back_to_train = model.training
    if back_to_train:
        model.eval()
    back_to_cpu = next(model.parameters()).is_cuda
    model.cuda()
    outs = model(tensor.cuda())
    if cell_idxs is not None:
        outs = outs[:,cell_idxs]
    outs.sum().backward()
    grad = tensor.grad.data.detach().cpu().numpy()
    if back_to_train:
        model.train()
    if back_to_cpu:
        model.detach().cpu()
    return grad

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
    Combines two convolutional filters in a mathematically equal way to performing
    the convolutions one after the other. Forgive the quadruple for-loop... There's
    probably a way to parallelize but this was much easier to implement.

    base_filt - torch FloatTensor (Q, R, K1, K1)
        the first filter in the conv sequence.
    stack_filt - torch FloatTensor (S, Q, K2, K2)
        the filter following base_filt in the conv sequence.
    """
    device = torch.device("cuda:0") if base_filt.is_cuda else torch.device("cpu")
    kb = base_filt.shape[-1]
    ks = stack_filt.shape[-1]
    new_filt = torch.zeros(stack_filt.shape[0], base_filt.shape[1], base_filt.shape[2]+(ks-1), base_filt.shape[3]+(ks-1))
    new_filt = new_filt.to(device)
    for out_chan in range(stack_filt.shape[0]):
        for in_chan in range(stack_filt.shape[1]): # same as out_chan in base_filt/new_filt
            for row in range(stack_filt.shape[2]):
                for col in range(stack_filt.shape[3]):
                    new_filt[out_chan:out_chan+1, :, row:row+kb, col:col+kb] += base_filt[in_chan]*stack_filt[out_chan, in_chan, row, col]
    return new_filt

def conv_backwards(z, filt, xshape):
    """
    Used for gradient calculations specific to a single convolutional filter.
    '_out' in the dims refers to the output of the forward pass of the convolutional layer.
    '_in' in the dims refers to the input of the forward pass of the convolutional layer.

    z - torch FloatTensor (Batch, C_out, W_out, H_out)
        the accumulated activation gradient up to this point
    filt - torch FloatTensor (C_in, k_w, k_h)
        a single convolutional filter from the convolutional layer
        note that this is taken from the greater layer that has dims (C_out, C_in
    xshape - list like 
        the shape of the activations of interest. the shape should be (Batch, C_in, W_in, H_in)
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
                matmul = matmul.permute(1,0).view(dx.shape[0], dx.shape[1], filt.shape[-2], filt.shape[-1])
                dx[:,:,row:row+filt.shape[-2], col:col+filt.shape[-1]] += matmul    
    return dx

class GaussRegularizer:
    def __init__(self, model, conv_idxs, std=1):
        """
        model - torch nn Module
        conv_idxs - list of indices of convolutional layers
        std - int
            standard deviation of gaussian in terms of pixels
        """
        assert "sequential" in dir(model) and type(model.sequential[conv_idxs[0]]) == type(nn.Conv2d(1,1,1)) # Needs to be Conv2d module
        self.weights = [model.sequential[i].weight for i in conv_idxs]
        self.std = std
        self.gaussians = []
        for i,weight in enumerate(self.weights):
            shape = weight.shape[1:]
            half_width = shape[1]//2
            pdf = 1/np.sqrt(2*np.pi*self.std**2) * np.exp(-(np.arange(-half_width,half_width+1)**2/(2*self.std**2)))
            gauss = np.outer(pdf, pdf)
            inverted = 1/(gauss+1e-5)
            inverted = (inverted-np.min(inverted))/np.max(inverted)
            full_gauss = np.asarray([gauss for i in range(shape[0])])
            self.gaussians.append(torch.FloatTensor(full_gauss))
    
    def get_loss(self):
        if self.weights[0].data.is_cuda and not self.gaussians[0].is_cuda:
            self.gaussians = [g.cuda() for g in self.gaussians]
        loss = 0
        for weight,gauss in zip(self.weights,self.gaussians):
            loss += (weight*gauss).mean()
        return loss

def get_hs(model, batch_size, device, I20=None):
    hs = []
    hs.append(torch.zeros(batch_size, *model.h_shapes[0]).to(device))
    hs[0][:,0] = 1
    if isinstance(I20, np.ndarray):
        hs[0][:,3] = torch.from_numpy(I20)[:,None].to(device)
    hs.append(deque([],maxlen=model.seq_len))
    for i in range(model.seq_len):
        hs[1].append(torch.zeros(batch_size, *model.h_shapes[1]).to(device))
    return hs
