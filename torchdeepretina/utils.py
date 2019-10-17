import numpy as np
import copy
import torch
import torch.nn as nn
import subprocess
import json
import os
import torchdeepretina.stimuli as tdrstim
from torchdeepretina.physiology import Physio
from tqdm import tqdm
import pyret.filtertools as ft
import scipy.stats
import pickle

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
            layer_dict[key] = out.detach().cpu().numpy()
    elif to_cpu:
        def hook(module, inp, out):
            layer_dict[key] = out.cpu()
    else:
        def hook(module, inp, out):
            layer_dict[key] = out
    return hook

def inspect(model, X, insp_keys={}, batch_size=None, to_numpy=True, verbose=False):
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
            X = X.cuda()
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
                x = x.cuda()
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

def get_stim_grad(model, X, layer, cell_idx, batch_size=500, layer_shape=None, verbose=True):
    """
    Gets the gradient of the model output at the specified layer and cell idx with respect
    to the inputs (X). Returns a gradient array with the same shape as X.
    """
    if verbose:
        print("layer:", layer)
    requires_grad(model, False)
    device = next(model.parameters()).get_device()

    if model.recurrent:
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
        if model.recurrent:
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

def compute_sta(model, contrast, layer, cell_index, layer_shape=None, verbose=True):
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
    drdx = get_stim_grad(model, X, layer, cell_index, layer_shape=layer_shape, verbose=verbose)
    sta = drdx.mean(0)

    del X
    return sta

def get_mean(x, axis=None, batch_size=1000):
    """
    Returns mean of x along argued axis. Used in cases of large datasets.

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
        return cumu_sum/len(x)

def get_std(x, axis=None, batch_size=1000, mean=None):
    """
    Returns std of x along argued axis. Used in cases of large datasets.

    x: ndarray or torch tensor
    axis: int
    batch_size: int
        size of increment when calculating mean
    mean: int or ndarray or torch tensor
        The mean to be used in calculating the std. If None, mean is automatically calculated.
        If ndarray or torch tensor, must match datatype of x.
    """
    if mean is None:
        mean = get_mean(x,axis,batch_size)
    cumu_sum = 0
    if axis is None:
        for i in range(0,len(x), batch_size):
            cumu_sum = cumu_sum + ((x[i:i+batch_size]-mean)**2).sum()
        return torch.sqrt(cumu_sum/x.numel())
    else:
        for i in range(0,len(x), batch_size):
            cumu_sum = cumu_sum + ((x[i:i+batch_size]-mean)**2).sum(axis)
        return torch.sqrt(cumu_sum/len(x))


def pearsonr(x,y):
    """
    Calculates the pearson correlation coefficient. This gives same results as scipy's
    version but allows you to calculate the coefficient over much larger data sizes.
    Additionally allows calculation for torch tensors.

    x: ndarray or torch tensor 
    y: ndarray or torch tensor 
    """
    x = x.reshape(len(x), -1)
    y = y.reshape(len(y), -1)
    try:
        mux = x.mean()
        muy = y.mean()
        sigx = x.std()
        sigy = y.std()
    except MemoryError as e:
        mux = get_mean(x) 
        muy = get_mean(y) 
        sigx = get_std(x,mean=mux)
        sigy = get_std(y,mean=muy)
    x = x-mux
    y = y-muy
    numer = (x*y).mean()
    denom = sigx*sigy
    r = numer/denom
    return r

class poly1d:
    """
    Creates a polynomial with the argued fit
    """
    def __init__(self, fit):
        self.coefficients = fit
        self.poly = self.get_poly(fit)

    def get_poly(self, fit):
        def poly(x):
            cumu_sum = 0
            for i in range(len(fit)):
                cumu_sum = cumu_sum + fit[i]*(x**(len(fit)-i-1))
            return cumu_sum
        return poly
    
    def __call__(self, x):
        return self.poly(x)

def revcor_sta(model, layers=['sequential.0','sequential.6'], chans=[8,8], n_samples=10000,
                                                             batch_size=500, verbose=True):
    """
    Computes the sta using reverse correlation. Uses the central unit for computation

    model - torch Module

    returns:
        dict of sta lists for each channel in each layer
        keys: layer names
            vals: lists of stas for each channel in the layer
    """
    if type(layers) == type(str()):
        layers = [layers]
    noise = np.random.randn(n_samples,50,50)
    filter_size = model.img_shape[0]
    X = tdrstim.concat(noise, nh=filter_size)
    noise = noise[filter_size:]
    response = inspect(model, X, insp_keys=set(layers), batch_size=batch_size, to_numpy=True)
    stas = {layer:[] for layer in layers}
    for layer,chan in zip(layers,chans):
        resp = response[layer]
        if len(resp.shape) == 2:
            if layer == "sequential.2":
                resp = resp.reshape(len(resp), len(chan), *model.shapes[0])
            else:
                resp = resp.reshape(len(resp), len(chan), *model.shapes[1])
        center = resp.shape[-1]//2
        chan = tqdm(range(chan)) if verbose else range(chan)
        for c in chan:
            sta,_ = ft.revcorr(noise, scipy.stats.zscore(resp[:,c,center,center]),
                                                             0,model.img_shape[0])
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

def save_ln(model,file_name):
    """
    Saves an LNModel to file

    model: LNModel object (see models.py)
    file_name: str
        path to save model to
    """
    model_dict = {"filt":model.filt, "fit":model.poly.coefficients, "span":model.span, 
                                                                "center":model.center}
    with open(file_name,'wb') as f:
        pickle.dump(model_dict,f)

def save_ln_group(models, file_name):
    """
    Saves group of LNModels to file. Often models are trained using crossvalidation,
    which implicitely ties the models together as an individual unit.

    model: list of LNModel objects (see models.py)
    file_name: str
        path to save models to
    """
    model_dict = {
                  "filts":[model.filt for model in models],
                  "fits":[model.poly.coefficients for model in models],
                  "spans":[model.span for model in models],
                  "centers":[model.center for model in models],
                  "norm_stats":[model.norm_stats for model in models],
                  "cell_file":[model.cell_file for model in models],
                  "cell_idx":[model.cell_idx for model in models],
    }
    with open(file_name,'wb') as f:
        pickle.dump(model_dict,f)
    

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

