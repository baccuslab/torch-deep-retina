import numpy as np
import copy
import torch
import subprocess
import json
import os

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

def freeze_weights(model, unfreeze=False):
    for p in model.parameters():
        try:
            p.requires_grad = unfreeze
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

def stack_filters(base_filt, stack_filt):
    """
    Combines two convolutional filters in a mathematically equal way to performing
    the convolutions one after the other. Forgive the quadruple for-loop... There's
    probably a way to parallelize but this was much easier to implement.

    base_filt - torch FloatTensor (Q, R, K1, K1)
        the first filter in the conv sequence.
    stack_filt - torch FloatTensor (S, Q, K2, K2)
        the filter following base_filt in the conv sequence.
    """
    kb = base_filt.shape[-1]
    ks = stack_filt.shape[-1]
    new_filt = torch.zeros(stack_filt.shape[0], base_filt.shape[1], base_filt.shape[2]+(ks-1), base_filt.shape[3]+(ks-1))
    new_filt = new_filt.to(base_filt.get_device())
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

