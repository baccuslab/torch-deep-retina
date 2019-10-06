import numpy as np
import torch
import torch.nn as nn
import h5py as h5
import os
import sys
import pickle
from torchdeepretina.models import *
import matplotlib.pyplot as plt
import torchdeepretina.stimuli as stimuli
import torchdeepretina.utils as tdrutils
from torchdeepretina.physiology import Physio
import pyret.filtertools as ft
import scipy
import re
import pickle
from tqdm import tqdm
import gc
import resource
import time
import math
import pandas as pd

DEVICE = torch.device("cuda:0")

def make_correlation_frame(model_stats):
    """
    model_stats: dict
        keys should be model save folders
    """
    keys = ['stim_type', 'cellfile', 'cell_type', 'save_folder', 'cell_idx',
                            'layer', 'channel', 'row', 'col', 'cor_coef']
    frame = {k:[] for k in keys}
    for folder in model_stats.keys():
        if 'intrnrn_info' not in model_stats[folder]:
            continue
        infos = model_stats[folder]['intrnrn_info']
        for info in infos:
            const_vals = [info[k] for k in keys[:5]]
            for cor in info['all_correlations']:
                layer, channel, (row, col), cor_coef = cor
                vals = const_vals + [layer, channel, row, col, cor_coef]
                for k,v in zip(keys, vals):
                    frame[k].append(v)
    return pd.DataFrame(frame)

def index_of(arg, arr):
    """
    Used to find the index of the argument in the array.
    """
    for i in range(len(arr)):
        if arg == arr[i]:
            return i
    return -1

def make_intrnrn_frame(model_stats, headers, main_dir="../training_scripts"):
    data = dict()
    for header in headers:
        data[header] = []
    for folder in model_stats.keys():
        if "intrnrn_info" in model_stats[folder]:
            intrnrn_info = model_stats[folder]['intrnrn_info'] # list of data dicts for each interneuron cell
            for info in intrnrn_info:
                untouched_keys = set(data.keys())
                for key in info.keys():
                    if key in data:
                        untouched_keys.remove(key)
                        data[key].append(info[key])
                for k in list(untouched_keys):
                    data[k].append(None)
    return pd.DataFrame(data)

def make_model_frame(model_stats, headers, main_dir="../training_scripts"):
    data = dict()
    for header in headers:
        data[header] = []
    for folder in model_stats.keys():
        untouched_keys = set(data.keys())
        untouched_keys.remove("save_folder")
        for k in model_stats[folder].keys():
            if k in data and k in untouched_keys:
                untouched_keys.remove(k)
                data[k].append(model_stats[folder][k])
        hyps = get_hyps(folder, main_dir)
        for k,v in hyps.items():
            if k in data and k in untouched_keys:
                untouched_keys.remove(k)
                data[k].append(v)
        with open(os.path.join(main_dir, folder, "hyperparams.txt")) as f:
            architecture = []
            for i,line in enumerate(f):
                if "(" in line or ")" in line:
                    l = line.replace("\n", "#")
                    architecture.append(l)
            data['architecture'].append("".join(architecture))
            untouched_keys.remove('architecture')
        data['save_folder'].append(folder)
        for k in list(untouched_keys):
            data[k].append(None)
    return pd.DataFrame(data)

def get_architecture(folder, data, hyps=None, main_dir="../training_scripts/"):
    # load_model changed to get_architecture
    try:
        model = globals()[data['model_type']](**data['model_hyps'])
    except:
        try:
            hyps= hyps if hyps is not None else get_hyps(folder, main_dir)
            hyps['model_type'] = hyps['model_type'].split(".")[-1].split("\'")[0].strip()
            hyps['model_type'] = globals()[hyps['model_type']]
            model = hyps['model_type'](**data['model_hyps'])
        except Exception as e:
            model_hyps = {"n_units":5,"noise":float(hyps['noise'])}
            if "bias" in hyps:
                model_hyps['bias'] = hyps['bias'] == "True"
            if "chans" in hyps:
                model_hyps['chans'] = [int(x) for x in
                                       hyps['chans'].replace("[", "").replace("]", "").strip().split(",")]
            if "adapt_gauss" in hyps:
                model_hyps['adapt_gauss'] = hyps['adapt_gauss'] == "True"
            if "linear_bias" in hyps:
                model_hyps['linear_bias'] = hyps['linear_bias'] == "True"
            if "softplus" in hyps:
                model_hyps['softplus'] = hyps['softplus'] == "True"
            if "abs_bias" in hyps:
                model_hyps['abs_bias'] = hyps['abs_bias'] == "True"
            fn_args = set(hyps['model_type'].__init__.__code__.co_varnames)
            keys = list(model_hyps.keys())
            for k in keys:
                if k not in fn_args:
                    del model_hyps[k]
            model = hyps['model_type'](**model_hyps)
    return model

def get_hyps(folder, main_dir="../training_scripts"):
    try:
        path = os.path.join(main_dir, folder, "hyperparams.json")
        return tdrutils.load_json(path)
    except Exception as e:
        stream = tdr.analysis.stream_folder(path)
        hyps = dict()
        for k in stream.keys():
            if "state_dict" not in k and "model_hyps" not in k:
                hyps[k] = stream[k]
    return hyps

def load_model(folder, data=None, hyps=None, main_dir="../training_scripts/"):
    """
    Can load a specific model file both architecture and state_dict if the file 
    contains a model_state_dict key, or can just load the architecture.

    folder: str
        can be a path to a specific model folder or to a specific checkpoint file.
        if argument is path to a checkpoint file, then data is unnecessary parameter.
    data: dict or None
        this should either be the loaded checkpoint dict if folder is a model folder,
        or it can be None if the folder is a path to a checkpoint file.
    hyps: dict
        specific hyperparameters used in the model. this is an optional parameter to 
        save time in the function.
    main_dir: string
        this is the folder to search for the argued folder.

    """
    if data is None:
        data = torch.load(folder, map_location=torch.device("cpu"))
    model = get_architecture(folder, data, hyps, main_dir)
    try:
        model.load_state_dict(data['model_state_dict'])
    except KeyError as e:
        print("Failed to load state_dict. This checkpoint does not contain a model state_dict!")
    return model

def requires_grad(model, state):
    for p in model.parameters():
        try:
            p.requires_grad = state
        except:
            pass

def read_model(folder):
    """
    Recreates model architecture and loads the saved statedict from a model folder
    """
    try:
        _, _, fs = next(os.walk(folder.strip()))
    except Exception as e:
        print(e)
        print("It is likely that folder", folder.strip(),"does not exist")
        assert False
    for i in range(len(fs)+100):
        f = os.path.join(folder.strip(),"test_epoch_{0}.pth".format(i))
        try:
            with open(f, "rb") as fd:
                data = torch.load(fd, map_location=torch.device("cpu"))
        except Exception as e:
            pass
    try:
        model = data['model']
    except Exception as e:
        model = load_model(folder, data)
    try:
        model.load_state_dict(data['model_state_dict'])
    except RuntimeError as e:
        keys = list(data['model_state_dict'].keys())
        for key in keys:
            if "cuda_param" in key:
                new_key = key.replace("cuda_param", "sigma")
                data['model_state_dict'][new_key] = data['model_state_dict'][key]
                del data['model_state_dict'][key]
        model.load_state_dict(data['model_state_dict'])
    model = model.to(DEVICE)
    return model

def read_model_file(file_name, model_type=None):
    """
    file_name: str
        path to the model save file. The save should contain "model_hyps", "model_state_dict",
        and ideally "model_type" as keys.
    model_type: str or None
        optional string name of model class to be loaded
    """
    data = torch.load(file_name, map_location="cpu")
    if model_type is None:
        model_type = data['model_type']
    model = globals()[model_type](**data['model_hyps'])
    model.load_state_dict(data['model_state_dict'])
    return model

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
            hook = tdrutils.get_hook(hook_outs,key=layer,to_numpy=False)
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

def inspect(*args, **kwargs):
    """
    Get the response from the argued layers in the model as np arrays

    returns dict of np arrays
    """
    return tdrutils.inspect(*args,**kwargs)

def compute_sta(model, contrast, layer, cell_index, layer_shape=None, verbose=True):
    """
    Computes the STA using the model gradient
    """
    # generate some white noise
    #X = stim.concat(white(1040, contrast=contrast)).copy()
    X = stimuli.concat(contrast*np.random.randn(10000,50,50))
    X = torch.FloatTensor(X)
    X.requires_grad = True

    # compute the gradient of the model with respect to the stimulus
    drdx = get_stim_grad(model, X, layer, cell_index, layer_shape=layer_shape, verbose=verbose)
    sta = drdx.mean(0)

    del X
    return sta

def rev_cor(response, X):
    """
    Computes the STA using reverse correlation

    response - ndarray (T,)
    X - ndarray (T,N)
    """
    X = X.reshape(len(X), -1)
    X = (X-X.mean(0))/(X.std(0)+1e-5)
    resp = scipy.stats.zscore(response)
    matmul = np.einsum("ij,i->j",X, resp.squeeze())
    sta = matmul/len(X)
    return sta

def batch_compute_model_response(stimulus, model, batch_size=500, recurrent=False, insp_keys={'all'}, cust_h_init=False, verbose=False):
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

def stream_folder(folder,main_dir=""):
    """
    Gets the save dict from the save folder.
    """
    try:
        _, _, fs = next(os.walk(folder.strip()))
    except Exception as e:
        print(e)
        print("It is likely that folder", folder.strip(),"does not exist")
        assert False
    for i in range(len(fs)+200):
        f = os.path.join(folder.strip(),"test_epoch_{0}.pth".format(i))
        try:
            data = torch.load(f, map_location=torch.device("cpu"))
        except Exception as e:
            pass
    return data

def get_model_folders(main_folder):
    """
    Returns a list of paths to the model folders contained within the argued main_folder

    main_folder - str
        path to main folder
    """
    for d, sub_ds, files in os.walk(main_folder):
        folders = []
        for sub_d in sub_ds:
            contents = os.listdir(os.path.join(d,sub_d))
            for content in contents:
                if ".pt" in content:
                    folders.append(sub_d)
                    break
    return folders

#def analysis_pipeline(main_folder):
#    """
#    Evaluates model on test set, calculates interneuron correlations, and creates retinal phenomena figures.
#    """
#    model_folders = get_model_folders(main_folder)
#    main_frame = 
#    for folder in model_folders:
#        frame = analyze_model(folder)
