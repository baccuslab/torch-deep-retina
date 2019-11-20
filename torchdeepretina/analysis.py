import numpy as np
import torch
import torch.nn as nn
from torch.nn import MSELoss, PoissonNLLLoss
import h5py as h5
import os
import sys
import pickle
from torchdeepretina.models import *
import torchdeepretina.datas as tdrdatas
import matplotlib.pyplot as plt
import torchdeepretina.utils as tdrutils
import torchdeepretina.intracellular as tdrintr
from torchdeepretina.retinal_phenomena import retinal_phenomena_figs
import torchdeepretina.stimuli as tdrstim
import torchdeepretina.visualizations as tdrvis
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
            # list of data dicts for each interneuron cell
            intrnrn_info = model_stats[folder]['intrnrn_info'] 
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
                model_hyps['chans'] = [
                            int(x) for x in hyps['chans'].replace("[", "")\
                            .replace("]", "").strip().split(",")
                          ]
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

def extract_hypstxt(path):
    """
    Gets the hyperparameters from the corresponding hyperparams.txt file

    path: str
        path to the txt file
    """
    hyps = dict()
    with open(path, 'r') as hypfile:
        for line in hypfile:
            if ("(" in line and ")" in line) or ":" not in line:
                continue
            splt = line.strip().split(":")
            splt[0] = splt[0].strip()
            splt[1] = splt[1].strip()
            hyps[splt[0]] = splt[1]
            if hyps[splt[0]].lower() == "true" or hyps[splt[0]].lower() == "false":
                hyps[splt[0]] = hyps[splt[0]] == "true"
            elif hyps[splt[0]] == "None":
                hyps[splt[0]] = None
            elif splt[0] in {"lr", "l1", 'l2', 'noise', 'bn_moment', "bnorm_momentum"}:
                hyps[splt[0]] = float(splt[1])
            elif splt[0] in {"n_epochs", "exp_num", 'batch_size'}:
                hyps[splt[0]] = int(splt[1])
            elif splt[0] in {"img_shape", "chans"} and "," in splt[1]:
                temp = splt[1].replace('[','').replace(']','').split(",")
                hyps[splt[0]] = [int(x.strip()) for x in temp]
    return hyps
                


def get_hyps(folder, main_dir="../training_scripts"):
    try:
        path = os.path.join(main_dir, folder, "hyperparams.json")
        return tdrutils.load_json(path)
    except Exception as e:
        try:
            path = os.path.join(main_dir, folder)
            stream = stream_folder(path)
            hyps = dict()
            for k in stream.keys():
                if "state_dict" not in k and "model_hyps" not in k:
                    hyps[k] = stream[k]
            assert 'lr' in hyps 
            return hyps
        except Exception as ee:
            path = os.path.join(main_dir, folder, "hyperparams.txt")
            return extract_hypstxt(path)

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
        try:
            model.load_state_dict(data['model_state_dict'])
        except:
            sd = data['model_state_dict']
            sd['sequential.4.sigma'] = sd['sequential.3.sigma']
            del sd['sequential.3.sigma']
            model.load_state_dict(sd)
    except KeyError as e:
        print("Failed to load state_dict. This chkpt does not contain a model state_dict!")
    model.norm_stats = data['norm_stats']
    return model

def read_model(folder, ret_metrics=False):
    """
    Recreates model architecture and loads the saved statedict from a model folder

    folder - str
        path to folder that contains model checkpoints
    ret_metrics - bool
        if true, returns the recorded training metric history (i.e. val loss, val acc, etc)
    """
    metrics = dict()
    try:
        _, _, fs = next(os.walk(folder.strip()))
    except Exception as e:
        print(e)
        print("It is likely that folder", folder.strip(),"does not exist")
        assert False
    for i in range(len(fs)+100):
        f = os.path.join(folder.strip(),"test_epoch_{0}.pth".format(i))
        f = os.path.expanduser(f)
        try:
            with open(f, "rb") as fd:
                data = torch.load(fd, map_location=torch.device("cpu"))
            if ret_metrics:
                for k,v in data.items():
                    if k == "loss" or k == "epoch" or k == "val_loss" or k == "val_acc" or\
                       k == "exp_val_acc" or k == "test_pearson":
                        if k not in metrics:
                            metrics[k] = [v]
                        else:
                            metrics[k].append(v)
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
        try:
            model.load_state_dict(data['model_state_dict'])
        except:
            sd = data['model_state_dict']
            sd['sequential.4.sigma'] = sd['sequential.3.sigma']
            del sd['sequential.3.sigma']
            model.load_state_dict(sd)
    model = model.to(DEVICE)
    metrics['norm_stats'] = [data['norm_stats']['mean'], data['norm_stats']['std']]
    model.norm_stats = data['norm_stats']
    if 'y_stats' not in data:
        if 'ymean' in data:
            data['y_stats'] = {'mean':data['ymean'], 'std':data['ystd']}
        else:
            data['y_stats'] = {'mean':None, 'std':None}
    model.y_stats = data['y_stats']
    metrics['y_stats'] = [data['y_stats']['mean'], data['y_stats']['std']]
    metrics
    if ret_metrics:
        return model, metrics
    return model

def read_model_file(file_name, model_type=None, load_state_dict=True):
    """
    file_name: str
        path to the model save file. The save should contain "model_hyps", "model_state_dict",
        and ideally "model_type" as keys.
    model_type: str or None
        optional string name of model class to be loaded
    load_state_dict: bool
        if true, the state dict is loaded into the model
    """
    file_name = os.path.expanduser(file_name)
    data = torch.load(file_name, map_location="cpu")
    if model_type is None:
        model_type = data['model_type']
    model = globals()[model_type](**data['model_hyps'])
    if load_state_dict:
        try:
            model.load_state_dict(data['model_state_dict'])
        except:
            sd = data['model_state_dict']
            sd['sequential.4.sigma'] = sd['sequential.3.sigma']
            del sd['sequential.3.sigma']
            model.load_state_dict(sd)
    model.norm_stats = data['norm_stats']
    return model

def inspect(*args, **kwargs):
    """
    Get the response from the argued layers in the model as np arrays. See utils for details.

    returns dict of np arrays or torch tensors depending on arguments
    """
    return tdrutils.inspect(*args,**kwargs)

def compute_sta(*args, **kwargs):
    """
    Computes the STA using the average of instantaneous receptive 
    fields (gradient of output with respect to input)
    """
    return tdrutils.compute_sta(*args, **kwargs)

def get_sta(*args, **kwargs):
    """
    Computes the sta using reverse correlation. Uses the central unit for computation

    model - torch Module

    returns:
        dict of sta lists for each channel in each layer
        keys: layer names
            vals: lists of stas for each channel in the layer
    """
    return tdrutils.revcor_sta(*args, **kwargs)

def rev_cor(response, X):
    """
    Computes the STA using reverse correlation

    response - ndarray (T,)
    X - ndarray (T,N)

    returns:
        sta - ndarray (N,)
    """
    X = X.reshape(len(X), -1)
    X = (X-X.mean(0))/(X.std(0)+1e-5)
    resp = scipy.stats.zscore(response)
    matmul = np.einsum("ij,i->j",X, resp.squeeze())
    sta = matmul/len(X)
    return sta

def batch_compute_model_response(*args, **kwargs):
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
    return tdrutils.batch_compute_model_response(*args, **kwargs)

def stream_folder(folder,main_dir=""):
    """
    Gets the save dict from the save folder.
    """
    try:
        _, _, fs = next(os.walk(folder.strip()))
    except Exception as e:
        print(e)
        assert False, "It is likely that folder {} does not exist".format(folder.strip())
    for i in range(len(fs)+200):
        f = os.path.join(folder.strip(),"test_epoch_{0}.pth".format(i))
        try:
            data = torch.load(f, map_location=torch.device("cpu"))
        except Exception as e:
            pass
    return data

def get_checkpts(model_folder):
    """
    Returns a list of checkpoint file paths that can be used with read_model_file

    model_folder: str
    """
    checkpts = []
    for d,_,files in os.walk(model_folder):
        for f in files:
            if ".pt" in f:
                path = os.path.join(d,f)
                path = os.path.expanduser(path)
                checkpts.append(path)
        break
    checkpts = sorted(checkpts, key=lambda x: int(x.split(".")[-2].split("_")[-1]))
    return checkpts

def get_model_folders(main_folder):
    """
    Returns a list of paths to the model folders contained within the argued main_folder

    main_folder - str
        path to main folder
    """
    folders = []
    for d, sub_ds, files in os.walk(main_folder):
        for sub_d in sub_ds:
            contents = os.listdir(os.path.join(d,sub_d))
            for content in contents:
                if ".pt" in content:
                    folders.append(sub_d)
                    break
    return sorted(folders, key=lambda x: x.split("/")[-1].split("_")[1])

def get_analysis_table(folder, hyps=None):
    """
    Returns a dict that can easily be converted into a dataframe
    """
    table = dict()
    if hyps is None:
        hyps = get_hyps(folder)
    for k,v in hyps.items():
        if "state_dict" not in k and "model_hyps" not in k:
            table[k] = [v]
    return table

def test_model(model, hyps):
    data = tdrdatas.loadexpt(expt=hyps['dataset'], cells=hyps['cells'], 
                                            filename=hyps['stim_type'],
                                            train_or_test="test",
                                            history=model.img_shape[0],
                                            norm_stats=hyps['norm_stats'])
    with torch.no_grad():
        response = tdrutils.inspect(model, data.X, batch_size=1000, to_numpy=False)['outputs']
        if hyps['lossfxn'] == "PoissonNLLLoss":
            lossfxn = globals()[hyps['lossfxn']](log_input=hyps['log_poisson'])
        else:
            lossfxn = globals()[hyps['lossfxn']]()
        loss = lossfxn(response, torch.FloatTensor(data.y)).item()
    resp = response.data.numpy()
    pearson = scipy.stats.pearsonr
    cors = [pearson(resp[:,i], data.y[:,i])[0] for i in range(resp.shape[1])]
    cor = np.mean(cors)
    return loss, cor

def get_intrneuron_rfs(stims, mem_pots, filt_len=40,verbose=False):
    """
    stims: dict
        keys: string cell_file
        vals: dict
            keys: string stim_key
            vals: ndarray (T,H,W)
                the corresponding stimulus for each group of cells
    mem_pots: dict
        keys: string cell_file
        vals: dict
            keys: string stim_key
            vals: ndarray (N,T)
                the membrane potentials for each cell. N is number of cells. The
                T dimension contains the potential response
    filt_len: int
        the length of the filter

    Returns:
        rfs: dict
            keys: str cell_file
            vals: dict
                keys: str stim_type
                vals: ndarray (N,C,H,W)
                    the stas for each cell within the cell file
    """
    rfs = dict()
    keys = list(mem_pots.keys())
    if verbose:
        print("Calculating Interneuron Receptive Fields")
        keys = tqdm(keys)
    for cell_file in keys:
        rfs[cell_file] = dict()
        for stim_key in mem_pots[cell_file].keys():
            rfs[cell_file][stim_key] = []
            mem_pot_arr = mem_pots[cell_file][stim_key]
            stim = stims[cell_file][stim_key]
            stim = scipy.stats.zscore(stim)
            prepped_stim = tdrstim.rolling_window(stim, filt_len)
            for ci,mem_pot in enumerate(mem_pots[cell_file][stim_key]):
                # Get Cell Receptive Field
                zscored_mem_pot = scipy.stats.zscore(mem_pot).squeeze()
                sta = tdrutils.revcor(prepped_stim, zscored_mem_pot, to_numpy=True)
                rfs[cell_file][stim_key].append(sta)
        rfs[cell_file][stim_key] = np.asarray(rfs[cell_file][stim_key])
    return rfs

def sample_model_rfs(model, layers=['sequential.0','sequential.6'], verbose=False):
    """
    Returns a receptive field of the central unit of each channel in each layer of the model.

    model: torch Module
    layers: list of str
        names of desired layers to sample from. The layers must be in layers 1 or 2 of the model.
    """
    table = {
        "layer":[],
        "chan":[],
        "row":[],
        "col":[]
    }

    layer_names = []
    prev_i = 0

    # Determine what layers exist in the model
    for i,(name,modu) in enumerate(model.named_modules()):
        if isinstance(modu,nn.ReLU):
            l_names = {'sequential.'+str(l) for l in range(prev_i,i)}
            layer_names.append(l_names)

    # Loop to create data frame
    for layer in layers:
        # Determines what layer the layer name falls under
        for chan_idx,l_names in enumerate(layer_names):
            if layer in l_names:
                break
        n_chans = model.chans[chan_idx]
        shape = model.shapes[chan_idx]
        row = shape[0]//2
        col = shape[1]//2
        for chan in range(n_chans):
            table['layer'].append(layer)
            table['chan'].append(chan)
            table['row'].append(row)
            table['col'].append(col)
    df = pd.DataFrame(table)

    rfs = get_model_rfs(model, df, verbose=verbose)
    return rfs

def get_model_rfs(model, data_frame, verbose=False):
    """
    Searches through each entry in the data frame and computes an STA for the model unit
    in that entry. Returns a dict containing the STAs.

    model: torch nn.Module
    data_frame: pandas DataFrame
        Necessary Columns: 
            'layer',
            'chan',
            'row',
            'col'

    Returns:
        rfs: dict
            keys: tuple (layer,chan)
                layer is a str and chan is an int
            vals: ndarray (C,H,W)
                the sta of the model unit
    """
    rfs = dict()
    layer1_dups = set() # Used to prevent duplicate calculations
    layer1_names = {'sequential.'+str(i) for i in range(5)}
    rng = range(len(data_frame))
    if verbose:
        print("Calculating Model Receptive Fields")
        rng = tqdm(rng)
    for i in rng:
        layer, chan, row, col = data_frame.loc[:,['layer','chan','row','col']].iloc[i]
        cell_idx = (chan,row,col)
        unit_id = (layer,chan,row,col)
        if unit_id in rfs or (layer in layer1_names and (layer,chan) in layer1_dups):
            continue
        elif layer in layer1_names:
            layer1_dups.add((layer,chan))
        chans = model.chans
        shapes = model.shapes
        layer_shape = (chans[0],*shapes[0]) if layer in layer1_names else (chans[1],*shapes[1])
        sta = compute_sta(model, layer=layer, cell_index=cell_idx, layer_shape=layer_shape,
                                                                       n_samples=10000,
                                                                       contrast=1,
                                                                       to_numpy=True,
                                                                       verbose=False)

        rfs[unit_id] = sta
    return rfs

def get_model2model_cors(model1, model2, model1_layers={"sequential.0", "sequential.6"},
                                          model2_layers={"sequential.0", "sequential.6"},
                                          contrast=1, n_samples=5000, use_ig=False,
                                          ret_model1_rfs=False, ret_model2_rfs=False,
                                          row_stride=1, col_stride=1, only_max=False,
                                          verbose=True):
    """
    Gets and returns a DataFrame of the best activation correlations between the two models.

    model1 - torch Module
    model2 - torch Module
    layers - set of str
        names of layers to be correlated with interneurons
    ret_model_rfs - bool
        the sta of the model are returned if this is true
    use_ig: bool
        if true, correlations are completed using the integrated gradient
    row_stride: int
        the number of rows to skip in model1 when doing correlations
    col_stride: int
        the number of cols to skip in model1 when doing correlations
    only_max: bool
        if true, returns only the maximum correlation calculations. If false, all
        correlation combinations are calculated.
    """
    model1.eval()
    model2.eval()
    table = tdrintr.model2model_cors(model1,model2, model1_layers=model1_layers, use_ig=use_ig,
                                                        model2_layers=model2_layers,
                                                        batch_size=500, contrast=contrast,
                                                        n_samples=n_samples, only_max=only_max,
                                                        row_stride=row_stride, 
                                                        col_stride=col_stride, verbose=verbose)
    df = pd.DataFrame(table)
    if ret_model1_rfs:
        print("Receptive Field Calculations not implemented yet...")
    if ret_model2_rfs:
        print("Receptive Field Calculations not implemented yet...")
    return df

def get_intr_cors(model, layers=['sequential.0', 'sequential.6'], stim_keys={"boxes"},
                                                             files=None,ret_real_rfs=False, 
                                                             ret_model_rfs=False,
                                                             verbose=True):
    """
    Gets and returns a DataFrame of the interneuron correlations with the model.

    model - torch Module
    layers - list of str
        names of layers to be correlated with interneurons
    stim_keys - set of str
        the stim types you would like to correlate with. 
        Options are generally boxes and lines
    files - list of str or None
        the names of the files you would like to use
    ret_real_rfs - bool
        the sta of the interneuron are returned if this is true
    ret_model_rfs - bool
        the sta of the most correlated model unit are returned if this is true
    """
    if verbose:
        print("Reading data for interneuron correlations...")
        print("Using stim keys:", ", ".join(list(stim_keys)))
    filt_len = model.img_shape[0]
    interneuron_data = tdrdatas.load_interneuron_data(root_path="~/interneuron_data",
                                                  filter_length=filt_len,files=files)
    stim_dict, mem_pot_dict, _ = interneuron_data
    if ret_real_rfs:
        real_rfs = get_intrneuron_rfs(stim_dict, mem_pot_dict, filt_len=model.img_shape[0],
                                                                           verbose=verbose)

    table = tdrintr.get_intr_cors(model, stim_dict, mem_pot_dict, layers=set(layers),
                                                       batch_size=500, verbose=verbose)
    df = pd.DataFrame(table)
    if ret_model_rfs:
        dups = ['cell_file', 'cell_idx']
        temp_df = df.sort_values(by='cor', ascending=False).drop_duplicates(dups)
        model_rfs = get_model_rfs(model, temp_df, verbose=verbose)
    if ret_real_rfs and ret_model_rfs:
        return df, real_rfs, model_rfs
    elif ret_real_rfs:
        return df, real_rfs
    elif ret_model_rfs:
        return df, model_rfs
    return df

def get_analysis_figs(folder, model, metrics=None, verbose=True):
    if metrics is not None:
        # Plot Loss Curves
        if "epoch" in metrics and 'loss' in metrics and 'val_loss' in metrics:
            fig = plt.figure()
            plt.plot(metrics['epoch'], metrics['loss'],color='k')
            plt.plot(metrics['epoch'], metrics['val_loss'],color='b')
            plt.legend(["train", "validation"])
            plt.title("Loss Curves")
            plt.savefig(os.path.join(folder,'loss_curves.png'))

        # Plot Acc Curves
        if 'epoch' in metrics and 'test_pearson' in metrics and 'val_acc' in metrics:
            fig = plt.figure()
            plt.plot(metrics['epoch'], metrics['test_pearson'],color='k')
            plt.plot(metrics['epoch'], metrics['val_acc'],color='b')
            plt.legend(["test", "validation"])
            plt.title("Correlation Accuracies")
            plt.savefig(os.path.join(folder,'acc_curves.png'))

    ## Get retinal phenomena plots
    figs, fig_names, metrics = retinal_phenomena_figs(model, verbose=verbose)

    for fig, name in zip(figs, fig_names):
        save_name = name + ".png"
        fig.savefig(os.path.join(folder, save_name))

def analyze_model(folder, make_figs=True, make_model_rfs=False, verbose=True):
    """
    Calculates model performance on the testset and calculates interneuron correlations.

    folder: str
        the folder full of checkpoints
    make_figs: bool
    make_model_rfs: bool
        returns a dict of model receptive fields if set to true. Can be used with 
        plot_model_rfs from the visualizations package
    """
    hyps = get_hyps(folder)
    table = get_analysis_table(folder, hyps=hyps)

    model,metrics = read_model(folder,ret_metrics=True)
    hyps['norm_stats'] = metrics['norm_stats']
    model.eval()
    model.to(DEVICE)
    if make_figs:
        if verbose:
            print("Making figures")
        get_analysis_figs(folder, model, metrics, verbose=verbose)

    gc_loss, gc_cor = test_model(model, hyps)
    table['test_acc'] = [gc_cor]
    table['test_loss'] = [gc_loss]
    df = pd.DataFrame(table)
    if verbose:
        print("GC Cor:", gc_cor,"  Loss:", gc_loss)

    layers = ["sequential.2", 'sequential.8']
    intr_df = get_intr_cors(model, layers=layers, verbose=verbose)
    intr_df['save_folder'] = folder
    if make_model_rfs:
        rfs = sample_model_rfs(model, layers=layers, verbose=verbose)
        save_name = os.path.join(folder, "model_rf")
        tdrvis.plot_model_rfs(rfs, save_name)

    # Drop duplicates and average over celll type
    bests = intr_df.sort_values(by='cor',ascending=False)
    bests = bests.drop_duplicates(['cell_file', 'cell_idx'])
    bip_intr_cor = bests.loc[bests['cell_type']=="bipolar",'cor'].mean()
    df['bipolar_intr_cor'] = bip_intr_cor
    amc_intr_cor = bests.loc[bests['cell_type']=="amacrine",'cor'].mean()
    df['amacrine_intr_cor'] = amc_intr_cor
    hor_intr_cor = bests.loc[bests['cell_type']=="horizontal",'cor'].mean()
    df['horizontal_intr_cor'] = hor_intr_cor
    df['intr_cor'] = bests['cor'].mean()

    return df, intr_df

def analysis_pipeline(main_folder, make_figs=True, make_model_rfs=True, verbose=True):
    """
    Evaluates model on test set, calculates interneuron correlations, 
    and creates figures.

    main_folder: str
        the folder full of model folders that contain checkpoints
    """
    model_folders = get_model_folders(main_folder)
    csvs = ['model_data.csv', 'intr_data.csv']
    dfs = dict()
    for csv in csvs:
        csv_path = os.path.join(main_folder,csv)
        if os.path.exists(csv_path):
            dfs[csv] = pd.read_csv(csv_path, sep="!")
        else:
            dfs[csv] = {"empty":True}
    for folder in model_folders:
        save_folder = os.path.join(main_folder, folder)
        if "save_folder" in dfs[csvs[0]] and save_folder in set(dfs[csvs[0]]['save_folder']):
            intr_save_folders = set(dfs[csvs[1]]['save_folder'])
            if "save_folder" in dfs[csvs[1]] and save_folder in intr_save_folders:
                if verbose:
                    print("Skipping",folder," due to previous record")
                continue
        if verbose:
            print("\n\nAnalyzing", folder)
        
        df, intr_df = analyze_model(save_folder, make_figs=make_figs, 
                                        make_model_rfs=make_model_rfs, 
                                        verbose=verbose)
        if not("save_folder" in dfs[csvs[0]] and folder in set(dfs[csvs[0]]['save_folder'])):
            if 'empty' in dfs[csvs[0]]:
                dfs[csvs[0]] = df
            else:
                dfs[csvs[0]] = dfs[csvs[0]].append(df, sort=True)
        if not("save_folder" in dfs[csvs[1]] and folder in set(dfs[csvs[1]]['save_folder'])):
            if 'empty' in dfs[csvs[1]]:
                dfs[csvs[1]] = intr_df
            else:
                dfs[csvs[1]] = dfs[csvs[1]].append(intr_df,sort=True)
    return dfs










