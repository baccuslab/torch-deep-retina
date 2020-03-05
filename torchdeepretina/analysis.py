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
import torchdeepretina.custom_modules as custmods
import torchdeepretina.intracellular as tdrintr
import torchdeepretina.io as tdrio
from torchdeepretina.retinal_phenomena import retinal_phenomena_figs
import torchdeepretina.stimuli as tdrstim
import torchdeepretina.pruning as pruning
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

def std_consistency(model_files, ret_all=False, verbose=True):
    """
    This function serves as a way to determine the consistency
    of a number of models. It does this by feeding in natural
    scenes and whitenoise to each model and determining the avg
    standard deviation of the outputs.

    model_files: list of strings
        a list of model files to be loaded and compared
    ret_all: bool
        if true, returns a tuple of the naturalscene and whitenoise 
        stds in addition to the avg of the two.
    """
    nat_scenes = tdrdatas.loadexpt(expt="15-10-07", cells="all", 
                                            filename="naturalscene",
                                            train_or_test="test",
                                            history=None)
    nat_scenes = nat_scenes.X
    whitenoise = np.random.randn(*nat_scenes.shape)
    s = np.s_[len(whitenoise)//3:2*len(whitenoise)//3]
    whitenoise[s] = 2*whitenoise[s]
    s = np.s_[2*len(whitenoise)//3:]
    whitenoise[s] = 3*whitenoise[s]

    natputs = []
    whitputs = []
    n_units = None
    if verbose:
        print("Collecting model responses")
    for model_file in model_files:
        model = tdrio.load_model(model_file)
        if n_units is None:
            n_units = model.n_units
        else:
            assert model.n_units == n_units,\
                        "models must have same output shape"
        model.to(DEVICE)
        model.eval()
        temp = tdrstim.rolling_window(nat_scenes, model.img_shape[0])
        natput = tdrutils.inspect(model, temp, verbose=verbose)
        natputs.append(natput['outputs'])

        temp = tdrstim.rolling_window(whitenoise, model.img_shape[0])
        whitput = tdrutils.inspect(model, temp, verbose=verbose)
        whitputs.append(whitput['outputs'])
    nat_std = np.asarray(natputs).std(0).mean()/nat_scenes.std()
    white_std = np.asarray(whitputs).std(0).mean()/whitenoise.std()
    avg_std = (nat_std+white_std)/2
    if verbose:
        print("Naturalscene STD:", nat_std)
        print("Whitenoise STD:", white_std)
        print("Total STD:", avg_std)
    if ret_all:
        return avg_std, nat_std, white_std
    return avg_std

def get_metrics(folder):
    """
    Returns the recorded training history of the model training
    (i.e. val loss, val acc, etc)

    folder - str
        path to folder that contains model checkpoints
    """
    metrics = dict()
    folder = folder.strip()
    folder = os.path.expanduser(folder)
    assert os.path.isdir(folder.strip())
    checkpts = tdrio.get_checkpoints(folder)
    metric_set = {'loss', 'epoch', 'val_loss', 'val_acc',
                            'exp_val_acc', 'test_pearson'}
    for f in checkpts:
        chkpt = tdrio.load_checkpoint(f)
        for k,v in chkpt.items():
            if k in metric_set:
                if k not in metrics:
                    metrics[k] = [v]
                else:
                    metrics[k].append(v)
    metrics['norm_stats'] = [chkpt['norm_stats']['mean'],
                                    chkpt['norm_stats']['std']]
    return metrics

def compute_sta(*args, **kwargs):
    """
    Computes the STA using the average of instantaneous receptive 
    fields (gradient of output with respect to input). Differs from
    `get_sta` in that is uses the average gradient rather than the
    correlation matrix.
    """
    return tdrutils.compute_sta(*args, **kwargs)

def get_sta(*args, **kwargs):
    """
    Computes the sta using reverse correlation. Uses the central unit
    for computation. Differs from compute_sta in that it uses
    correlation to calculate the sta rather than the average grad.

    model - torch Module

    returns:
        dict of sta lists for each channel in each layer
        keys: layer names
            vals: lists of stas for each channel in the layer
    """
    return tdrutils.revcor_sta(*args, **kwargs)

def get_analysis_table(folder=None, hyps=None):
    """
    Returns a dict that can easily be converted into a dataframe. The
    dict includes a key for each of the hyperparams.

    folder: str or None
        path to the model folder. if None, hyps must be not None.
    hyps: dict or None
        dict of hyperparameters. if None, folder must be not None
    """
    table = dict()
    assert hyps is not None or folder is not None,\
                            "either folder or hyps must not be None"
    if hyps is None:
        hyps = tdrio.get_hyps(folder)
    for k,v in hyps.items():
        if "state_dict" not in k and "model_hyps" not in k:
            table[k] = [v]
    return table

def test_model(model, hyps):
    """
    Runs the test data through the model and determines the loss and
    correlation with the truth.

    model: torch Module
    hyps: dict
        keys: str
            hyperparameter names
        vals: values corresponding to each hyperparameter key
    """
    data = tdrdatas.load_test_data(hyps)
    with torch.no_grad():
        response = tdrutils.inspect(model, data.X, batch_size=1000,
                                         to_numpy=False)['outputs']
        if hyps['lossfxn'] == "PoissonNLLLoss":
            lp = hyps['log_poisson']
            lossfxn = globals()[hyps['lossfxn']](log_input=lp)
        else:
            lossfxn = globals()[hyps['lossfxn']]()
        loss = lossfxn(response, torch.FloatTensor(data.y)).item()
    resp = response.data.numpy()
    pearson = scipy.stats.pearsonr
    cors = [pearson(resp[:,i], data.y[:,i])[0] for i in\
                                    range(resp.shape[1])]
    cor = np.mean(cors)
    return loss, cor

def get_interneuron_rfs(stims, mem_pots, filt_len=40,verbose=False):
    """
    Calculates the receptive fields of the recorded interneurons.

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
                the membrane potentials for each cell. N is number of
                cells. The T dimension contains the potential response
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
            mps = mem_pots[cell_file][stim_key]
            for ci,mem_pot in enumerate(mps):
                # Get Cell Receptive Field
                zscored_mem_pot =scipy.stats.zscore(mem_pot).squeeze()
                sta = tdrutils.revcor(prepped_stim, zscored_mem_pot,
                                                      to_numpy=True)
                rfs[cell_file][stim_key].append(sta)
        rfs[cell_file][stim_key] =np.asarray(rfs[cell_file][stim_key])
    return rfs

def sample_model_rfs(model, layers=[], use_grad=True, verbose=False):
    """
    Returns a receptive field of the central unit of each channel
    in each layer of the model.

    model: torch Module
    layers: list of ints or strs
        indexes or names of desired layers to sample from.
    use_grad: bool
        if true, rfs are calculated by averaging over a number of
        instantaneous receptive fields

    returns:
        rfs: dict
            keys: tuple (layer,chan)
                layer is a str and chan is an int
            vals: ndarray (C,H,W)
                the sta of the model unit
    """
    table = {
        "layer":[],
        "chan":[],
        "row":[],
        "col":[]
    }

    if len(layers) == 0:
        layers = tdrutils.get_conv_layer_names(model)
    if isinstance(layers[0],int):
        layer_names = tdrutils.get_conv_layer_names(model)
        layers = [layer_names[i] for i in layers]
    prev_i = 0

    # Loop to create data frame containing each desired unit
    for layer in layers:
        chan_idx = tdrutils.get_layer_idx(model, layer=layer)
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

    rfs = get_model_rfs(model, df, use_grad=use_grad, verbose=verbose)
    return rfs

def get_model_rfs(model, data_frame, contrast=1, use_grad=False,
                                                  verbose=False):
    """
    Searches through each entry in the data frame and computes an STA
    for the model unit in that entry. Returns a dict containing the
    STAs.

    model: torch nn.Module
    data_frame: pandas DataFrame
        Necessary Columns: 
            'layer',
            'chan',
            'row',
            'col'
    contrast: float
        the standard deviation of the whitenoise used for calculating
        the sta
    use_grad: bool
        determines method by which to calculate receptive fields

    Returns:
        rfs: dict
            keys: tuple (layer,chan)
                layer is a str and chan is an int
            vals: ndarray (C,H,W)
                the sta of the model unit
    """
    rfs = dict()
    rf_dups = set() # Used to prevent duplicate calculations
    keys = ['layer','chan','row','col']

    rng = range(len(data_frame))
    if verbose:
        print("Calculating Model Receptive Fields")
        rng = tqdm(rng)
    for i in rng:
        layer, chan, row, col = data_frame.loc[:,keys].iloc[i]
        layer_idx = tdrutils.get_layer_idx(model, layer)
        cell_idx = (chan,row,col)
        unit_id = (layer,chan,row,col)
        if (layer,chan) in rf_dups:
            continue
        rf_dups.add((layer,chan))
        chans = model.chans
        shapes = model.shapes
        layer_shape = (chans[layer_idx],*shapes[layer_idx])
        if use_grad:
            sta = compute_sta(model, layer=layer, cell_index=cell_idx,
                                              layer_shape=layer_shape,
                                              n_samples=10000,
                                              contrast=contrast,
                                              to_numpy=True,
                                              verbose=False)
        else:
            sta = get_sta(model, layer=layer, cell_index=cell_idx,
                                          layer_shape=layer_shape,
                                          n_samples=15000,
                                          contrast=contrast,
                                          to_numpy=True,
                                          verbose=False)

        rfs[unit_id] = sta
    return rfs

def get_model2model_cors(model1, model2, model1_layers=[],
                                         model2_layers=[],
                                         stim=None,
                                         response1=None,
                                         response2=None,
                                         contrast=1, n_samples=5000,
                                         use_ig=False,
                                         ret_model1_rfs=False,
                                         ret_model2_rfs=False,
                                         verbose=True):
    """
    Gets and returns a DataFrame of the best activation correlations
    between the two models.

    model1 - torch Module
    model2 - torch Module
    stim - ndarray (T,H,W) or None
        optional argument to specify the stimulus. If none, stimulus
        defaults to whitenoise
    layers - set of str
        names of layers to be correlated with interneurons
    ret_model_rfs - bool
        the sta of the model are returned if this is true
    use_ig: bool
        if true, correlations are completed using the integrated
        gradient
    only_max: bool
        if true, returns only the maximum correlation calculations. If
        false, all correlation combinations are calculated.

    Returns:
        cors: dict
            keys: str
                the model1 layer names
            vals: float
                the maximum correlation between a single layer in
                model1 and all the layers in model2
    """
    model1.eval()
    model2.eval()
    if len(model1_layers) == 0:
        model1_layers = tdrutils.get_conv_layer_names(model1)[:-1]
    if len(model2_layers) == 0:
        model2_layers = tdrutils.get_conv_layer_names(model2)[:-1]
    if verbose:
        print("Correlating Model Layers")
        print("Model1:", model1_layers)
        print("Model2:", model2_layers)
    with torch.no_grad():
        intr_cors = tdrintr.model2model_cor_mtxs(model1,model2,
                                        model1_layers=model1_layers,
                                        model2_layers=model2_layers,
                                        stim=stim,
                                        response1=response1,
                                        response2=response2,
                                        use_ig=use_ig,
                                        batch_size=500,
                                        contrast=contrast,
                                        n_samples=n_samples,
                                        verbose=verbose)
    cors = dict()
    for l1 in model1_layers:
        all_cors = []
        for l2 in model2_layers:
            all_cors.append(intr_cors[l1][l2])
        layer = np.concatenate(all_cors, axis=-1)
        cors[l1] = np.max(layer, axis=-1).mean()
    return cors

def get_model2model_cca(model1, model2, model1_layers=[],
                                        model2_layers=[],
                                        n_samples=10000,
                                        stim=None,
                                        response1=None,
                                        response2=None,
                                        n_components=2,
                                        alpha=1,
                                        use_ig=False,
                                        np_cca=False,
                                        combine_layers=True,
                                        verbose=True):
    """
    Gets and returns a DataFrame of the canonical correlation
    between the activations or integrated gradient of the two models.

    model1: torch Module
    model2: torch Module
    stim: ndarray or torch FloatTensor (T,H,W)
    layers: set of str
        names of layers to be correlated with interneurons
    n_components: int
        number of components to use in decomposition
    alpha: float
        cca regularization term
    use_ig: bool
        if true, correlations are completed using the integrated
        gradient
    np_cca: bool
        if true, uses numpy version of cca

    Returns:
        ccor: float
            average canonical correlation for all components
    """
    if stim is None:
        nx = model1.img_shape[1]
        stim = tdrstim.repeat_white(n_samples, nx=nx,
                                        contrast=1,
                                        n_repeats=3,
                                        rand_spat=True)
    if len(stim.shape) >= 4:
        print("Stimulus has wrong dimensions. Removing dim 1")
        stim = stim[:,0]

    with torch.no_grad():
        model1.to(DEVICE)
        model2.cpu()
        if response1 is None:
            if len(model1_layers) == 0:
                model1_layers=tdrutils.get_conv_layer_names(model1)
                model1_layers = model1_layers[:-1]
            response1 = tdrintr.get_response(model1, stim,
                                            model1_layers,
                                            batch_size=1000,
                                            use_ig=use_ig,
                                            verbose=verbose)
        else:
            model1_layers = list(response1.keys())
        model2.to(DEVICE)
        model1 = model1.cpu()
        if response2 is None:
            if len(model2_layers) == 0:
                model2_layers=tdrutils.get_conv_layer_names(model2)
                model2_layers = model2_layers[:-1]
            response2 = tdrintr.get_response(model2, stim,
                                            model2_layers,
                                            batch_size=1000,
                                            use_ig=use_ig,
                                            verbose=verbose)
        else:
            model2_layers = list(response2.keys())
        model2 = model2.cpu()
    keys = list(response1.keys())
    resp1 = []
    resp2 = []
    for key in keys:
        response1[key] =response1[key].reshape(len(response1[key]),-1)
        response1[key] = torch.FloatTensor(response1[key])
        response2[key] =response2[key].reshape(len(response2[key]),-1)
        resp = torch.FloatTensor(response2[key])
        resp2.append(resp)
    resp2 = torch.cat(resp2,dim=-1)
    if combine_layers:
        resp = [response1[k] for k in response1.keys()]
        resp = [r.reshape(len(r),-1) for r in resp]
        response1 = {"all": torch.cat(resp, dim=-1)}
    if np_cca:
        resp2 = resp2.cpu().numpy()
    ccors = {k:None for k in response1.keys()}
    for key in response1.keys():
        if np_cca:
            response1[key] = response1[key].cpu().numpy()
            ccor = tdrutils.np_cca(response1[key], resp2,
                                n_components=n_components,
                                alpha=alpha,
                                verbose=verbose)
        else:
            ccor = tdrutils.cca(response1[key], resp2,
                                n_components=n_components,
                                alpha=alpha,
                                to_numpy=True,
                                verbose=verbose)
        ccors[key] = np.mean(ccor)
    return ccors

def get_intr_cors(model, layers=['sequential.0', 'sequential.6'],
                                     stim_keys={"boxes",'lines'},
                                     files=None,
                                     ret_real_rfs=False,
                                     ret_model_rfs=False,
                                     slide_steps=0,
                                     verbose=True):
    """
    Gets and returns a DataFrame of the interneuron correlations with
    the model.

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
        the sta of the most correlated model unit are returned if
        this is true
    slide_steps - int
        slides the stimulus so that misaligned receptive fields of
        the ganglion cells and interneurons can be accounted for.
        This is the number of slides to try. Note that it operates
        in both the x and y dimension so the total number of attempts
        is equal to slide_steps squared.
    """
    if verbose:
        print("Reading data for interneuron correlations...")
        print("Using stim keys:", ", ".join(list(stim_keys)))
    filt_len = model.img_shape[0]
    rp = "~/interneuron_data"
    interneuron_data = tdrdatas.load_interneuron_data(root_path=rp,
                                            filter_length=filt_len,
                                            files=files,
                                            stim_keys=stim_keys,
                                            join_stims=True)
    stim_dict, mem_pot_dict, _ = interneuron_data
    # Using join_stims returns a dict with keys of the file names and
    # values of the ndarray stimulus and membrane potentials. In this
    # case it has joined the boxes and lines stimulus.
    # get_interneuron_rfs and get_intr_cors expect there to be an
    # additional dict listing the stimuli types. So here we provide
    # that structure and name the stimulus as 'test'
    for k in stim_dict.keys():
        stim_dict[k] = {"test":stim_dict[k]}
        mem_pot_dict[k] = {"test":mem_pot_dict[k]}

    if ret_real_rfs:
        real_rfs = get_interneuron_rfs(stim_dict, mem_pot_dict,
                                  filt_len=model.img_shape[0],
                                  verbose=verbose)

    verbose = slide_steps!=0 and verbose
    df = tdrintr.get_intr_cors(model, stim_dict, mem_pot_dict,
                                           layers=set(layers),
                                           batch_size=500,
                                           slide_steps=slide_steps,
                                           verbose=verbose)
    if ret_model_rfs:
        dups = ['cell_file', 'cell_idx']
        temp_df = df.sort_values(by='cor', ascending=False)
        temp_df = temp_df.drop_duplicates(dups)
        model_rfs = get_model_rfs(model, temp_df, verbose=verbose)
    if ret_real_rfs and ret_model_rfs:
        return df, real_rfs, model_rfs
    elif ret_real_rfs:
        return df, real_rfs
    elif ret_model_rfs:
        return df, model_rfs
    return df

def get_analysis_figs(folder, model, metrics=None, ret_phenom=True,
                                                     verbose=True):
    if metrics is not None:
        # Plot Loss Curves
        if "epoch" in metrics and 'loss' in metrics and\
                                            'val_loss' in metrics:
            fig = plt.figure()
            plt.plot(metrics['epoch'], metrics['loss'],color='k')
            plt.plot(metrics['epoch'], metrics['val_loss'],color='b')
            plt.legend(["train", "validation"])
            plt.title("Loss Curves")
            plt.savefig(os.path.join(folder,'loss_curves.png'))

        # Plot Acc Curves
        if 'epoch' in metrics and 'test_pearson' in metrics and\
                                            'val_acc' in metrics:
            fig = plt.figure()
            plt.plot(metrics['epoch'], metrics['test_pearson'],color='k')
            plt.plot(metrics['epoch'], metrics['val_acc'],color='b')
            plt.legend(["test", "validation"])
            plt.title("Correlation Accuracies")
            plt.savefig(os.path.join(folder,'acc_curves.png'))

    ## Get retinal phenomena plots
    if ret_phenom:
        figs, fig_names, metrics = retinal_phenomena_figs(model,
                                                verbose=verbose)

        for fig, name in zip(figs, fig_names):
            save_name = name + ".png"
            fig.savefig(os.path.join(folder, save_name))

def analyze_model(folder, make_figs=True, make_model_rfs=False,
                                                 slide_steps=0,
                                                 verbose=True):
    """
    Calculates model performance on the testset and calculates
    interneuron correlations.

    folder: str
        complete path to the model folder full of checkpoints
    make_figs: bool
    make_model_rfs: bool
        returns a dict of model receptive fields if set to true. Can
        be used with plot_model_rfs from the visualizations package
    slide_steps - int
        slides the interneuron stimulus so that misaligned rfs of the
        ganglion cells and interneurons can be accounted for. This is
        the number of slides to try. Note that it operates in both
        the x and y dimension so the total number of attempts is
        equal to slide_steps squared.
    """
    hyps = tdrio.get_hyps(folder)
    table = get_analysis_table(folder, hyps=hyps)
    if 'save_folder' in table:
        table['save_folder'] = folder

    model = tdrio.load_model(folder)
    metrics = get_metrics(folder)
    hyps['norm_stats'] = metrics['norm_stats']
    model.eval()
    model.to(DEVICE)
    # Pruning
    if hasattr(model, "zero_dict"):
        zero_dict = model.zero_dict
        pruning.zero_chans(model, zero_dict)
        keys = sorted(list(zero_dict.keys()))
        pruned_chans = []
        s = ""
        for pi, k in enumerate(keys):
            diff = model.chans[pi]-len(zero_dict[k])
            pruned_chans.append(diff)
            chans = [str(c) for c in zero_dict[k]]
            s += "\n{}: {}".format(k,",".join(chans))
        if verbose:
            print("Pruned Channels:"+s)
        if 'pruned_chans' not in table:
            table['pruned_chans'] = []
        table['pruned_chans'].append(pruned_chans) # Pruned chan count
    # Figs
    if make_figs:
        if verbose:
            print("Making figures")
        get_analysis_figs(folder, model, metrics, verbose=verbose)
    # GC Testing
    gc_loss, gc_cor = test_model(model, hyps)
    table['test_acc'] = [gc_cor]
    table['test_loss'] = [gc_loss]
    df = pd.DataFrame(table)
    if verbose:
        print("Test Cor:", gc_cor,"  Loss:", gc_loss)

    layers = []
    layers = tdrutils.get_conv_layer_names(model)
    layers = sorted(layers, key=lambda x: int(x.split(".")[-1]))
    if verbose:
        print("Calculating intrnrn correlations for:", layers)
    intr_df = get_intr_cors(model, layers=layers,
                                    slide_steps=slide_steps,
                                    verbose=verbose)
    intr_df['save_folder'] = folder
    if make_model_rfs:
        rfs = sample_model_rfs(model, layers=layers, verbose=verbose)
        save_name = os.path.join(folder, "model_rf")
        tdrvis.plot_model_rfs(rfs, save_name)

    # Drop duplicates and average over celll type
    bests = intr_df.sort_values(by='cor',ascending=False)
    bests = bests.drop_duplicates(['cell_file', 'cell_idx'])
    loc = bests['cell_type']=="bipolar"
    bip_intr_cor = bests.loc[loc,'cor'].mean()
    df['bipolar_intr_cor'] = bip_intr_cor
    loc = bests['cell_type']=="amacrine"
    amc_intr_cor = bests.loc[loc,'cor'].mean()
    df['amacrine_intr_cor'] = amc_intr_cor
    loc = bests['cell_type']=="horizontal"
    hor_intr_cor = bests.loc[loc,'cor'].mean()
    df['horizontal_intr_cor'] = hor_intr_cor
    df['intr_cor'] = bests['cor'].mean()
    if verbose:
        print("Interneuron Correlations:")
        print("Bipolar Avg:", bip_intr_cor)
        print("Amacrine Avg:", amc_intr_cor)
        print("Horizontal Avg:", hor_intr_cor)

    return df, intr_df

def evaluate_ln(ln, hyps):
    """
    ln: RevCorLN object (see models.py)
    """
    table = get_analysis_table(hyps=hyps)
    data = tdrdatas.loadexpt(expt=hyps['dataset'],cells=hyps['cells'],
                                           filename=hyps['stim_type'],
                                           train_or_test="test",
                                           history=model.img_shape[0],
                                           norm_stats=ln.norm_stats)
    with torch.no_grad():
        response = model(data.X)
    resp = response.data.numpy()
    cor = scipy.stats.pearsonr(response, data.y[:,model.cell_idx])
    table['test_acc'] = [gc_cor]
    df = pd.DataFrame(table)
    if verbose:
        print("Test Cor:", gc_cor,"  Loss:", gc_loss)
    return df

def analysis_pipeline(main_folder, make_figs=True,make_model_rfs=True,
                                                        save_dfs=True,
                                                        slide_steps=0,
                                                        verbose=True):
    """
    Evaluates model on test set, calculates interneuron correlations,
    and creates figures.

    main_folder: str
        the folder full of model folders that contain checkpoints
    make_figs: bool
        automatically creates and saves figures in the model folders
    make_model_rfs: bool
        automatically creates and saves model receptive field figures
        in the model folders
    save_dfs: bool
        automatically saves the analysis dataframe checkpoints in the
        main folder
    slide_steps - int
        slides the interneuron stimulus so that misaligned rfs of the
        ganglion cells and interneurons can be accounted for. This is
        the number of slides to try. Note that it operates in both
        the x and y dimension so the total number of attempts is
        equal to slide_steps squared.
    """
    model_folders = tdrio.get_model_folders(main_folder)
    csvs = ['model_data.csv', 'intr_data.csv']
    dfs = dict()
    save_folders = dict()
    for csv in csvs:
        csv_path = os.path.join(main_folder,csv)
        if os.path.exists(csv_path):
            dfs[csv] = pd.read_csv(csv_path, sep="!")
            save_folders[csv] = set(dfs[csv]['save_folder'])
        else:
            dfs[csv] = {"empty":True}
            save_folders[csv] = set()
    for folder in model_folders:
        save_folder = os.path.join(main_folder, folder)
        if save_folder in save_folders[csvs[0]] and\
                                save_folder in save_folders[csvs[1]]:
            if verbose:
                print("Skipping",folder," due to previous record")
            continue
        if verbose:
            print("\n\nAnalyzing", folder)
        
        anal_dfs = analyze_model(save_folder, make_figs=make_figs,
                                        make_model_rfs=make_model_rfs,
                                        slide_steps=slide_steps,
                                        verbose=verbose)
        for i,csv in enumerate(csvs):
            if save_folder not in save_folders[csv]:
                if 'empty' in dfs[csv]:
                    dfs[csv] = anal_dfs[i]
                else:
                    dfs[csv] = dfs[csv].append(anal_dfs[i],sort=True)

        if save_dfs:
            for i,csv in enumerate(csvs):
                path = os.path.join(main_folder,csv)
                if i==1 and os.path.exists(path):
                    temp = pd.read_csv(path,sep="!",nrows=10)
                    dfs[csv][temp.columns].to_csv(path, sep="!",
                                                index=False,
                                                header=False,
                                                mode='a')
                    dfs[csv] = dfs[csv].iloc[:0]
                else:
                    dfs[csv].to_csv(path, sep="!", index=False,
                                                 header=True,
                                                 mode='w')
    return dfs

def get_resps(model, stim, model_layers, batch_size=1000,
                                            to_numpy=True):
    """
    Helper function to collect responses and dry code.

    stim: ndarray (T,H,W)
    model_layers: set of str

    Returns:
        tuple of responses, one raw and one integrated gradient.
        each response has the following form:
        response: dict
            keys: str
                layer names
            vals: FloatTensor
                the activations for the layer
    """
    with torch.no_grad():
        act_resp = tdrintr.get_response(model, stim,
                                        model_layers,
                                        batch_size=batch_size,
                                        use_ig=False,
                                        to_numpy=to_numpy,
                                        verbose=False)
        ig_resp = tdrintr.get_response(model, stim,
                                        model_layers,
                                        batch_size=batch_size,
                                        use_ig=True,
                                        to_numpy=to_numpy,
                                        verbose=False)
    return act_resp, ig_resp

def similarity_pipeline(model_paths, n_samples=20000,
                        table_checkpts=True,
                        calc_cor=True,
                        calc_cca=True,
                        np_cca=False,
                        save_file="similarity.csv",
                        batch_size=1000,
                        verbose=True):
    """
    Calculates the similarity between each of the models, pairwise.
    Includes the maximum activation correlations, the maximum
    integrated gradient correlations, the activation cca, and the
    integrated gradient cca.

    model_paths: list of str
        a list of paths to the models that can be loaded using the io
        package.
    n_samples: int
        the number of samples to draw for correlation analyses
    table_checkpts: bool
        if true, dataframe is saved after every comparison
    calc_cor: bool
        if true, correlations are calculated
    calc_cca: bool
        if true, cca is calculated
    np_cca: bool
        if true and calc_cca is true, cca is calculated using numpy
        code
    save_file: str
        path to save the csv to
    batch_size: int
        size of batching when calculating activations

    Returns:
        df: pd DataFrame
            columns:
                model1: str
                model2: str
                cor_type: str
                    the name of the correlation type
                layer: str
                    the name of the layer in model1 used for the
                    correlation calculations. "all" means all layers
                cor: float
                    the actual measured correlation
    """
    table = {
        "model1":[],
        "model2":[],
        "cor_type":[],
        "layer":[],
        "cor":[],
    }
    main_df = pd.DataFrame(table)
    if os.path.exists(save_file):
        main_df = pd.read_csv(save_file, sep="!")
    torch.cuda.empty_cache()
    data = tdrdatas.loadexpt("15-10-07", "all", "naturalscene",
                                            'train', history=0)
    stim = data.X[:n_samples]
    for i in range(len(model_paths)):
        # Check if model has already been compared to all other models
        idx = (main_df['model1']==model_paths[i])
        prev_comps = set(main_df.loc[idx,'model2'])
        missing_comp = False
        for path in model_paths:
            if path != model_paths[i] and path not in prev_comps:
                missing_comp = True
                break
        if not missing_comp:
            print("Skipping", model_paths[i],
                    "due to previous records")
            continue
        if verbose:
            print("Beginning model:", model_paths[i],"| {}/{}".format(
                                                  i,len(model_paths)))
            print()

        ############### Prep
        model1 = tdrio.load_model(model_paths[i])
        model1.eval()
        model1.to(DEVICE)
        model1_layers = tdrutils.get_conv_layer_names(model1)[:-1]
        if verbose:
            print("Computing Responses")
        act_resp1, ig_resp1 = get_resps(model1, stim, model1_layers,
                                                batch_size=batch_size,
                                                to_numpy=True)
        model1 = model1.cpu()
        for j in range(len(model_paths)):
            if i==j: continue
            idx = (main_df['model1']==model_paths[i])
            if model_paths[j] in set(main_df.loc[idx,"model2"]):
                print("Skipping:", model_paths[j])
                continue
            if verbose:
                s = "Comparing: {} to {} | {} comparisons left"
                s = s.format(model_paths[i],model_paths[j],
                                        len(model_paths)-j)
                print(s)
            stats_string = ""

            torch.cuda.empty_cache()
            model2 = tdrio.load_model(model_paths[j])
            model2.eval()
            model2.to(DEVICE)
            model2_layers = tdrutils.get_conv_layer_names(model2)[:-1]
            if verbose:
                print("Computing Responses")
            with torch.no_grad():
                act_resp2, ig_resp2 = get_resps(model2, stim,
                                                model2_layers,
                                                batch_size=batch_size,
                                                to_numpy=True)
            model2 = model2.cpu()

            ################### Correlation
            if calc_cor:
                torch.cuda.empty_cache()
                # Activation Max Correlation
                if verbose:
                    print("Beginning Activation Correlations")
                act_cor_dict = get_model2model_cors(model1, model2,
                                                response1=act_resp1,
                                                response2=act_resp2,
                                                stim=stim,
                                                use_ig=False,
                                                verbose=verbose)
                stats_string += "\nActivation Correlations:\n"
                for layer in act_cor_dict.keys():
                    table['model1'].append(model_paths[i])
                    table['model2'].append(model_paths[j])
                    table['cor_type'].append("act_cor")
                    table['layer'].append(layer)
                    table['cor'].append(act_cor_dict[layer])
                    stats_string += "{}: {}\n".format(layer,
                                                    act_cor_dict[layer])

                torch.cuda.empty_cache()
                # Integrated Gradient Max Correlation
                if verbose:
                    print("Beginning Integrated Grdient Correlations")
                ig_cor_dict = get_model2model_cors(model1, model2,
                                                 response1=ig_resp1,
                                                 response2=ig_resp2,
                                                 stim=stim,
                                                 use_ig=True,
                                                 verbose=verbose)
                stats_string += "IG Correlations:\n"
                for layer in ig_cor_dict.keys():
                    table['model1'].append(model_paths[i])
                    table['model2'].append(model_paths[j])
                    table['cor_type'].append("ig_cor")
                    table['layer'].append(layer)
                    table['cor'].append(ig_cor_dict[layer])
                    stats_string += "{}: {}\n".format(layer,
                                                ig_cor_dict[layer])

            ################### CCA
            torch.cuda.empty_cache()
            if calc_cca:
                if verbose:
                    print("Beginning Activation CCA")
                act_cca = get_model2model_cca(model1, model2,
                                                response1=act_resp1,
                                                response2=act_resp2,
                                                stim=stim,
                                                use_ig=False,
                                                np_cca=np_cca,
                                                verbose=verbose)
                table['model1'].append(model_paths[i])
                table['model2'].append(model_paths[j])
                table['cor_type'].append("act_cca")
                table['layer'].append("all")
                table['cor'].append(act_cca)
                stats_string += "Activs CCA: {}\n".format(act_cca)

                torch.cuda.empty_cache()

                # Integrated Gradient CCA
                if verbose:
                    print("Beginning Integrated Gradient CCA")
                ig_cca = get_model2model_cca(model1, model2,
                                                response1=ig_resp1,
                                                response2=ig_resp2,
                                                stim=stim,
                                                use_ig=True,
                                                np_cca=np_cca,
                                                verbose=verbose)
                table['model1'].append(model_paths[i])
                table['model2'].append(model_paths[j])
                table['cor_type'].append("ig_cca")
                table['layer'].append("all")
                table['cor'].append(ig_cca)
                stats_string += "IG CCA: {}\n".format(ig_cca)
            if verbose:
                print(stats_string)
            if table_checkpts:
                df = pd.DataFrame(table)
                table = {k:[] for k in table.keys()}
                if len(main_df['cor']) == 0:
                    main_df = df
                else:
                    main_df = main_df.append(df, sort=True)
                main_df.to_csv(save_file, sep="!",
                                 header=True, index=False)
            gc.collect()
            max_mem_used = resource.getrusage(resource.RUSAGE_SELF)
            max_mem_used = max_mem_used.ru_maxrss/1024
            print("Memory Used: {:.2f} mb".format(max_mem_used))
            gpu_mem = tdrutils.get_gpu_mem()
            s = ["gpu{}: {}".format(k,v) for k,v in gpu_mem.items()]
            s = "\n".join(s)
            print(s)
    return main_df

def pathway_similarities(model1, model2, stim=None, layers=[],
                                                m1_chans=[],
                                                m2_chans=[],
                                                batch_size=500,
                                                n_samples=10000,
                                                sim_type="dot",
                                                same_model=False,
                                                save_file=None):
    """
    Calcs the pairwise similarity of the integrated gradient pathway
    between each channel of the output layer (each channel is compared
    once to each other channel). Cells are sampled from each channel
    at different locations for comparisons. One cell is held in a
    fixed spatial location while the other cell is sampled from a
    grid created in the window centered around the first cell with
    the specified striding.

    model: torch Module
    stim: ndarray or FloatTensor (T,C,H,W) or (T,H,W)
        optional stimulus argument. if None, whitenoise is used.
    layers: list of str
        optional argument specifying the model layers of interest.
        if None, defaults to all intermediary layers.
    channels: list of int
        the output channels of interest. each pairwise channel
        combination is compared including comparison with itself as a
        control. thus there are N choose 2 plus N comparisons
        where N is the number of argued channels.
    n_samples: int
        the number of samples to be used for the stimulus. Only
        applies if stim is None
    sim_type: str
        denotes the similarity metric to be used. available options
        are 'maximum', 'one2one', 'cca', 'np_cca', 'dot'

    Returns:
        pd DataFrame
            sim_type: str
            cell1: int
            row1: int
            col1: int
            cell2: int
            row2: int
            col2: int
            cor: float

    """
    table = {
        "sim_type": [],
        "cell1": [],
        "row1": [],
        "col1": [],
        "cell2": [],
        "row2": [],
        "col2": [],
        "cor": [],
        "gc_cor": []
    }
    if save_file is not None and os.path.exists(save_file):
        main_df = pd.read_csv(save_file, sep="!")
    else:
        main_df = pd.DataFrame(table)
    if m1_chans is None or len(m1_chans)==0:
        m1_chans = list(range(model1.n_units))
    if m2_chans is None or len(m2_chans)==0:
        m2_chans = list(range(model2.n_units))

    for i,cell1 in enumerate(m1_chans):
        if same_model: m2_chans = m1_chans[i:]
        sim, gc_cors = tdrintr.compare_cell_pathways(model1, model2,
                                            stim=stim,
                                            layers=layers,
                                            cell1=cell1,
                                            comp_cells=m2_chans,
                                            batch_size=batch_size,
                                            n_samples=n_samples,
                                            sim_type=sim_type)
        for cell2 in sim.keys():
            gc_cor = gc_cors[cell2]
            for coords1 in sim[cell2].keys():
                row1,col1 = coords1
                for coords2,cor in sim[cell2][coords1].items():
                    row2,col2 = coords2
                    table["sim_type"].append(sim_type)
                    table["cell1"].append(cell1)
                    table["row1"].append(row1)
                    table["col1"].append(col1)
                    table["cell2"].append(cell2)
                    table["row2"].append(row2)
                    table["col2"].append(col2)
                    table["cor"].append(cor)
                    table["gc_cor"].append(gc_cor)
                    if isinstance(save_file, str):
                        df = pd.DataFrame(table)
                        main_df = main_df.append(df, sort=True)
                        main_df.to_csv(save_file, sep="!",
                                            header=True, index=False)
                        table = {k:[] for k in table.keys()}
    if len(table['row1']) > 0:
        df = pd.DataFrame(table)
        main_df = main_df.append(df, sort=True)
    return main_df


