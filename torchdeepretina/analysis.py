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

def get_metrics(folder, ret_metrics=False):
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
        if ret_metrics:
            for k,v in data.items():
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
    data = tdrdatas.loadexpt(expt=hyps['dataset'],
                              cells=hyps['cells'],
                              filename=hyps['stim_type'],
                              train_or_test="test",
                              history=model.img_shape[0],
                              norm_stats=hyps['norm_stats'])
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

def get_intrneuron_rfs(stims, mem_pots, filt_len=40,verbose=False):
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

def sample_model_rfs(model, layers=[], verbose=False):
    """
    Returns a receptive field of the central unit of each channel
    in each layer of the model.

    model: torch Module
    layers: list of ints or strs
        indexes or names of desired layers to sample from.

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

    rfs = get_model_rfs(model, df, verbose=verbose)
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
                                         contrast=1, n_samples=5000,
                                         use_ig=False,
                                         ret_model1_rfs=False,
                                         ret_model2_rfs=False,
                                         row_stride=1, col_stride=1,
                                         only_max=False,
                                         verbose=True):
    """
    Gets and returns a DataFrame of the best activation correlations
    between the two models.

    model1 - torch Module
    model2 - torch Module
    layers - set of str
        names of layers to be correlated with interneurons
    ret_model_rfs - bool
        the sta of the model are returned if this is true
    use_ig: bool
        if true, correlations are completed using the integrated
        gradient
    row_stride: int
        the number of rows to skip in model1 when doing correlations
    col_stride: int
        the number of cols to skip in model1 when doing correlations
    only_max: bool
        if true, returns only the maximum correlation calculations. If
        false, all correlation combinations are calculated.
    """
    model1.eval()
    model2.eval()
    if len(model1_layers) == 0:
        model1_layers = get_conv_layer_names(model1)
    if len(model2_layers) == 0:
        model2_layers = get_conv_layer_names(model2)
    if verbose:
        print("Correlating Model Layers")
        print("Model1:", model1_layers)
        print("Model2:", model2_layers)
    table = tdrintr.model2model_cors(model1,model2,
                                        model1_layers=model1_layers,
                                        model2_layers=model2_layers,
                                        use_ig=use_ig,
                                        batch_size=500,
                                        contrast=contrast,
                                        n_samples=n_samples,
                                        only_max=only_max,
                                        row_stride=row_stride,
                                        col_stride=col_stride,
                                        verbose=verbose)
    df = pd.DataFrame(table)
    if ret_model1_rfs:
        print("Receptive Field Calculations not implemented yet...")
    if ret_model2_rfs:
        print("Receptive Field Calculations not implemented yet...")
    return df

def get_intr_cors(model, layers=['sequential.0', 'sequential.6'], 
                                                 stim_keys={"boxes"},
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
                                            files=files)
    stim_dict, mem_pot_dict, _ = interneuron_data
    if ret_real_rfs:
        real_rfs = get_intrneuron_rfs(stim_dict, mem_pot_dict,
                                  filt_len=model.img_shape[0],
                                  verbose=verbose)

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
        if verbose:
            s = "Pruned Channels:"
            keys = sorted(list(zero_dict.keys()))
            for k in keys:
                chans = [str(c) for c in zero_dict[k]]
                s += "\n{}: {}".format(k,",".join(chans))
            print(s)
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
        print("GC Cor:", gc_cor,"  Loss:", gc_loss)

    layers = []
    bnorms = {nn.BatchNorm2d, nn.BatchNorm1d, custmods.AbsBatchNorm2d,
                                              custmods.AbsBatchNorm1d}
    for name,modu in model.named_modules():
        if len(name.split(".")) == 2 and type(modu) in bnorms:
            layers.append(name)
    layers = sorted(layers, key=lambda x: int(x.split(".")[-1]))
    layers = layers[:2]
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
        print("GC Cor:", gc_cor,"  Loss:", gc_loss)
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
                if os.path.exists(path):
                    temp = pd.read_csv(path,sep="!",nrows=10)
                    dfs[csv][temp.columns].to_csv(path, sep="!",
                                                index=False,
                                                header=False,
                                                mode='a')
                else:
                    dfs[csv].to_csv(path, sep="!", index=False,
                                                 header=True,
                                                 mode='w')
                dfs[csv] = dfs[csv].iloc[:0]
    return dfs










