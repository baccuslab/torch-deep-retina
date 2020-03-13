import matplotlib
from scipy.stats import sem, pearsonr
from tqdm import tqdm
import os
import re
import h5py
import collections
import pyret.filtertools as ft
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
import torchdeepretina.utils as tdrutils
import torchdeepretina.stimuli as tdrstim
import torchdeepretina.datas as tdrdatas
from torchdeepretina.custom_modules import GrabUnits
import torch.nn as nn
import pandas as pd

centers = {
      "bipolars_late_2012":  [(19, 22), (18, 20), (20, 21), (18, 19)],
      "bipolars_early_2012": [(17, 23), (17, 23), (18, 23)],
      "amacrines_early_2012":[(18, 23), (19, 24), (19, 23), (19, 23),
                                                            (18, 23)],
      "amacrines_late_2012":[ (20, 20), (22, 19), (20, 20), (19, 22),
                              (22, 25), (19, 21), (19, 17), (20, 19),
                              (17, 20), (19, 23), (17, 20), (17, 20),
                              (20, 19), (18, 18), (19, 17), (17, 17),
                              (18, 20), (20, 19), (17, 19), (19, 18),
                              (17, 17), (25, 15)
                            ]
}

if torch.cuda.is_available():
    DEVICE = torch.device('cuda:0')
else:
    DEVICE = torch.device('cpu')

STD_CUTOFF = -12
ABS_MEAN_CUTOFF = -15


def load_interneuron_data(*args, **kwargs):
    """ 
    Load data interneuron data

    num_pots: (number of potentials) stores the number of cells per
              stimulus
    mem_pots: (membrane potentials) stores the membrane potential
              psst, you can find the "data" folder in /home/grantsrb
              on deepretina server if you need

    returns:
    stims - dict
        keys are the cell files, vals are a list of nd array stimuli
        for each cell within the file
    mem_pots - dict
        keys are the cell files, vals are a list of nd array membrane
        potential responses for each cell within the file
    """
    return tdrdatas.load_interneuron_data(*args,**kwargs)

def make_intr_cor_maps(model, layers=['sequential.2','sequential.8'],
                                         data_len=None,verbose=True):
    """
    Collects the correlation maps for each layer in layers for
    all interneuron cells.

    model: torch nn Module
    layers: list of str
        desired layers to be analyzed

    Returns:
        cor_maps: dict
            keys: intrnrn files
            vals: lists of dicts
                The correlation map dictionaries returned from
                all_correlation_maps (see intracellular)
                keys: layer names str
                vals: ndarray (C,H,W)
                    The correlation maps. A unique correlation value
                    for each unit in the layer.
    """
    files = ['bipolars_late_2012.h5', 'bipolars_early_2012.h5', 
            'amacrines_early_2012.h5', 'amacrines_late_2012.h5']
    cor_maps = dict()
    for i,f in enumerate(files):
        cor_maps[f] = []
        s = "~/interneuron_data/"
        intr_data = load_interneuron_data(root_path=s,
                                     filter_length=model.img_shape[0],
                                     files=[f], stim_keys={'boxes'})
        stim_dict, mem_pot_dict, intrnrn_files = intr_data
        k = list(stim_dict.keys())[0] # Some variant of the file name
        stim = stim_dict[k]['boxes']
        stim = tdrstim.spatial_pad(stim, 50)
        stim = tdrstim.rolling_window(stim, model.img_shape[0])
        rnge = range(len(mem_pot_dict[k]['boxes']))
        if verbose:
            print("File: {}/{}".format(i,len(files)))
            rnge = tqdm(rnge)
        for ci in rnge:
            mem_pot = mem_pot_dict[k]['boxes'][ci]
            endx = data_len if data_len is not None and\
                            data_len < len(mem_pot) else\
                                             len(mem_pot)
            mem_pot = mem_pot[:endx]
            response = tdrutils.inspect(model, stim,
                                        insp_keys=set(layers),
                                        batch_size=500)
            for key in response.keys():
                response[key] = response[key][:endx]
            maps = all_correlation_maps(mem_pot, response,
                                        layer_keys=layers)
            cor_maps[f].append(maps)
    return cor_maps

def all_correlation_maps(mem_pot, model_response,
                           layer_keys=['sequential.2','sequential.8'],
                           verbose=False):
    """
    Returns a dict of correlation maps between the model and the 
    interneuron recordings for each argued model layer

    mem_pot: ndarray (T,)
    model_response: dict
        keys: str layer names
        vals: ndarrays (T,C) or (T,C,H,W)
            layer activations
    layer_keys: list of str
        the names of the model layers of interest

    Returns:
        cor_maps: dict
            keys: str layer names
            vals: ndarrays (C) or (C,H,W)
                correlations with mem_pot
    """
    cor_maps = dict()
    for layer in layer_keys:
        resp = model_response[layer]
        shape = resp.shape
        if len(shape) == 4: # (T,C,H,W) case
            chan_maps = []
            for chan in range(resp.shape[1]):
                activs = resp[:,chan]
                chan_map = correlation_map(mem_pot, activs)
                chan_maps.append(chan_map)
            cor_map = np.asarray(chan_maps) # (C, H, W)
            assert tuple(cor_map.shape) == tuple(shape[1:])
        else:
            assert len(shape)==2 # Must have shape of 2 or 4
            cors = []
            for i in range(resp.shape[1]):
                r,_ = pearsonr(mem_pot.squeeze(), resp[:,i])
                r = r if not np.isnan(r) and r > -1 else 0
                if np.log(np.abs(resp[:,i]).mean()) < ABS_MEAN_CUTOFF\
                              or np.log(resp[:,i].std()) < STD_CUTOFF:
                    if verbose and abs(r) > 0.1:
                        s = "Extremely small layer values, "+\
                                "pearson of {} can not be "+\
                                "trusted and is being set to 0"
                        print(s.format(r))
                    r = 0
                cors.append(r)
            cor_map = np.asarray(cors)
        cor_maps[layer] = cor_map
    return cor_maps

def correlation_map(mem_pot, activ_layer, verbose=False):
    '''
    Takes a 1d membrane potential and computes the correlation with
    every unit in activation layer.
    
    Args:
        mem_pot: 1-d numpy array
        activ_layer: ndarray or torch FloatTensor (T,H,W) or (T,C,H,W)
            layer of activations

    Returns:
        correlations: ndarray (space, space)
    '''
    shape = activ_layer.shape[1:]
    activs = activ_layer.reshape(len(activ_layer), -1)
    mem_pot = mem_pot.squeeze()[:,None]
    cor_mtx = tdrutils.mtx_cor(mem_pot, activs, to_numpy=True)
    cor_mtx = cor_mtx.reshape(*shape)
    if len(cor_mtx.shape) == 1:
        cor_mtx = cor_mtx[None]
    return cor_mtx

def max_correlation(mem_pot, model_layer, abs_val=False, verbose=False):
    '''
    Takes a 1d membrane potential and computes the correlation with
    every unit in model_layer. Do not use abs_val for publishable
    analysis.

    Args:
        mem_pot: 1-d numpy array, membrane potential
        model_layer: ndarray or torch FloatTensor (T, C, H, W)
            layer of activations
        abs_val: take absolute value of correlation
    '''
    if len(model_layer.shape) == 2:
        shape = (len(model_layer), 1, model_layer.shape[1])
        model_layer = model_layer.reshape(shape)
    cor_maps = [correlation_map(mem_pot,model_layer[:,c]) 
                                for c in range(model_layer.shape[1])]
    if abs_val:
        cor_maps = [np.absolute(m) for m in cor_maps]
    return np.max([np.max(m) for m in cor_maps])

def argmax_correlation(mem_pot, model_layer, ret_max_cor=False,
                                   abs_val=False, verbos=False):
    '''
    Takes a 1d membrane potential and finds the activation with the
    highest correlation in the model layer.
    
    Args:
        mem_pot: nd array (T,)
        model_layer: (T, C, H, W) or (T, D) layer of activities
        ret_max_cor: returns value of max correlation in addition to idx
        abs_val: use absolute value of correlations
    Returns:
        best_idx: tuple (chan, row, col) or (unit,)
        max_r: float
    '''
    shape = model_layer.shape[1:]
    activs = model_layer.reshape(len(model_layer),-1)
    cors = tdrutils.revcor(activs, mem_pot.squeeze(),to_numpy=True)
    if abs_val:
        cors = np.abs(cors)
    argmax = np.argmax(cors)
    best_idx = np.unravel_index(argmax, shape)
    if ret_max_cor:
        return best_idx, cors[argmax]
    return best_idx

def get_response(model, stim, model_layers, batch_size=500,
                                              use_ig=False,
                                              cell_idx=None,
                                              to_numpy=False,
                                              no_grad=True,
                                              verbose=False):
    """
    Helper function to dry up code in model2model functions.

    See model2model functions for better descriptions of variables.

    model: torch nn Module
    stim: ndarray (T,H,W) or (T,C,H,W)
    model_layers: set of str
        model activation/ig layers to be collected
    use_ig: bool
        indicates if integrated gradient should be used
    cell_idx: int or tuple of ints (chan, row, col)
        ganglion cell(s) of interest. if None, uses all cells. If
        tuple, must argue chan, row, and column
    to_numpy: bool
        if true, all responses are converted to ndarrays
    no_grad: bool
        if true, no gradients are calculated. silent otherwise.

    Returns:
        response: dict
            keys: str
                layer names
            vals: FloatTensor
                the activations for the layer
    """
    if verbose:
        if use_ig:
            print("Collecting model integrated gradient")
        else:
            print("Collecting model response")
    if len(stim.shape) == 3:
        stim = tdrstim.spatial_pad(stim, H=model.img_shape[1],
                                         W=model.img_shape[2])
        stim = tdrstim.rolling_window(stim, model.img_shape[0])
    else:
        stim = tdrstim.spatial_pad(stim, H=model.img_shape[1],
                                         W=model.img_shape[2])
    prev_cuda = next(model.parameters()).is_cuda
    model.to(DEVICE)
    if use_ig:
        response = dict()
        gc_resps = None
        for layer in model_layers:
            if layer == "outputs":
                continue
            intg_grad, gc_resps = tdrutils.integrated_gradient(model,
                                                stim,
                                                batch_size=batch_size,
                                                layer=layer,
                                                gc_idx=cell_idx,
                                                to_numpy=to_numpy,
                                                verbose=verbose)
            response[layer] = intg_grad
        if "outputs" in model_layers:
            if gc_resps is None:
                bsize = batch_size
                temp = tdrutils.inspect(model, stim, batch_size=bsize,
                                                    insp_keys={},
                                                    to_numpy=to_numpy,
                                                    no_grad=no_grad,
                                                    verbose=verbose)
                gc_resps = temp['outputs']
            response['outputs'] = gc_resps
    else:
        response = tdrutils.inspect(model,stim,batch_size=batch_size,
                                               insp_keys=model_layers,
                                               to_cpu=True,
                                               no_grad=True,
                                               to_numpy=to_numpy,
                                               verbose=verbose)
    if not prev_cuda:
        model.cpu()
    return response

def model2model_cors(model1, model2, model1_layers=[],
                                            model2_layers=[],
                                            stim=None,
                                            batch_size=500,
                                            contrast=1.0,
                                            n_samples=5000,
                                            n_repeats=3,
                                            use_ig=True,
                                            verbose=True):
    """
    Takes two models and correlates the activations at each layer.
    Returns a dict of correlation marices.

    model1 - torch Module
    model2 - torch Module
    stim - ndarray (T,H,W) or None
        optional argument to specify the stimulus. If none, stimulus
        defaults to whitenoise
    model1_layers - set or list of strs
        the layers of model 1 to be correlated
    model2_layers - set or list of strs
        the layers of model 2 to be correlated
    batch_size: int
        size of batches when performing computations on GPU
    contrast: float
        contrast of whitenoise stimulus for model input
    n_samples: int
        number of time points of stimulus for model input
    n_repeats: int
        number of times to repeat whitenoise stimulus in temporal dim.
        Essentially adjusts the frame rate of the video. Larger values
        of n_repeats leads to slower video frame rates.
    use_ig: bool
        if true, uses integrated gradient rather than activations for
        model correlations. if the specified layer is outputs, then
        use_ig is ignored

    returns:
        intr_cors - dict
                - mod1_layer: list
                - mod1_chan: list
                - mod1_row: list
                - mod1_col: list
                - mod2_layer: list
                - mod2_chan: list
                - mod2_row: list
                - mod2_col: list
                - contrast: list
                - cor: list
    """
    intr_cors = {
        "mod1_layer":[],
        "mod1_chan":[],
        "mod1_row":[],
        "mod1_col":[],
        "mod2_layer":[],
        "mod2_chan":[],
        "mod2_row":[],
        "mod2_col":[],
        "contrast":[],
        "cor":[],
    }
    if len(model1_layers) == 0:
        model1_layers = get_conv_layer_names(model1)
    if len(model2_layers) == 0:
        model2_layers = get_conv_layer_names(model2)
    model1_cuda = next(model1.parameters()).is_cuda
    model2_cuda = next(model2.parameters()).is_cuda
    model2.cpu()

    nx = min(model1.img_shape[1], model2.img_shape[1])

    if stim is None:
        stim = tdrstim.repeat_white(n_samples, nx=nx,
                                        contrast=contrast,
                                        n_repeats=n_repeats,
                                        rand_spat=True)

    # Collect Responses
    model1.to(DEVICE)
    response1 = get_response(model1, stim, use_ig=use_ig,
                                        model_layers=model1_layers,
                                        batch_size=batch_size,
                                        verbose=verbose)
    model1.cpu()
    model2.to(DEVICE)
    response2 = get_response(model2, stim, use_ig=use_ig,
                                           model_layers=model2_layers,
                                           batch_size=batch_size,
                                           verbose=verbose)
    model2.cpu()

    for mod1_layer in intr_cors.keys():
        resp1 = response1[mod1_layer]
        resp1 = resp1.reshape(len(resp1), -1)
        for mod2_layer in intr_cors[mod1_layer].keys():
            resp2 = response2[mod2_layer]
            resp2 = resp2.reshape(len(resp2),-1)
            cor_mtx = tdrutils.mtx_cor(resp1,resp2,
                                    batch_size=batch_size,
                                    to_numpy=True)
            for mod1_idx in range(cor_mtx.shape[0]):
                for mod2_idx in range(cor_mtx.shape[1]):
                    intr_cors['contrast'].append(contrast)
                    cor = cor_mtx[mod1_idx,mod2_idx]
                    intr_cors['cor'].append(cor)
                    (chan,row,col) = np.unravel_index(mod1_idx,
                                          response1[mod1_layer].shape)
                    intr_cors['mod1_layer'].append(mod1_layer)
                    intr_cors['mod1_chan'].append(chan)
                    intr_cors['mod1_row'].append(row)
                    intr_cors['mod1_col'].append(col)
                    (chan,row,col) = np.unravel_index(mod2_idx,
                                          response1[mod2_layer].shape)
                    intr_cors['mod2_layer'].append(mod2_layer)
                    intr_cors['mod2_chan'].append(chan)
                    intr_cors['mod2_row'].append(row)
                    intr_cors['mod2_col'].append(col)
    if model1_cuda:
        model1.to(DEVICE)
    if model2_cuda:
        model2.to(DEVICE)
    return intr_cors

def model2model_cor_mtxs(model1, model2,
                      model1_layers={"sequential.0", "sequential.6"},
                      model2_layers={"sequential.0", "sequential.6"},
                      stim=None, response1=None, response2=None,
                      batch_size=500, contrast=1.0,
                      n_samples=5000, n_repeats=3, use_ig=True,
                      verbose=True):
    """
    Takes two models and correlates the activations at each layer.
    Returns a dict of correlation marices.

    model1 - torch Module
    model2 - torch Module
    model1_layers - set of strs
        the layers of model 1 to be correlated
    model2_layers - set of strs
        the layers of model 2 to be correlated
    stim - ndarray (T,H,W) or None
        optional argument to specify the stimulus. If none, stimulus
        defaults to whitenoise
    batch_size: int
        size of batches when performing computations on GPU
    contrast: float
        contrast of whitenoise stimulus for model input
    n_samples: int
        number of time points of stimulus for model input
    n_repeats: int
        number of times to repeat whitenoise stimulus in temporal dim.
        Essentially adjusts the frame rate of the video. Larger values
        of n_repeats leads to slower video frame rates.
    use_ig: bool
        if true, uses integrated gradient rather than activations for
        model correlations. if the specified layer is outputs, then
        use_ig is ignored

    returns:
        intr_cors - dict
                keys: str
                    model1 layer names
                vals: dict
                    keys: str
                        model2 layer names
                    vals: ndarray (M1, M2)
                        where M1 is the flattened activations
                        in model1 at that layer and M2 are the
                        flattened activations for model2 at
                        that layer
    """
    model1_cuda = next(model1.parameters()).is_cuda
    model2_cuda = next(model2.parameters()).is_cuda
    model2.cpu()

    nx = min(model1.img_shape[1], model2.img_shape[1])

    if stim is None and (response1 is None or response2 is None):
        stim = tdrstim.repeat_white(n_samples, nx=nx,
                                     contrast=contrast,
                                     n_repeats=n_repeats,
                                     rand_spat=True)

    # Collect responses
    with torch.no_grad():
        model1.to(DEVICE)
        if response1 is None:
            response1 = get_response(model1, stim, use_ig=use_ig,
                                            model_layers=model1_layers,
                                            batch_size=batch_size,
                                            verbose=verbose)
        model1.cpu()
        model2.to(DEVICE)
        if response2 is None:
            response2 = get_response(model2, stim, use_ig=use_ig,
                                            model_layers=model2_layers,
                                            batch_size=batch_size,
                                            verbose=verbose)
        model2.cpu()

    intr_cors = {m1_layer:{m2_layer:None for m2_layer in\
                        model2_layers} for m1_layer in model1_layers}

    for mod1_layer in intr_cors.keys():
        resp1 = response1[mod1_layer]
        resp1 = resp1.reshape(len(resp1), -1)
        for mod2_layer in intr_cors[mod1_layer].keys():
            resp2 = response2[mod2_layer]
            resp2 = resp2.reshape(len(resp2),-1)
            bsize = batch_size
            cor_mtx = tdrutils.mtx_cor(resp1,resp2,batch_size=bsize)
            intr_cors[mod1_layer][mod2_layer] = cor_mtx
    if model1_cuda:
        model1.to(DEVICE)
    if model2_cuda:
        model2.to(DEVICE)
    return intr_cors

def model2model_one2one_cors(model1, model2,
                       model1_layers={"sequential.0", "sequential.6"},
                       model2_layers={"sequential.0", "sequential.6"},
                       batch_size=500,  contrast=1.0, n_samples=5000,
                       n_repeats=3, use_ig=True, verbose=True):
    """
    Finds the correlation mapping that maximizes the sum total of
    correlation subject to the constraint that no two model units can
    correlate with the same unit in the other model. i.e. no two
    model1 units can correlate with the same model2 unit.

    model1 - torch Module
    model2 - torch Module
    model1_layers - set of strs
        the layers of model 1 to be correlated
    model2_layers - set of strs
        the layers of model 2 to be correlated
    batch_size: int
        size of batches when performing computations on GPU
    contrast: float
        contrast of whitenoise stimulus for model input
    n_samples: int
        number of time points of stimulus for model input
    n_repeats: int
        number of times to repeat whitenoise stimulus in temporal dim.
        Essentially adjusts the frame rate of the video. Larger values
        of n_repeats leads to slower video frame rates.
    use_ig: bool
        if true, uses integrated gradient rather than activations for
        model correlations. if the specified layer is outputs, then
        use_ig is ignored

    This is ultimately a recursive algorithm. 

    Algorithm:
        Get correlation matrix
        Create argsort matrix along both dim 0 and 1
        for each unit in arg matrix 1:
            sequentially check free ordered unit correlations
            if unit 1 and unit 2 maximally correlate with each other, 
                pair them and mark it
            otherwise, sequentially check unit2's oredered unit cors 
                until an open match is found
    """
    
    ## Dict of correlation matrices between each layer combination
    cor_mtxs = model2model_cor_mtxs(model1, model2,
                                         model1_layers=model1_layers,
                                         model2_layers=model2_layers,
                                         batch_size=batch_size,
                                         contrast=contrast, 
                                         n_samples=n_samples,
                                         n_repeats=n_repeats,
                                         use_ig=use_ig,
                                         verbose=verbose)
    ## Create complete correlation matrix
    cor_mtx = []
    m1_layers = sorted(list(cor_mtxs.keys()))
    # Assumes same m2 layers for every m1 layer
    m2_layers = sorted(list(cor_mtxs[m1_layers[0]]))
    for m1_layer in m1_layers:
        mtx = []
        for m2_layer in m2_layers:
            mtx.append(cor_mtxs[m1_layer][m2_layer])
        mtx = np.concatenate(mtx, axis=1)
        cor_mtx.append(mtx)
    cor_mtx = np.concatenate(cor_mtx, axis=0) # (N_m1units,N_m2units)

    bests1, bests2 = tdrutils.best_one2one_mapping(cor_mtx)
    return cor_mtx, bests1, bests2

def get_intr_cors(model, stim_dict, mem_pot_dict,
                              layers={"sequential.2", "sequential.8"},
                              batch_size=500,
                              slide_steps=0,
                              window=True,
                              verbose=False):
    """
    Takes a model and dicts of stimuli and membrane potentials to find
    all correlations for each layer in the model. Returns a dict that
    can easily be converted into a pandas data frame.

    model - torch Module
    stim_dict - dict of interneuron stimuli
        keys: str (interneuron data file name)
            vals: dict
                keys: stim_types
                    vals: ndarray (T, H, W)
    mem_pot_dict - dict of interneuron membrane potentials
        keys: str (interneuron data file name)
            vals: dict
                keys: stim_types
                    vals: ndarray (CI, T)
                        CI is cell idx and T is time
    slide_steps - int
        slides the stimulus in strides by this amount in an attempt
        to align the receptive fields of the interneurons with the
        ganglion cells

    window - bool
        if true, stimulus is windowed prior to being run through model.

    returns:
        intr_df - pandas DataFrame
                - cell_file: string
                - cell_idx: int
                - stim_type: string
                - cell_type: string
                - cor: float
                - layer: string
                - chan: int
                - row: int
                - col: int
                - xshift: int
                - yshift: int
    """
    intr_cors = {
        "cell_file":[], 
        "cell_idx":[],
        "stim_type":[],
        "cell_type":[],
        "layer":[],
        "chan":[],
        "row":[],
        "col":[],
        "cor":[],
        "xshift":[],
        "yshift":[]
    }
    layers = sorted(list(layers))
    for cell_file in stim_dict.keys():
        for stim_type in stim_dict[cell_file].keys():
            lam = lambda: collections.defaultdict(lambda: dict())
            best_mtxs = collections.defaultdict(lam)
            shifts = tdrutils.get_shifts(row_steps=slide_steps,
                                col_steps=slide_steps,
                                n_row=model.img_shape[1],
                                n_col=model.img_shape[2])
            for xshift, yshift in shifts:
                if verbose:
                    print("Shift x:{}, Shift y:{}".format(xshift,
                                                          yshift))
                D,H,W = model.img_shape
                stim = stim_dict[cell_file][stim_type]
                if not (xshift == 0 and yshift == 0):
                    zeros = np.zeros((len(stim),H,W))
                    stim = tdrstim.shifted_overlay(zeros, stim,
                                              row_shift=xshift,
                                              col_shift=yshift)
                stim = tdrstim.spatial_pad(stim, W=H, H=W)
                if window:
                    stim = tdrstim.rolling_window(stim, D)

                if verbose:
                    temp = cell_file.split("/")[-1].split(".")[0]
                    s = "cell_file:{}, stim_type:{}..."
                    cellstim = s.format(temp, stim_type)
                    print("Collecting model response for "+cellstim)
                response = tdrutils.inspect(model, stim,
                                       insp_keys=layers,
                                       batch_size=batch_size,
                                       to_numpy=True,
                                       verbose=verbose)
                pots = mem_pot_dict[cell_file][stim_type]
                shapes = dict()
                for layer in layers:
                    if verbose:
                        print("Calculating cors for layer:", layer)
                    resp = response[layer]
                    shapes[layer] = resp.shape
                    resp = resp.reshape(len(resp),-1)
                    # Retrns ndarray (Model Neurons, Potentials)
                    cor_mtx = tdrutils.mtx_cor(resp, pots.T,
                                      batch_size=batch_size,
                                      to_numpy=True)
                    # We need to store the best mtx for each potential
                    for ci in range(cor_mtx.shape[1]):
                        if ci in best_mtxs[layer]:
                            bests = best_mtxs[layer][ci]['cors']
                            diff = bests.max(0)-cor_mtx[:,ci].max(0)
                        if ci not in best_mtxs[layer] or diff < 0:
                            best_mtxs[layer][ci]["cors"]=cor_mtx[:,ci]
                            best_mtxs[layer][ci]["xshift"] = xshift
                            best_mtxs[layer][ci]["yshift"] = yshift

            if verbose:
                print("Recording best shifts")
            for layer in best_mtxs.keys():
                modu = tdrutils.get_module_by_name(model,layer)
                shape = shapes[layer][1:]
                for cell_idx in range(cor_mtx.shape[1]):
                    bests =    best_mtxs[layer][cell_idx]['cors']
                    xshift =   best_mtxs[layer][cell_idx]['xshift']
                    yshift =   best_mtxs[layer][cell_idx]['yshift']
                    for unit_idx in range(bests.shape[0]):
                        intr_cors['cell_file'].append(cell_file)
                        intr_cors['cell_idx'].append(cell_idx)
                        intr_cors['stim_type'].append(stim_type)
                        cell_type = cell_file.split("/")[-1] 
                        cell_type = cell_type.split("_")[0][:-1]
                        # amacrine or bipolar
                        intr_cors['cell_type'].append(cell_type)
                        cor = bests[unit_idx]
                        intr_cors['cor'].append(cor)
                        intr_cors['layer'].append(layer)
                        if len(shape) == 1:
                            chan,row,col = unit_idx,0,0
                        else:
                            (chan,row,col)= np.unravel_index(unit_idx,
                                                             shape)
                        intr_cors['chan'].append(chan)
                        intr_cors['row'].append(row)
                        intr_cors['col'].append(col)
                        intr_cors['xshift'] = xshift
                        intr_cors['yshift'] = yshift

    intr_df = pd.DataFrame(intr_cors)
    dups = ['cell_file', 'cell_idx', 'stim_type', "layer", "chan"]
    intr_df = intr_df.sort_values(by="cor", ascending=False)
    return intr_df.drop_duplicates(dups)

def get_cor_generalization(model, stim_dict, mem_pot_dict,
                               layers={"sequential.2","sequential.8"},
                               batch_size=500, verbose=False):
    """
    Collects the interneuron correlation of each channel of each layer
    with each membrane potential. Also collects the correlation of
    each of the best correlated units with the other stimuli. Acts as
    a measure of generalization to the cell response regardless of
    stimulus type.

    model - torch Module
    stim_dict - dict of interneuron stimuli
        keys: str (interneuron data file name)
            vals: dict
                keys: stim_types (must be more than one!!)
                    vals: ndarray (T, H, W)
    mem_pot_dict - dict of interneuron membrane potentials
        keys: str (interneuron data file name)
            vals: dict
                keys: stim_types (must be more than one!!)
                    vals: ndarray (CI, T)
                        CI is cell idx and T is time
    returns:
        For each entry, there should be a corresponding entry matching
        in all fields except for stim_type and cor.
        intr_cors - dict
                - cell_file: list
                - cell_idx: list
                - stim_type: list
                - cor: list
                - layer: list
                - chan: list
                - row: list
                - col: list
    """
    table = {"cell_file":[], "cell_idx":[], "stim_type":[], "cor":[],
                           "layer":[], "chan":[], "row":[], "col":[]}
    layers = sorted(list(layers))
    for cell_file in stim_dict.keys():
        responses = dict()
        # Get all responses
        for stim_type in stim_dict[cell_file].keys():
            stim = stim_dict[cell_file][stim_type]
            stim = tdrstim.spatial_pad(stim,model.img_shape[1])
            stim = tdrstim.rolling_window(stim, model.img_shape[0])
            response = tdrutils.inspect(model, stim, insp_keys=layers,
                                                batch_size=batch_size)
            # Fix response shapes
            for layer in layers:
                resp = response[layer]
                if layer != "outputs" and len(resp.shape) <= 2:
                    layer_idx = tdrutils.get_layer_idx(model,
                                                 layer=layer)
                    shape = (len(resp), model.chans[layer_idx],
                                        *model.shapes[layer_idx])
                    resp = resp.reshape(shape)
                response[layer] = resp
            responses[stim_type] = response
        for stim_type in stim_dict[cell_file].keys():
            pots = mem_pot_dict[cell_file][stim_type]
            for cell_idx in range(len(pots)):
                for l,layer in enumerate(layers):
                    if verbose:
                        name = cell_file.split("/")[-1]
                        s = "Evaluating file:{}, stim:{}, idx:{},\
                                                         layer:{}"
                        print(s.format(name,stim_type,cell_idx,layer))
                    resp = responses[stim_type][layer]
                    cor_map = correlation_map(pots[cell_idx],resp)
                    for chan in range(len(cor_map)):
                        idx = np.argmax(cor_map[chan].ravel())
                        r = cor_map[chan].ravel()[idx]
                        idx = np.unravel_idx(idx, cor_map[chan].shape)
                        row, col = idx
                        table['cell_file'].append(cell_file)
                        table['cell_idx'].append(cell_idx)
                        table['stim_type'].append(stim_type)
                        table["cor"].append(r)
                        table['layer'].append(layer)
                        table['chan'].append(chan)
                        table['row'].append(row)
                        table['col'].append(col)

                        for stim_t in stim_dict[cell_file].keys():
                            if stim_t != stim_type:
                                pot = mem_pot_dict[cell_file]
                                pot = pot[stim_t][cell_idx]
                                temp_resp = responses[stim_t][layer]
                                temp_resp = temp_resp[:,chan,row,col]
                                r,_ = pearsonr(temp_resp, pot)
                                r = r if not np.isnan(r) and r > -1\
                                                              else 0
                                if np.log(np.abs(temp_resp).mean()) <\
                                            ABS_MEAN_CUTOFF or\
                                            np.log(temp_resp.std()) <\
                                            STD_CUTOFF:
                                    if verbose and abs(r) > 0.1:
                                        s = "Extremely small layer\
                                             values, pearson of {}\
                                             can not be trusted and\
                                             is being set to 0"
                                        print(s.format(r))
                                    r = 0
                                table['cell_file'].append(cell_file)
                                table['cell_idx'].append(cell_idx)
                                table['stim_type'].append(stim_t)
                                table["cor"].append(r)
                                table['layer'].append(layer)
                                table['chan'].append(chan)
                                table['row'].append(row)
                                table['col'].append(col)
    return table

def get_stim_attr(model, stim, cell_idx=None):
    """
    Computes the attribution of each pixel to the final output.
    """
    raise NotImplementedError

def compare_cell_pathways(model1, model2, stim=None, layers=[],
                                                 cell1=0,
                                                 comp_cells=[],
                                                 batch_size=500,
                                                 n_samples=10000,
                                                 sim_type="maximum",
                                                 verbose=True):
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
    cell1: int
        the first output channel of interest.
    comp_cells: list of ints
        a list of the cells to be compared with cell1
    n_samples: int
        the number of samples to be used for the stimulus. Only
        applies if stim is None
    sim_type: str
        denotes the similarity metric to be used. available options
        are 'maximum', 'one2one', 'cca', 'np_cca', 'dot'

    Returns:
        similarities: dict of dicts
            keys: int
                the cell that was compared to cell1
            vals: dict
                keys: tuple of ints
                    the row and col of the first channel
                vals: dict
                    keys: tuple of ints
                        the row and col of the second channel
                    vals: float
                        the measured similarity between their integrated
                        gradients
    """
    if layers is None or len(layers) == 0:
        layers = tdrutils.get_conv_layer_names(model1)[:-1]
    if stim is None:
        stim = tdrstim.repeat_white(n_samples, nx=model1.img_shape[1],
                                                    n_repeats=3,
                                                    rand_spat=True)
    similarities = dict()
    gc_cors = dict()
    shape = model1.shapes[-1]
    center = (shape[0]//2, shape[1]//2)

    cell_idx = (cell1, *center)
    grab_idx = tdrutils.get_module_idx(model1, GrabUnits)
    if grab_idx >= 0:
        m = nn.Sequential(*model1.sequential[:grab_idx],
                            model1.sequential[grab_idx+1:])
    else:
        m = model1.sequential[:-1]
    gc_activs = tdrutils.inspect(m.to(DEVICE),stim,
                                        batch_size=batch_size,
                                        to_numpy=True,
                                        no_grad=True)['outputs']
    del m
    torch.cuda.empty_cache()
    c1_resp = gc_activs[:,cell1,center[0],center[1]]
    grab_idx = tdrutils.get_module_idx(model2, GrabUnits)
    if grab_idx >= 0:
        m = nn.Sequential(*model2.sequential[:grab_idx],
                            model2.sequential[grab_idx+1:])
    else:
        m = model2.sequential[:-1]
    gc_activs = tdrutils.inspect(m.to(DEVICE),stim, batch_size=batch_size,
                                        to_numpy=True,
                                        no_grad=True)['outputs']
    del m
    torch.cuda.empty_cache()
    gc_shape = gc_activs.shape[-2:]

    if verbose:
        print("Collecting ig response at:", cell_idx)
    resp1 = get_response(model1, stim, model_layers=layers,
                                      batch_size=batch_size,
                                      cell_idx=cell_idx,
                                      use_ig=True,
                                      to_numpy=False)
    resp1 = tdrutils.flatcat(resp1)
    for i,cell2 in enumerate(comp_cells):
        flat = gc_activs[:,cell2].reshape(len(c1_resp),-1)
        cor_mtx = tdrutils.mtx_cor(flat, c1_resp[:,None],to_numpy=True)
        arg = np.argmax(cor_mtx[:,0])
        (row,col) = np.unravel_index(arg,gc_shape)
        row,col = int(row),int(col)
        c2_resp = gc_activs[:,cell2,row,col]

        cell_idx = (cell2, row, col)
        if verbose:
            s = "Comparing cell {} to {} | ".format(cell1, cell2)
            s += " {} cells left".format(len(comp_cells)-i)
            print(s)
        resp2 = get_response(model2, stim, model_layers=layers,
                                batch_size=batch_size,
                                cell_idx=cell_idx,
                                use_ig=True,
                                to_numpy=False)
        resp2 = tdrutils.flatcat(resp2)
        sim = tdrutils.get_similarity(resp1, resp2, sim_type,
                                       batch_size=batch_size)
        similarities[cell2] = {center:dict()}
        similarities[cell2][center][(row,col)] = sim
        if verbose:
            r = tdrutils.pearsonr(c1_resp,c2_resp)
            gc_cors[cell2] = r[0]
            print("Similarity:", sim, "| GC:", r[0])
    return similarities, gc_cors


