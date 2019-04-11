import matplotlib
from .plotting import adjust_spines
from scipy.stats import sem, pearsonr
from tqdm import tqdm
import os
import re
import h5py
import collections
import pyret.filtertools as ft
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, FormatStrFormatter

# Functions for correlation maps and loading David's stimuli in deep retina models.
def pad_to_edge(stim):
    '''
    Pads the spatial dimensions of a stimulus to be 50x50 for deep retina models.
    
    Args:
        stim: (time, space, space) numpy array.
    Returns:
        padded_stim: (time, space, space) padded numpy array.
    '''
    padded_stim = np.zeros((stim.shape[0], 50, 50))
    height = stim.shape[1]
    width = stim.shape[2]
    
    if height >= 50 or width >= 50: # Height must be less than 50.
        return stim.copy()
    
    y_start = int((50 - height)/2.0)
    x_start = int((50 - width)/2.0)
    
    padded_stim[:, y_start:y_start+height, x_start:x_start+width] = stim
    return padded_stim

def correlation_map(membrane_potential, model_layer):
    '''
    Takes a 1d membrane potential and computes the correlation with every tiled unit in model_layer.
    
    Args:
        membrane_potential: 1-d numpy array
        model_layer: (time, space, space) layer of activities
    '''
    height = model_layer.shape[-2]
    width = model_layer.shape[-1]
    correlations = np.zeros((height, width))
    for y in range(height):
        for x in range(width):
            adjusted_layer = model_layer[:,y,x]/(np.max(model_layer[:,y,x])+1e-40)
            r,_ = pearsonr(membrane_potential, adjusted_layer)
            correlations[y,x] = r if not np.isnan(r) and r < 1 and r > -1 else 0
    return correlations

def max_correlation(membrane_potential, model_layer, abs_val=False):
    '''
    Takes a 1d membrane potential and computes the correlation with every tiled unit in model_layer.
    
    Args:
        membrane_potential: 1-d numpy array
        model_layer: (time, celltype, space, space) layer of activities
    '''
    if len(model_layer.shape) > 2:
        cor_maps = [correlation_map(membrane_potential, model_layer[:,c]) for c in range(model_layer.shape[1])]
        if abs_val:
            cor_maps = [np.absolute(m) for m in cor_maps]
        return np.max([np.max(m) for m in cor_maps])
    else:
        adjusted_layer = model_layer/(np.max(model_layer)+1e-40)
        pearsons = [pearsonr(membrane_potential, adjusted_layer)[0] for c in range(model_layer.shape[1])]
        pearsons = [r if not np.isnan(r) and r < 1 and r > -1 else 0 for r in pearsons]
        if abs_val:
            pearsons = [np.absolute(r) for r in pearsons]
        return np.max(pearsons)

def sorted_correlation(membrane_potential, model_layer):
    '''
    Takes a 1d membrane potential and computes the maximum correlation with respect to each model celltype,
    sorted from highest to lowest.
    
    Args:
        membrane_potential: 1-d numpy array
        model_layer: (time, celltype, space, space) layer of activities
    '''
    if len(model_layer.shape) > 2:
        return sorted(
            [np.max(correlation_map(membrane_potential, model_layer[:,c])) for c in range(model_layer.shape[1])])
    else:
        adjusted_layer = model_layer/(np.max(model_layer)+1e-40)
        pearsons = [pearsonr(membrane_potential, adjusted_layer)[0] for c in range(model_layer.shape[1])]
        pearsons = [r if not np.isnan(r) and r < 1  and r > -1 else 0 for r in pearsons]
        return sorted(pearsons)

def max_correlation_all_layers(membrane_potential, model_response, layer_keys=['conv1', 'conv2'], abs_val=False):
    '''
    Takes a 1d membrane potential and computes the maximum correlation over the 2 conv layers within a model.
    Args: 
        membrane_potential: 1-d numpy array
        model_response: a dict of layer activities
        layer_keys: keys of model_response to be tested
        abs_val: use absolute value of correlations
    '''
    max_cors = [max_correlation(membrane_potential, model_response[k], abs_val=abs_val) for k in layer_keys]
    return max(max_cors)

def argmax_correlation_recurse_helper(membrane_potential, model_layer, shape, idx, abs_val=False):
    if len(shape) == 0: # base case
        layer = model_layer[:,idx[0]]
        # Does nothing if model_layer was originally (time, celltype) dims
        # Otherwise sequentially widdles layer down to 1 dimension
        for i in idx[1:]: 
            layer = layer[:,i] # Sequentially selects dims of the model resulting in layer[:,idx[1],idx[2]...]
        r, _ = pearsonr(membrane_potential, layer/np.max(layer))
        if abs_val:
            r = np.absolute(r)
        if np.isnan(r) or r >= 1 or r <= -1: r = 0
        return r, idx
    else:
        max_r = -1
        best_idx = None
        for i in range(shape[0]):
            args = (membrane_potential, model_layer, shape[1:], (*idx, i), abs_val)
            r, local_idx = argmax_correlation_recurse_helper(*args)
            if not np.isnan(r) and r < 1 and r > max_r:
                max_r = r
                best_idx = local_idx
        return max_r, best_idx

def argmax_correlation(membrane_potential, model_layer, ret_max_cor=False, abs_val=False):
    '''
    Takes a 1d membrane potential and computes the correlation with every tiled unit in model_layer.
    
    Args:
        membrane_potential: 1-d numpy array
        model_layer: (time, celltype, space, space) or (time, celltype) layer of activities
        abs_val: use absolute value of correlations
    '''
    assert len(model_layer.shape) >= 2
    max_r = -1
    best_idx = None
    for i in range(model_layer.shape[1]):
        r, idx = argmax_correlation_recurse_helper(membrane_potential,model_layer,model_layer.shape[2:],(i,), abs_val=abs_val)
        if not np.isnan(r) and r < 1 and r > max_r:
            max_r = r
            best_idx = idx
    if ret_max_cor:
        return best_idx, max_r
    return best_idx

def argmax_correlation_all_layers(membrane_potential, model_response, layer_keys=['conv1', 'conv2'], ret_max_cor_all_layers=False, abs_val=False):
    '''
    Takes a 1d membrane potential and computes the maximum correlation over the 2 conv layers within a model.
    Args: 
        membrane_potential: 1-d numpy array
        model_response: a dict of layer activities
        layer_keys: keys of model_response to be tested
        abs_val: use absolute value of correlations
    '''
    max_r = -1
    best_idx = None
    for key in layer_keys:
        response = model_response[key]
        idxs, r = argmax_correlation(membrane_potential, response, ret_max_cor=True, abs_val=abs_val)
        if not np.isnan(r) and r < 1 and r > max_r:
            max_r = r
            best_idx = (key, *idxs)
    if ret_max_cor_all_layers:
        return best_idx, max_r
    return best_idx

def classify(membrane_potential, model_response, time, layer_keys=['conv1', 'conv2'], abs_val=False):
    '''
    Finds the most correlated cell in a model to a membrane potential.
    Args:
        membrane_potential: 1-d numpy array
        model_response: dict of activity at each layer of model
        time: the time to take into consideration
        layer_keys: keys of model_response to be tested
        abs_val: use absolute value of correlations
    Returns:
        a tuple with the layer, celltype, spatial indices, and the correlation value
    '''
    model_response_time = {k:model_response[k][:time] for k in layer_keys}
    best_cell, max_cor_all_layers = argmax_correlation_all_layers(membrane_potential[:time], model_response_time, layer_keys=layer_keys, ret_max_cor_all_layers=True, abs_val=abs_val)
    if len(best_cell) == 2:
        return best_cell[0], best_cell[1], max_cor_all_layers
    return best_cell[0], best_cell[1], (best_cell[2], best_cell[3]), max_cor_all_layers

def classify_subtypes(membrane_potential, model_response, time, layer_keys=['conv1', 'conv2']):
    '''
    Finds the highest correlation for each subtype in each layer.
    Args:
        membrane_potential: 1-d numpy array
        model_response: dict of activity at each layer of model
        time: time up to which to consider
        layer_keys: keys of model_response to be tested
    Returns:
        the correlations as an array
    '''
    correlations = [np.max(correlation_map(membrane_potential[:time], model_response[k][:time,c])) for c in range(model_response[k].shape[1])]
    return correlations

def plot_max_correlations(membrane_potential, model_response):
    '''
    Plots the top correlations of celltypes over time
    membrane_potential: 1-d numpy array
    model_response: dict of activity at each layer of model
    num_cells: the number of celltypes to plot
    '''
    y_0 = []
    y_1 = []
    y_2 = []
    top_cell_subtypes = np.argsort(classify_subtypes(membrane_potential, model_response, membrane_potential.shape[0]))[-3:]
    for time in range(100, 1000, 100):
        y_0.append(classify_subtypes(membrane_potential, model_response, time)[top_cell_subtypes[0]])
        y_1.append(classify_subtypes(membrane_potential, model_response, time)[top_cell_subtypes[1]])
        y_2.append(classify_subtypes(membrane_potential, model_response, time)[top_cell_subtypes[2]])
    plt.plot(y_0)
    plt.plot(y_1)
    plt.plot(y_2)
    plt.show()

def create_rfs(stimulus, membrane_potential, model_cell_response, filter_length, cell_info, index):
    '''
    Creates plots of cell and model rfs side by side
    Args:
        stimulus: stimulus presented to cell and model, 3-d numpy array
        membrane_potential: 1-d numpy array
        model_cell_response: 1-d numpy array
        cell_info: the information of the cell returned from classify
        filter_length: filter length
        index: the index of the cell 1-34. 1-6 are bipolar cells, the rest are amacrine.
    '''
    stim_size = stimulus.shape[0]
    stim_tmp = stimulus[filter_length:]
    model_title = "Model:{0}, {1}".format(cell_info[0], cell_info[1])
    if index < 7:
        cell_title = "bipolar cell"
    else:
        cell_title = "amacrine cell"
    rc_model, lags_model = ft.revcorr(scipy.stats.zscore(stimulus)[filter_length:], model_cell_response, nsamples_before=0, nsamples_after=filter_length)
    rc_cell, lags_cell = ft.revcorr(scipy.stats.zscore(stimulus)[filter_length:], membrane_potential, nsamples_before=filter_length)
    spatial_model, temporal_model = ft.decompose(rc_model)
    spatial_cell, temporal_cell = ft.decompose(rc_cell)
    plt.subplot(2,2,1)
    img =plt.imshow(spatial_model, cmap = 'seismic', clim=[-np.max(abs(spatial_model)), np.max(abs(spatial_model))])
    plt.title(model_title)
    plt.subplot(2,2,2)
    img =plt.imshow(spatial_cell, cmap = 'seismic', clim=[-np.max(abs(spatial_cell)), np.max(abs(spatial_cell))])
    plt.title(cell_title)
    plt.subplot(2,2,3)
    plt.plot(temporal_model)
    plt.subplot(2,2,4)
    plt.plot(temporal_cell)
    fname = 'natural_scene_results/{0}.png'.format(index)
    plt.savefig(fname)
    plt.clf()
    
def plot_responses(membrane_potential, model_cell_response):
    '''
    Plots the activity of the most correlated cell and membrane potential against each other.
    '''
    plt.subplot(1,2,1)
    plt.plot(model_cell_response)
    plt.subplot(1,2,2)
    plt.plot(membrane_potential)
