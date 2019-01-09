import matplotlib
from plotting import adjust_spines
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
    
    assert height < 50, 'Height must be less than 50.'
    assert width < 50, 'Width must be less than 50.'
    
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
    height = model_layer.shape[1]
    width = model_layer.shape[2]
    correlations = np.zeros((height, width))
    for y in range(height):
        for x in range(width):
            correlations[y,x] = pearsonr(membrane_potential, model_layer[:, y, x])[0]
    return correlations

def max_correlation(membrane_potential, model_layer):
    '''
    Takes a 1d membrane potential and computes the correlation with every tiled unit in model_layer.
    
    Args:
        membrane_potential: 1-d numpy array
        model_layer: (time, celltype, space, space) layer of activities
    '''
    if len(model_layer.shape) > 2:
        return np.max(
            [np.max(correlation_map(membrane_potential, model_layer[:,c])) for c in range(model_layer.shape[1])])
    else:
        return np.max(
            [pearsonr(membrane_potential, model_layer[:,c])[0] for c in range(model_layer.shape[1])])

def argmax_correlation(membrane_potential, model_layer):
    '''
    Takes a 1d membrane potential and computes the correlation with every tiled unit in model_layer.
    
    Args:
        membrane_potential: 1-d numpy array
        model_layer: (time, celltype, space, space) layer of activities
    '''
    if len(model_layer.shape) > 2:
        space_idx = [np.unravel_index(np.argmax(correlation_map(membrane_potential, model_layer[:,c])), (model_layer.shape[2], model_layer.shape[3])) for c in range(model_layer.shape[1])]
        celltype_idx = np.argmax(
            [np.max(correlation_map(membrane_potential, model_layer[:,c])) for c in range(model_layer.shape[1])])
    return celltype_idx, space_idx

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
        return sorted(
            [pearsonr(membrane_potential, model_layer[:,c])[0] for c in range(model_layer.shape[1])])
def max_correlation_all_layers(membrane_potential, model_response):
    '''
    Takes a 1d membrane potential and computes the maximum correlation over the 2 conv layers within a model.
    Args: 
        membrane_potential: 1-d numpy array
        model_response: a dict of layer activities
    '''
    return max(max_correlation(membrane_potential, model_response['conv2d_1']), 
               max_correlation(membrane_potential, model_response['conv2d_2']))

def argmax_correlation_all_layers(membrane_potential, model_response):
    '''
    Takes a 1d membrane potential and computes the maximum correlation over the 2 conv layers within a model.
    Args: 
        membrane_potential: 1-d numpy array
        model_response: a dict of layer activities
    '''
    if max_correlation(membrane_potential, model_response['conv2d_1']) == max_correlation_all_layers(membrane_potential, model_response):
        layer = 'conv2d_1'
    else:
        layer = 'conv2d_2'
    cell_type = argmax_correlation(membrane_potential, model_response[layer])[0]
    # an example return: ('conv2d_1', 8, (10, 5)) 
    return (layer, cell_type, argmax_correlation(membrane_potential, model_response[layer])[1][cell_type])

def classify(membrane_potential, model_response, time):
    '''
    Finds the most correlated cell in a model to a membrane potential.
    Args:
        membrane_potential: 1-d numpy array
        model_response: dict of activity at each layer of model
        time: the time to take into consideration
    Returns:
        a tuple with the layer, celltype, spatial indices, and the correlation value
    '''
    model_response_time = {}
    model_response_time['conv2d_1'] = model_response['conv2d_1'][:time]
    model_response_time['conv2d_2'] = model_response['conv2d_2'][:time]
    best_cell = argmax_correlation_all_layers(membrane_potential[:time], model_response_time)
    return best_cell[0], best_cell[1], best_cell[2], max_correlation_all_layers(membrane_potential[:time], model_response_time)

def classify_subtypes(membrane_potential, model_response, time):
    '''
    Finds the highest correlation for each subtype in each layer.
    Args:
        membrane_potential: 1-d numpy array
        model_response: dict of activity at each layer of model
        time: time up to which to consider
    Returns:
        the correlations as an array
    '''
    correlations = [np.max(correlation_map(membrane_potential[:time], model_response['conv2d_1'][:time,c])) for c in range(model_response['conv2d_1'].shape[1])]
    correlations.extend([np.max(correlation_map(membrane_potential[:time], model_response['conv2d_2'][:time,c])) for c in range(model_response['conv2d_2'].shape[1])])
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