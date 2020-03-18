"""
Saves new csv for each hyperparameter combination. Tracks interneuron correlations with
increasing amounts of ganglion cell data.
"""
import torchdeepretina as tdr
import torch
import numpy as np
import os
import scipy.stats
from torchdeepretina.intracellular import make_intr_cor_maps
from torchdeepretina.models import *
import pyret.filtertools as ft
import time
import pickle
from queue import Queue
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pandas as pd

if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")

def plot_rf(sta, save_name):
    """
    sta: ndarray (C,H,W)
    """
    spat, temp = ft.decompose(sta)
    fig=plt.figure(figsize=(9,6), dpi= 80, facecolor='w', edgecolor='k')
    plt.clf()
    gridspec.GridSpec(6,1)

    plt.subplot2grid((6,1),(5,0))
    plt.plot(temp,'k',linewidth=5)
    plt.plot(np.arange(len(temp)), np.zeros(len(temp)), 'k--',linewidth=1)
    ax = plt.gca()
    ax.set_xticks([])
    ax.set_xticklabels([])
    ax.set_yticks([])
    ax.set_yticklabels([])
    for k in ax.spines.keys():
        ax.spines[k].set_visible(False)

    plt.subplot2grid((6,1),(0,0),rowspan=5)
    plt.imshow(spat.squeeze(), cmap = 'seismic', 
                    clim=[-np.max(abs(spat)), np.max(abs(spat))])

    plt.savefig(save_name)

def plot_real_rfs(rfs, save_root="realrfs", save_ext=".png"):
    """
    rfs: dict
        keys: str cell_file
        vals: dict
            keys: str stim_type
            vals: ndarray (N, C, H, W)
                the rfs should be a numpy array with neurons as the first dimension and the 
                corresponding sta for the rest of the dimensions
    """
    for cell_file in rfs.keys():
        cell_file_name = cell_file.split("/")[-1].split(".")[0]
        for stim_type in rfs[cell_file].keys():
            for i,sta in enumerate(rfs[cell_file][stim_type]):
                save_name = "{}_{}_{}_{}{}".format(save_root, cell_file_name, stim_type, i,
                                                                                  save_ext)
                plot_rf(sta, save_name)

def plot_model_rfs(rfs, save_root="modelrfs", save_ext=".png"):
    """
    rfs: dict
        keys: str cell_file
        vals: dict
            keys: str stim_type
            vals: ndarray (N, C, H, W)
                the rfs should be a numpy array with neurons as the first dimension and the 
                corresponding sta for the rest of the dimensions
    """
    for tup in rfs.keys():
        sta = rfs[tup]
        layer,chan,row,col = tup
        save_name = "{}_{}_{}_{}_{}{}".format(save_root, layer,chan,row,col, save_ext)
        plot_rf(sta, save_name)

if __name__=="__main__":
    torch.manual_seed(0)
    files = ["bipolars_early_2012.h5", "bipolars_late_2012.h5",
                'amacrines_early_2012.h5', 'amacrines_late_2012.h5']
    layers = ['sequential.0', 'sequential.6']
    path_to_model = "~/src/torch-deep-retina/models/15-10-07_naturalscene.pt" 
    model = tdr.analysis.read_model_file(path_to_model)
    model.to(DEVICE)
    model.eval()

    with torch.no_grad():
        new_cors, real_rfs, model_rfs = tdr.analysis.get_intr_cors(model, layers=layers,
                                                              files=files, ret_rfs=True,
                                                              verbose=True)
    plot_real_rfs(rfs=real_rfs, save_root="wholereals")
    plot_model_rfs(rfs=model_rfs, save_root="wholemodel")
    



