import numpy as np
import torch
import torch.nn as nn
import h5py as h5
import os
import sys
import pickle
from torchdeepretina.models import *
import matplotlib.pyplot as plt
from torchdeepretina.deepretina_loader import loadexpt
from torchdeepretina.physiology import Physio
import torchdeepretina.intracellular as intracellular
import torchdeepretina.batch_compute as bc
import torchdeepretina.retinal_phenomena as rp
import torchdeepretina.stimuli as stimuli
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

#If you want to use stimulus that isnt just boxes
def prepare_stim(stim, stim_type):
    if stim_type == 'boxes':
        return 2*stim - 1
    elif stim_type == 'flashes':
        stim = stim.reshape(stim.shape[0], 1, 1)
        return np.broadcast_to(stim, (stim.shape[0], 38, 38))
    elif stim_type == 'movingbar':
        stim = block_reduce(stim, (1,6), func=np.mean)
        stim = pyret.stimulustools.upsample(stim.reshape(stim.shape[0], stim.shape[1], 1), 5)[0]
        return np.broadcast_to(stim, (stim.shape[0], stim.shape[1], stim.shape[1]))
    elif stim_type == 'lines':
        stim_averaged = np.apply_along_axis(lambda m: np.convolve(m, 0.5*np.ones((2,)), mode='same'), 
                                            axis=1, arr=stim)
        stim = stim_averaged[:,::2]
        # now stack stimulus to convert 1d to 2d spatial stimulus
        return stim.reshape(-1,1,stim.shape[-1]).repeat(stim.shape[-1], axis=1)
    else:
        print("Invalid stim type")
        assert False
    
def index_of(arg, arr):
    for i in range(len(arr)):
        if arg == arr[i]:
            return i
    return -1

def make_data_frame(model_stats, headers):
    data = dict()
    for header in headers:
        data[header] = []
    for folder in model_stats.keys():
        untouched_keys = set(data.keys())
        for k in model_stats[folder].keys():
            if k in data:
                data[k].append(model_stats[folder][k])
                if k in untouched_keys:
                    untouched_keys.remove(k)
        with open("../training_scripts/" + folder+"/hyperparams.txt") as f:
            architecture = []
            for i,line in enumerate(f):
                if "(" in line or ")" in line:
                    l = line.replace("\n", "#")
                    architecture.append(l)
                else:
                    splt_line = line.strip().split(":")
                    if len(splt_line) == 2 and splt_line[0].strip() in data:
                        data[splt_line[0].strip()].append(splt_line[1].strip())
                        if splt_line[0].strip() in untouched_keys:
                            untouched_keys.remove(splt_line[0].strip())
            data['architecture'].append("".join(architecture))
            untouched_keys.remove('architecture')
        for k in list(untouched_keys):
            data[k].append(None)
    return pd.DataFrame(data)

def load_model(folder, pth):
    try:
        hyps=get_hyps(folder)
        hyps['model_type'] = hyps['model_type'].split(".")[-1].split("\'")[0].strip()
        hyps['model_type'] = globals()[hyps['model_type']]
        bn_cnn = hyps['model_type'](**temp['model_hyps'])
    except Exception as e:
        model_hyps = {"n_units":5,"noise":float(hyps['noise'])}
        if "bias" in hyps:
            model_hyps['bias'] = bool(hyps['bias'])
        if "chans" in hyps:
            model_hyps['chans'] = [int(x) for x in 
                                   hyps['chans'].replace("[", "").replace("]", "").strip().split(",")]
        if "adapt_gauss" in hyps:
            model_hyps['adapt_gauss'] = hyps['adapt_gauss']
        fn_args = set(hyps['model_type'].__init__.__code__.co_varnames)
        for k in model_hyps.keys():
            if k not in fn_args:
                del model_hyps[k]
        bn_cnn = hyps['model_type'](**model_hyps)
    return bn_cnn

def get_hyps(folder):
    hyps = dict()
    with open(os.path.join(folder, "hyperparams.txt")) as f:
        for line in f:
            if "(" not in line and ")" not in line:
                splt = line.strip().split(":")
                if len(splt) > 1:
                    hyps[splt[0]] = splt[1].strip()
    return hyps
