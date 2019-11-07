"""
This script is made to automate the analysis of the model performance for a batch of models.
You must give a command line argument of the model search folder to be analyzed.

$ python3 search_analysis.py bncnn

"""
import numpy as np
import torch
import torch.nn as nn
import h5py as h5
import os
import sys
import pickle
from torchdeepretina.models import *
import matplotlib.pyplot as plt
from torchdeepretina.datas import loadexpt
from torchdeepretina.physiology import Physio
import torchdeepretina.intracellular as intracellular
import torchdeepretina.retinal_phenomena as rp
import torchdeepretina.stimuli as stimuli
import torchdeepretina.analysis as analysis
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

if __name__ == "__main__":
    start_idx = None
    if len(sys.argv) >= 2:
        try:
            start_idx = int(sys.argv[1])
            grand_folders = sys.argv[2:]
        except:
            grand_folders = sys.argv[1:]
    torch.cuda.empty_cache()
    for grand_folder in grand_folders:
        print("Analyzing", grand_folder)
        dfs = analysis.analysis_pipeline(grand_folder, make_figs=True, verbose=True)
        for k in dfs.keys():
            dfs[k].to_csv(os.path.join(grand_folder,k), sep="!", index=False, header=True)



