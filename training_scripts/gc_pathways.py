import torch
import torchdeepretina as tdr
import numpy as np
import sys
import os
import pandas as pd


if __name__=="__main__":
    model_paths = sys.argv[1:]
    same_model = False
    if len(model_paths) == 1:
        model_paths.append(model_paths[0])
        same_model = True
    save_file = ""
    for mp in model_paths:
        chkpt = tdr.io.load_checkpoint(mp)
        save_file += chkpt['exp_name']+str(chkpt['exp_num'])+"_"
    save_file += "gcpathsim.csv"

    n_samples = 5000
    sim_type = 'dot'
    model1 = tdr.io.load_model(model_paths[0])
    model1.eval()
    model2 = tdr.io.load_model(model_paths[1])
    model2.eval()

    stim = tdr.datas.loadexpt('15-10-07', 'all', 'naturalscene',
            'train', history=model1.img_shape[0])
    perm = np.random.permutation(len(stim.X)).astype(np.int)
    stim = stim.X[perm[:n_samples]]
    sim_df = tdr.analysis.pathway_similarities(model1, model2,
                                                stim=stim,
                                                sim_type=sim_type,
                                                same_model=same_model,
                                                save_file=save_file)
