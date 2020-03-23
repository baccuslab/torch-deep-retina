"""
This script calculates the correlation between all of the channels 
of different models. All channels from the first model at the spatial
center of the activation/integrated gradient tensors are correlated
with all channels of the second model at different spatial locations
centered around the spatial location of the first model. All values
in the correlation matrix are recorded for the location with the best
correlations. The metric used for best correlation is symmetric in that
it is the average of all max correlations taken along rows and columns
of the correlation matrix. When comparing a model with itself, the
diagonal is set to -1 to avoid the obvious perfect correlation.

This script also calculates the permutation similarity between all of the
channels of different models. All channels from the first model at the
spatial center of the activation/integrated gradient tensors are
correlated with all channels of the second model at different spatial
locations centered around the spatial location of the first model. The
permutation similarity for the best spatial location is recorded for
each model pair.

This script can be used by arguing experiment folders like so:

$ python3 intra_sim_pipeline.py <path_to_exp_folder1_here> <path_to_exp_folder2_here>

Each model from each of the experiment folders is correlated pairwise
"""
import torch
import pandas as pd
import torchdeepretina as tdr
from torchdeepretina.utils import perm_similarity,mtx_cor
import numpy as np
import h5py as h5
import matplotlib.pyplot as plt
import time
import sys
import os
import gc
import resource

DEVICE = torch.device("cuda:0")

if __name__=="__main__":
    n_samples = 5000 # Number of samples used for comparison
    window_lim = 5 # Half the size of the window centered on the comparison location
    verbose = True
    table_checkpts = True
    batch_size = 1000
    sim_folder = "similarity_csvs" # Folder to save comparison csv to
    save_ext = "all_similarities.csv"
    store_act_mtx = False # limits memory consumption if false
    max_mtx_storage = None # Limits mem consumption if not None

    if not os.path.exists(sim_folder):
        os.mkdir(sim_folder)
    grand_folders = sys.argv[1:]
    torch.cuda.empty_cache()
    model_paths = None
    for grand_folder in grand_folders:
        print("Analyzing", grand_folder)
        paths = tdr.io.get_model_folders(grand_folder)
        paths = [os.path.join(grand_folder,path) for path in paths]
        if model_paths is None:
            model_paths = paths
        else:
            model_paths = model_paths + paths
        save_file = grand_folder.replace("/","") + "_" + save_ext
    save_file = os.path.join(sim_folder,save_file)
    print("Models:")
    print("\n".join(model_paths))
    print("Saving to:", save_file)

    table = {
        "model1":[],
        "model2":[],
        "m1_layer":[],
        "m2_layer":[],
        "cor_type":[],
        "xy_coord":[],
        "chan1":[],
        "chan2":[],
        "cor":[],
    }
    main_df = pd.DataFrame(table)
    if os.path.exists(save_file):
        main_df = pd.read_csv(save_file, sep="!")
    torch.cuda.empty_cache()
    data = tdr.datas.loadexpt("15-10-07", "all", "naturalscene",
                                            'train', history=0)
    stim = data.X[:n_samples]

    act_vecs = dict()
    ig_vecs = dict()

    for i in range(len(model_paths)):
        # Check if model has already been compared to all other models
        idx = (main_df['model1']==model_paths[i])
        prev_comps = set(main_df.loc[idx,'model2'])
        missing_comp = False
        for path in model_paths:
            if path not in prev_comps:
                missing_comp = True
                break
        if not missing_comp:
            print("Skipping", model_paths[i],
                    "due to previous records")
            continue
        if verbose:
            print("\nBeginning model:", model_paths[i],"| {}/{}".format(
                                                  i,len(model_paths)))
            print()

        ############### Prep
        model1 = tdr.io.load_model(model_paths[i])
        model1.eval()
        model1_layers = tdr.utils.get_conv_layer_names(model1)
        if verbose:
            print("Computing Model1 Responses")
        act_resp1 = None
        ig_resp1 = None
        if model_paths[i] in act_vecs:
            act_resp1 = act_vecs[model_paths[i]]
        if model_paths[i] in ig_vecs:
            ig_resp1 = ig_vecs[model_paths[i]]

        calc_act = act_resp1 is None
        calc_ig = ig_resp1 is None
        model1.to(DEVICE)
        with torch.no_grad():
            loc = (model1.shapes[-1][0]//2, model1.shapes[-1][1]//2)
            loc = None if not model1.convgc else loc
            act, ig = tdr.analysis.get_resps(model1,
                                             stim,
                                             model1_layers,
                                             act=calc_act,
                                             ig=calc_ig,
                                             batch_size=batch_size,
                                             ig_spat_loc=loc,
                                             to_numpy=True,
                                             verbose=verbose)
            act_resp1 = act if calc_act else act_resp1
            ig_resp1 = ig if calc_ig else ig_resp1
        if max_mtx_storage is not None and len(ig_vecs)<max_mtx_storage:
            if store_act_mtx:
                act_vecs[model_paths[i]] = act_resp1
            ig_vecs[model_paths[i]] = ig_resp1
        model1 = model1.cpu()

        for j in range(i,len(model_paths)):
            idx = (main_df['model1']==model_paths[i])
            if model_paths[j] in set(main_df.loc[idx,"model2"]):
                print("Skipping", model_paths[j],"due to previous record")
                continue
            if verbose:
                s = "Comparing: {} to {} | {} comparisons left"
                s = s.format(model_paths[i],model_paths[j],
                                        len(model_paths)-j)
                print(s)

            torch.cuda.empty_cache()
            model2 = tdr.io.load_model(model_paths[j])
            model2.eval()
            model2_layers = tdr.utils.get_conv_layer_names(model2)
            if verbose:
                print("Computing Model2 Responses")
            act_resp2 = None
            ig_resp2 = None
            if model_paths[j] in act_vecs:
                act_resp2 = act_vecs[model_paths[j]]
            if model_paths[j] in ig_vecs:
                ig_resp2 = ig_vecs[model_paths[j]]
            calc_act = act_resp2 is None
            calc_ig = ig_resp2 is None
            model2.to(DEVICE)
            with torch.no_grad():
                loc =(model2.shapes[-1][0]//2,model2.shapes[-1][1]//2)
                loc = None if not model2.convgc else loc
                act, ig = tdr.analysis.get_resps(model2,
                                                 stim,
                                                 model2_layers,
                                                 act=calc_act,
                                                 ig=calc_ig,
                                                 batch_size=batch_size,
                                                 ig_spat_loc=loc,
                                                 to_numpy=True,
                                                 verbose=verbose)
            if max_mtx_storage is None or len(ig_vecs)<=max_mtx_storage:
                if store_act_mtx:
                    act_vecs[model_paths[j]] = act_resp2
                ig_vecs[model_paths[j]] = ig_resp2
            model2 = model2.cpu()
            model1_shapes = [act_resp1[l].shape for l in model1_layers]
            model2_shapes = [act_resp2[l].shape for l in model2_layers]

            ################### Channel Correlations
            stats_string = "\nChannel Sims\n"
            torch.cuda.empty_cache()
            # Activation Max Correlation
            if verbose:
                print("Beginning Channel Activation Correlations")
            stats_string += "\nActivation Correlations:\n"
            for l1,s1 in zip(model1_layers,model1_shapes):
                s1 = s1[-1]//2 if len(s1)>2 else None
                resp1 = act_resp1[l1][:,:,s1,s1].squeeze()
                for l2,s2 in zip(model2_layers,model2_shapes):
                    s2 = s2[-1]//2 if len(s2)>2 else None
                    torch.cuda.empty_cache()
                    best_sim = None
                    if s2 is not None:
                        lim = window_lim
                        for x in range(s2-lim,s2+lim+1):
                            for y in range(s2-lim,s2+lim+1):
                                resp2 = act_resp2[l2][:,:,x,y].squeeze()
                                sims = mtx_cor(resp1,resp2,to_numpy=True)
                                if i == j and l1==l2: # Zero the obvious
                                    rang = range(len(sims))
                                    sims[rang,rang] = -1
                                sims[np.isnan(sims)] = 0
                                temp = [sims.max(0),sims.max(1)]
                                temp = np.concatenate(temp)
                                sim = np.mean(temp)
                                if best_sim is None or sim > best_sim\
                                                or np.isnan(best_sim):
                                    best_sim = sim
                                    best_sims = sims
                                    best_xy = (x,y)
                    else:
                        resp2 = act_resp2[l2].squeeze()
                        sims = mtx_cor(resp1,resp2)
                        if i == j and l1==l2: # Zero the obvious
                            rang = range(len(sims))
                            sims[rang,rang] = -1
                        sims[np.isnan(sims)] = 0
                        temp = [sims.max(0),sims.max(1)]
                        temp = np.concatenate(temp)
                        best_sim = np.mean(temp)
                        best_xy = (0,0)
                    for chan1 in range(len(sims)):
                        for chan2 in range(len(sims[chan1])):
                            table['model1'].append(model_paths[i])
                            table['model2'].append(model_paths[j])
                            table['cor_type'].append("chan_act")
                            table['m1_layer'].append(l1)
                            table['m2_layer'].append(l2)
                            table['xy_coord'].append(best_xy)
                            table['chan1'].append(chan1)
                            table['chan2'].append(chan2)
                            table['cor'].append(sims[chan1,chan2])
                    stats_string += "{}-{}: {}\n".format(l1, l2,
                                                         best_sim)

            if verbose:
                print(stats_string)
            # Integrated Gradient Max Correlation
            torch.cuda.empty_cache()
            if verbose:
                print("Beginning Channel IG Correlations")
            stats_string = "IG Correlations:\n"
            model1_shapes = [ig_resp1[l].shape for l in model1_layers]
            model2_shapes = [ig_resp2[l].shape for l in model2_layers]
            for l1,s1 in zip(model1_layers,model1_shapes):
                if len(s1)<=2:
                    continue
                s1 = s1[-1]//2
                resp1 = ig_resp1[l1][:,:,s1,s1].squeeze()
                for l2,s2 in zip(model2_layers,model2_shapes):
                    if len(s2)<=2:
                        continue
                    s2 = s2[-1]//2
                    torch.cuda.empty_cache()
                    best_sim = None
                    lim = window_lim
                    for x in range(s2-lim,s2+lim+1):
                        for y in range(s2-lim,s2+lim+1):
                            resp2 = ig_resp2[l2][:,:,x,y]
                            sims = mtx_cor(resp1,resp2)
                            if i == j and l1==l2: # Zero the obvious
                                rang = range(len(sims))
                                sims[rang,rang] = -1
                            sims[np.isnan(sims)] = 0
                            temp = [sims.max(0),sims.max(1)]
                            temp = np.concatenate(temp)
                            sim = np.mean(temp)
                            if best_sim is None or sim > best_sim\
                                                or np.isnan(best_sim):
                                best_sim = sim
                                best_sims = sims
                                best_xy = (x,y)
                    for chan1 in range(len(sims)):
                        for chan2 in range(len(sims[chan1])):
                            table['model1'].append(model_paths[i])
                            table['model2'].append(model_paths[j])
                            table['cor_type'].append("chan_ig")
                            table['m1_layer'].append(l1)
                            table['m2_layer'].append(l2)
                            table['xy_coord'].append(best_xy)
                            table['chan1'].append(chan1)
                            table['chan2'].append(chan2)
                            table['cor'].append(sims[chan1,chan2])
                    stats_string += "{}-{}: {}\n".format(l1, l2,
                                                         best_sim)
            if verbose:
                print(stats_string)
            ########### Permutation correlations
            stats_string = "\nPermutation Sims\n"
            torch.cuda.empty_cache()
            # Activation Perm Correlation
            if verbose:
                print("Beginning Permutation Activation Correlations")
            stats_string += "\nActivation Correlations:\n"
            for l1,s1 in zip(model1_layers,model1_shapes):
                s1 = s1[-1]//2 if len(s1)>2 else None
                resp1 = act_resp1[l1][:,:,s1,s1].squeeze()
                for l2,s2 in zip(model2_layers,model2_shapes):
                    s2 = s2[-1]//2 if len(s2)>2 else None
                    torch.cuda.empty_cache()
                    best_sim = None
                    if s2 is not None:
                        lim = window_lim
                        for x in range(s2-lim,s2+lim+1):
                            for y in range(s2-lim,s2+lim+1):
                                resp2 = act_resp2[l2][:,:,x,y]
                                sim = perm_similarity(resp1, resp2,
                                                      grad_fit=grad_fit,
                                                      verbose=False)
                                if best_sim is None or sim > best_sim\
                                            or np.isnan(best_sim):
                                    best_sim = sim
                                    best_xy = (x,y)
                    else:
                        resp2 = act_resp2[l2].squeeze()
                        best_sim = perm_similarity(resp1, resp2,
                                                   grad_fit=grad_fit,
                                                   verbose=False)
                        best_xy = (0,0)
                    table['model1'].append(model_paths[i])
                    table['model2'].append(model_paths[j])
                    table['cor_type'].append("perm_act")
                    table['m1_layer'].append(l1)
                    table['m2_layer'].append(l2)
                    table['chan1'].append(None)
                    table['chan2'].append(None)
                    table['xy_coord'].append(best_xy)
                    table['cor'].append(best_sim)
                    stats_string += "{}-{}: {}\n".format(l1, l2,
                                                         best_sim)
            if verbose:
                print(stats_string)
            # Integrated Gradient Perm Correlation
            torch.cuda.empty_cache()
            if verbose:
                print("Beginning Permutation IG Correlations")
            stats_string = "IG Correlations:\n"
            model1_shapes = [ig_resp1[l].shape for l in model1_layers]
            model2_shapes = [ig_resp2[l].shape for l in model2_layers]
            for l1,s1 in zip(model1_layers,model1_shapes):
                if len(s1)<=2:
                    continue
                s1 = s1[-1]//2
                resp1 = ig_resp1[l1][:,:,s1,s1].squeeze()
                for l2,s2 in zip(model2_layers,model2_shapes):
                    if len(s2)<=2:
                        continue
                    s2 = s2[-1]//2
                    torch.cuda.empty_cache()
                    best_sim = None
                    lim = window_lim
                    for x in range(s2-lim,s2+lim+1):
                        for y in range(s2-lim,s2+lim+1):
                            resp2 = ig_resp2[l2][:,:,x,y]
                            sim = perm_similarity(resp1, resp2,
                                                  grad_fit=grad_fit,
                                                  verbose=False)
                            if best_sim is None or sim > best_sim\
                                            or np.isnan(best_sim):
                                best_sim = sim
                                best_xy = (x,y)
                    table['model1'].append(model_paths[i])
                    table['model2'].append(model_paths[j])
                    table['cor_type'].append("perm_ig")
                    table['m1_layer'].append(l1)
                    table['m2_layer'].append(l2)
                    table['chan1'].append(None)
                    table['chan2'].append(None)
                    table['xy_coord'].append(best_xy)
                    table['cor'].append(best_sim)
                    stats_string += "{}-{}: {}\n".format(l1, l2,
                                                         best_sim)


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
            gpu_mem = tdr.utils.get_gpu_mem()
            s = ["gpu{}: {}".format(k,v) for k,v in gpu_mem.items()]
            s = "\n".join(s)
            print(s)
            print()
        del act_vecs[model_paths[i]]
        del ig_vecs[model_paths[i]]
