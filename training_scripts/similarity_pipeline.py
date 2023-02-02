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

def get_chan_count(df, path,total=True):
    idx = df['save_folder']==path
    chan_str = df.loc[idx]
    if len(chan_str) > 0:
        chan_str = chan_str.iloc[0]['chans'][1:-1]
        chans = chan_str.split(", ")[:2]
        if total:
            return np.sum([int(c) for c in chans])
        return tuple([int(c) for c in chans])
    print(path,"not found in model_data.csv")
    return -1

if __name__=="__main__":
    n_samples = 5000 # Number of samples used for comparison
    window_lim = 5 # Half the size of the window centered on the comparison location
    verbose = True
    table_checkpts = True
    grad_fit = False # Determines if you would like to compute the permuted similarity via grad descent
    batch_size = 1000
    sim_folder = "similarity_csvs" # Folder to save comparison csv to
    save_ext = "similarities.csv"
    store_act_mtx = False # limits memory consumption if false
    max_mtx_storage = None # Limits mem consumption if not None
    same_chans_only = False # If true, models are only compared with models with the same total channel counts. Does not work for pruned models.
    total_chans = False # Only active if same_chans_only is true. If False, only compares models with exact same channel counts
    all_chans = True # ONly applies for max_act, collects max sim for all model2 channels
    abs_val = True # Uses the max absolute value of the correlation to select location of correlation

    sim_type_dict = dict()
    sim_type_dict['max_act']  = True
    sim_type_dict['max_ig']   = False
    sim_type_dict['perm_act'] = False
    sim_type_dict['perm_ig']  = False

    if all_chans:
        save_ext = "chans_"+save_ext
        if abs_val:
            save_ext = "abs_"+save_ext

    if same_chans_only:
        save_ext = "same_"+save_ext
    else:
        save_ext = "all_"+save_ext

    if not os.path.exists(sim_folder):
        os.mkdir(sim_folder)
    grand_folders = sys.argv[1:]
    torch.cuda.empty_cache()
    model_info_df = pd.DataFrame()
    comp_paths = None
    for i,grand_folder in enumerate(grand_folders):
        print("Analyzing", grand_folder)
        paths = tdr.io.get_model_folders(grand_folder)
        paths = [os.path.join(grand_folder,path) for path in paths]
        if i == 0:
            model_paths = paths
        elif i==1:
            comp_paths = paths
        else:
            comp_paths = comp_paths + paths
        temp_path = os.path.join(grand_folder,"model_data.csv")
        if os.path.exists(temp_path):
            temp = pd.read_csv(temp_path,sep="!")
            model_info_df = model_info_df.append(temp,sort=True)
    if comp_paths is None: comp_paths = model_paths
    #save_file = [grand_folder.replace("/","") + "_" + save_ext for 
    save_file = [gfold.replace("/","") + "_" for gfold in grand_folders]
    save_file = "_".join(save_file) + save_ext
    save_file = os.path.join(sim_folder,save_file).replace("__", "_")
    print("Models:")
    print("\n".join(model_paths))
    print("Comp Models:")
    print("\n".join(comp_paths))
    print("Saving to:", save_file)
    print("\n".join([k+": "+str(v) for k,v in sim_type_dict.items()]))

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
        if same_chans_only:
            tot_chans1 = get_chan_count(model_info_df, model_paths[i],
                                                       total=total_chans)
        # Check if model has already been compared to all other models
        idx = (main_df['model1']==model_paths[i])
        prev_comps = set(main_df.loc[idx,'model2'])
        missing_comp = False
        for path in comp_paths:
            if path not in prev_comps:
                if same_chans_only and "save_folder" in model_info_df:
                    tot_chans = get_chan_count(model_info_df, path,
                                                  total=total_chans)
                    if tot_chans1 == tot_chans:
                        missing_comp = True
                        break
                else:
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
        model1 = tdr.pruning.reduce_model(model1,model1.zero_dict)
        m1_chans = np.sum(model1.chans[:2])
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
        calc_ig = sim_type_dict['perm_ig'] or sim_type_dict['max_ig']
        calc_ig = ig_resp1 is None and calc_ig
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
        if max_mtx_storage is None or len(ig_vecs)<max_mtx_storage:
            if store_act_mtx:
                act_vecs[model_paths[i]] = act_resp1
            ig_vecs[model_paths[i]] = ig_resp1
        model1 = model1.cpu()

        for j in range(len(comp_paths)):
            idx = (main_df['model1']==model_paths[i])
            if comp_paths[j] in set(main_df.loc[idx,"model2"]):
                print("Skipping", comp_paths[j],"due to previous record")
                continue
            if verbose:
                s = "Comparing: {} to {} | {} comparisons left"
                s = s.format(model_paths[i],comp_paths[j],
                                        (len(model_paths)-i)*len(comp_paths))
                print(s)

            if same_chans_only:
                tot_chans = get_chan_count(model_info_df, comp_paths[j],
                                                       total=total_chans)
                if tot_chans != tot_chans1:
                    print("Skipping", comp_paths[j],"due to diff chan count")
                    continue

            torch.cuda.empty_cache()
            model2 = tdr.io.load_model(comp_paths[j])
            model2 = tdr.pruning.reduce_model(model2, model2.zero_dict)
            m2_chans = np.sum(model2.chans[:2])
            if same_chans_only and (m2_chans != m1_chans):
                print("Skipping due to different channel counts")
                continue
            model2.eval()
            model2_layers = tdr.utils.get_conv_layer_names(model2)
            if verbose:
                print("Computing Model2 Responses")
            act_resp2 = None
            ig_resp2 = None
            if comp_paths[j] in act_vecs:
                act_resp2 = act_vecs[comp_paths[j]]
            if comp_paths[j] in ig_vecs:
                ig_resp2 = ig_vecs[comp_paths[j]]
            calc_act = act_resp2 is None
            calc_ig = sim_type_dict['perm_ig'] or sim_type_dict['max_ig']
            calc_ig = ig_resp2 is None and calc_ig
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
                act_resp2 = act if calc_act else act_resp2
                ig_resp2 = ig if calc_ig else ig_resp2
            if max_mtx_storage is None or len(ig_vecs)<max_mtx_storage:
                if store_act_mtx:
                    act_vecs[comp_paths[j]] = act_resp2
                ig_vecs[comp_paths[j]] = ig_resp2
            model2 = model2.cpu()
            model1_shapes = [act_resp1[l].shape for l in model1_layers]
            model2_shapes = [act_resp2[l].shape for l in model2_layers]

            ################### Max Correlations
            if sim_type_dict['max_act']:
                stats_string = "\nMax Sims\n"
                torch.cuda.empty_cache()
                model1_shapes = [act_resp1[l].shape for l in model1_layers]
                model2_shapes = [act_resp2[l].shape for l in model2_layers]
                # Activation Max Correlation
                if verbose:
                    print("Beginning Max Activation Correlations")
                stats_string += "\nActivation Correlations:\n"
                for l1,s1 in zip(model1_layers,model1_shapes):
                    s1 = s1[-1]//2 if len(s1)>2 else None
                    if s1 is None: continue
                    resp1 = act_resp1[l1][:,:,s1,s1].squeeze()
                    for l2,s2 in zip(model2_layers,model2_shapes):
                        s2 = s2[-1]//2 if len(s2)>2 else None
                        if s2 is None: continue
                        print("s2:", s2)
                        torch.cuda.empty_cache()
                        lim = window_lim
                        cutout1 = tdr.stimuli.get_cutout(act_resp1[l1],
                                                        center=(s1,s1),
                                                        span=2*lim+1,
                                                        pad_to=0)
                        flat_cutout1 = cutout1.reshape(len(cutout1),-1)

                        #resp2 = act_resp2[l2][:,:,s2,s2].squeeze()
                        cutout2 = tdr.stimuli.get_cutout(act_resp2[l2],
                                                    center=(s2,s2),
                                                    span=2*lim+1,
                                                    pad_to=0)
                        og_shape = cutout2.shape
                        flat_cutout2 = cutout2.reshape(len(cutout2),-1)
                        sim1 = mtx_cor(resp1,flat_cutout2,to_numpy=True)
                        #sim2 = mtx_cor(resp2,flat_cutout1,to_numpy=True)
                        # same model and layer
                        if model_paths[i]==comp_paths[j] and l1==l2: 
                            n_chans = cutout1.shape[-3]
                            inc = cutout1.shape[-2]*cutout1.shape[-1]
                            for chan in range(n_chans):
                                sim1[chan,inc*chan:inc*chan+inc] = 0
                                #sim2[chan,inc*chan:inc*chan+inc] = 0
                        if all_chans:
                            sim1 = sim1.reshape(len(sim1),og_shape[1],-1)
                            if abs_val:
                                args = np.argmax(np.abs(sim1), axis=-1)
                                max_sims1 = np.take_along_axis(
                                    sim1,args[...,None],axis=-1
                                )
                                max_sims1 = max_sims1[...,0]
                            else:
                                max_sims1 = np.max(sim1,axis=-1)
                            #max_sims2 = np.max(sim2,axis=-1)
                            for chan in range(len(max_sims1)):
                                for chan2 in range(sim1.shape[1]):
                                    table['model1'].append(model_paths[i])
                                    table['model2'].append(comp_paths[j])
                                    table['cor_type'].append("max_act")
                                    table['m1_layer'].append(l1)
                                    table['m2_layer'].append(l2)
                                    table['xy_coord'].append((s1,s1))
                                    table['chan1'].append(chan)
                                    table['chan2'].append(chan2)
                                    table['cor'].append(max_sims1[chan,chan2])
                            stats_string += "{}-{}: {}\n".format(l1, l2,
                                                           np.mean(max_sims1))
                        else:
                            max_sims1 = np.max(sim1,axis=-1)
                            arg_maxes = np.argmax(sim1, axis=-1)
                            #max_sims2 = np.max(sim2,axis=-1)
                            for chan in range(len(max_sims1)):
                                table['model1'].append(model_paths[i])
                                table['model2'].append(comp_paths[j])
                                table['cor_type'].append("max_act")
                                table['m1_layer'].append(l1)
                                table['m2_layer'].append(l2)
                                table['xy_coord'].append((s1,s1))
                                table['chan1'].append(chan)
                                table['chan2'].append(
                                  np.unravel_index(arg_maxes, og_shape[1:])[0][0]
                                )
                                table['cor'].append(max_sims1[chan])
                            stats_string += "{}-{}: {}\n".format(l1, l2,
                                                           np.mean(max_sims1))
                        #if model_paths[i]!=comp_paths[j]:
                        #    for chan in range(len(max_sims2)):
                        #        table['model1'].append(comp_paths[j])
                        #        table['model2'].append(model_paths[i])
                        #        table['cor_type'].append("max_act")
                        #        table['m1_layer'].append(l2)
                        #        table['m2_layer'].append(l1)
                        #        table['xy_coord'].append((s2,s2))
                        #        table['chan1'].append(chan)
                        #        table['chan2'].append(None)
                        #        table['cor'].append(max_sims2[chan])
                        #    s = "rev {}-{}: {}\n".format(l2, l1,
                        #                           np.mean(max_sims2))
                        #    stats_string += s

            # Integrated Gradient Max Correlation
            torch.cuda.empty_cache()
            if sim_type_dict['max_ig']:
                if verbose:
                    print("Beginning Max IG Correlations")
                stats_string = "IG Correlations:\n"
                model1_shapes = [ig_resp1[l].shape for l in model1_layers]
                model2_shapes = [ig_resp2[l].shape for l in model2_layers]
                for l1,s1 in zip(model1_layers,model1_shapes):
                    if len(s1)<=2:
                        continue
                    s1 = s1[-1]//2
                    for l2,s2 in zip(model2_layers,model2_shapes):
                        if len(s2)<=2:
                            continue
                        s2 = s2[-1]//2
                        lim = window_lim
                        resp1 = ig_resp1[l1][:,:,s1,s1].squeeze()
                        resp2 = ig_resp2[l2][:,:,s2,s2].squeeze()
                        cutout1 = tdr.stimuli.get_cutout(ig_resp1[l1],
                                                        center=(s1,s1),
                                                        span=2*lim+1,
                                                        pad_to=0)
                        flat_cutout1 = cutout1.reshape(len(cutout1),-1)
                        cutout2 = tdr.stimuli.get_cutout(ig_resp2[l2],
                                                        center=(s2,s2),
                                                        span=2*lim+1,
                                                        pad_to=0)
                        og_shape = cutout2.shape
                        flat_cutout2 = cutout2.reshape(len(cutout2),-1)
                        sim1 = mtx_cor(resp1,flat_cutout2,to_numpy=True)
                        #sim2 = mtx_cor(resp2,flat_cutout1,to_numpy=True)
                        if i==j and l1==l2: # same model and layer
                            n_chans = cutout1.shape[-3]
                            inc = cutout1.shape[-2]*cutout1.shape[-1]
                            for chan in range(n_chans):
                                sim1[chan,inc*chan:inc*chan+inc] = 0
                                #sim2[chan,inc*chan:inc*chan+inc] = 0
                        max_sims1 = np.max(sim1,axis=-1)
                        arg_maxes = np.argmax(sim1, axis=-1)
                        #max_sims2 = np.max(sim2,axis=-1)
                        for chan in range(len(max_sims1)):
                            table['model1'].append(model_paths[i])
                            table['model2'].append(comp_paths[j])
                            table['cor_type'].append("max_ig")
                            table['m1_layer'].append(l1)
                            table['m2_layer'].append(l2)
                            table['xy_coord'].append((s1,s1))
                            table['chan1'].append(chan)
                            table['chan2'].append(
                              np.unravel_index(arg_maxes, og_shape[1:])[0][0]
                            )
                            table['cor'].append(max_sims1[chan])
                        stats_string += "{}-{}: {}\n".format(l1, l2,
                                                       np.mean(max_sims1))
                        #if model_paths[i]!=comp_paths[j]:
                        #    for chan in range(len(max_sims2)):
                        #        table['model1'].append(comp_paths[j])
                        #        table['model2'].append(model_paths[i])
                        #        table['cor_type'].append("max_ig")
                        #        table['m1_layer'].append(l2)
                        #        table['m2_layer'].append(l1)
                        #        table['xy_coord'].append((s2,s2))
                        #        table['chan1'].append(chan)
                        #        table['chan2'].append(None)
                        #        table['cor'].append(max_sims2[chan])
                        #    s = "rev {}-{}: {}\n".format(l2, l1,
                        #                           np.mean(max_sims2))
                        #    stats_string += s
            ########### Permutation correlations
            torch.cuda.empty_cache()
            if sim_type_dict['perm_act']:
                stats_string = "\nPermutation Sims\n"
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
                            start = s2-lim
                            end = s2+lim+1
                            resp2 = act_resp2[l2][:,:,start:end,start:end]
                            sim = perm_similarity(resp1,resp2,
                                                  grad_fit=grad_fit,
                                                  vary_space=True,
                                                  verbose=False)



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
                        table['model2'].append(comp_paths[j])
                        table['cor_type'].append("perm_act")
                        table['m1_layer'].append(l1)
                        table['m2_layer'].append(l2)
                        table['chan1'].append(None)
                        table['chan2'].append(None)
                        table['xy_coord'].append(best_xy)
                        table['cor'].append(best_sim)
                        stats_string += "{}-{}: {}\n".format(l1, l2,
                                                             best_sim)

            # Integrated Gradient Perm Correlation
            torch.cuda.empty_cache()
            if sim_type_dict['perm_ig']:
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
                        table['model2'].append(comp_paths[j])
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
            max_mem_used = max_mem_used.ru_maxrss/1024/1024
            print("Memory Used: {:.2f} mb".format(max_mem_used))
            gpu_mem = tdr.utils.get_gpu_mem()
            s = ["gpu{}: {}".format(k,v) for k,v in gpu_mem.items()]
            s = "\n".join(s)
            print(s)
            print()
        if store_act_mtx and model_paths[i] in act_vecs:
            del act_vecs[model_paths[i]]
        if model_paths[i] in ig_vecs:
            del ig_vecs[model_paths[i]]
