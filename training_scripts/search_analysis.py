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
from torchdeepretina.deepretina_loader import loadexpt
from torchdeepretina.physiology import Physio
import torchdeepretina.intracellular as intracellular
import torchdeepretina.batch_compute as bc
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

def normalize(x):
    return (x-x.mean())/(x.std()+1e-7)

def retinal_phenomena_figs(model):
    figs = []
    fig_names = []
    metrics = dict()

    (fig, (ax0,ax1)), X, resp = rp.step_response(model)
    figs.append(fig)
    fig_names.append("step_response")
    metrics['step_response'] = None

    (fig, (ax0,ax1)), X, resp, osr_resp_proportion = rp.osr(model, duration=1)
    figs.append(fig)
    fig_names.append("osr")
    metrics['osr'] = osr_resp_proportion 

    (fig, (ax0,ax1)), X, resp = rp.reversing_grating(model)
    figs.append(fig)
    fig_names.append("reversing_grating")
    metrics['reversing_grating'] = None

    (fig, (ax0,ax1)), envelope, responses = rp.contrast_adaptation(model, .35, .05)
    figs.append(fig)
    fig_names.append("contrast_adaptation")
    metrics['contrast_adaptation'] = None

    contrasts = [0.5, 1.2]
    unit = 0
    layer = "sequential."+str(len(model.sequential)-1)
    fig = rp.contrast_fig(model, contrasts, unit_index=unit, nonlinearity_type="bin")
    figs.append(fig)
    fig_names.append("contrast_fig")
    metrics['contrast_fig'] = None

    (fig, ax), (speed_left, speed_right), (rtl, resp_rtl), (ltr, resp_ltr), avg_resp = rp.motion_reversal(model)
    figs.append(fig)
    fig_names.append("motion_reversal")
    metrics['motion_reversal'] = None

    tup = rp.motion_anticipation(model)
    (fig, ax), (speed_left, speed_right), (c_right, stim_right, resp_right), (c_left, stim_left, resp_left), (flash_centers, flash_responses) = tup
    figs.append(fig)
    fig_names.append("motion_anticipation")
    #metrics['motion_anticipation'] = (symmetry, continuity, peak_height, right_anticipation, left_anticipation)
    metrics['motion_anticipation'] = None

    fig, diff_vid, global_vid, diff_response, global_response = rp.oms_random_differential(model)
    figs.append(fig)
    fig_names.append("oms")
    oms_ratios = global_response.mean(0)/diff_response.mean(0)
    metrics['oms'] = oms_ratios

    return figs, fig_names, metrics

def get_insp_layers(model, hyps):
    try:
        insp_layers = hyps['insp_layers']
        return insp_layers
    except KeyError as e:
        pass
    insp_layers = []
    for i, (name,module) in enumerate(model.named_modules()):
        if "BatchNorm" in module._get_name() and len(name) <= len("sequential.00"):
            insp_layers.append(name)
        if len(insp_layers) >= 2:
            return insp_layers
    return ['sequential.2', 'sequential.8']

def load_interneuron_data(root_path="~/interneuron_data/"):
    # Load data
    # num_pots stores the number of cells per stimulus
    # mem_pots stores the membrane potential
    # psst, you can find the "data" folder in /home/grantsrb on deepretina server
    files = ['bipolars_late_2012.h5', 'bipolars_early_2012.h5', 'amacrines_early_2012.h5', 
             'amacrines_late_2012.h5', 'horizontals_early_2012.h5', 'horizontals_late_2012.h5']
    files = [os.path.expanduser(root_path + name) for name in files]
    filter_length = 40
    window_size = 2
    num_pots = []
    stims = dict()
    mem_pots = dict()
    keys_to_use = {"boxes"}
    for fi in files:
        with h5.File(fi, 'r') as f:
            for k in f.keys():
                if k in keys_to_use:
                    if k not in stims:
                        stims[k] = []
                    if k not in mem_pots:
                        mem_pots[k] = []
                    try:
                        stims[k].append(analysis.prepare_stim(np.asarray(f[k+'/stimuli']), k))
                        mem_pots[k].append(np.asarray(f[k]['detrended_membrane_potential'])[:, filter_length:])
                    except Exception as e:
                        print(e)
                        print("stim error at", k)
            num = np.array(f['boxes/detrended_membrane_potential'].shape[0])
            num_pots.append(num)
    return num_pots, stims, mem_pots, files

def analyze_model(folder, interneuron_data, test_data=None, main_dir="../training_scripts/"):
    """
    Does the model analysis for the saved model.

    folder: string
        the name of the folder that contains the model save

    """
    starttime = time.time()
    # Read interneuron data
    # num_pots is the number of recorded potentials for each data file
    num_pots, stims, mem_pots, intrnrn_files = interneuron_data
    
    batch_compute_size = 500
    stats = dict()
    print("folder:", folder)
    hyps = analysis.get_hyps(folder)
    try:
        n_epochs = int(hyps['n_epochs'])
    except:
        n_epochs = 1
    # Prep
    losses = []
    val_losses = []
    val_accs = []
    model = None
    for i in range(n_epochs):
        f_name = os.path.join(main_dir, folder,"test_epoch_{0}.pth".format(i))
        try:
            with open(f_name, "rb") as fd:
                data = torch.load(fd)
            try:
                losses.append(data['loss'])
                val_losses.append(data['val_loss'])
                val_accs.append(data['val_acc'])
            except:
                losses.append(0)
                val_losses.append(0)
                val_accs.append(.5)
            # Remove legacy saving system
            if 'model' in data:
                model = data['model']
                del data['model']
                torch.save(data, f_name)
        except FileNotFoundError as e:
            pass
 
    if model is None:
        model = analysis.get_architecture(folder, data, hyps)
    insp_layers = get_insp_layers(model, hyps)
    print("Inspection layers:", insp_layers)
    
    # Load saved model parameters
    # Try-except blocks are used to accomodate some legacy models
    try:
        model.load_state_dict(data['model_state_dict'])
    except RuntimeError as e:
        print("Failed to load model state_dict, attempting fix")
        try:
            keys = list(data['model_state_dict'].keys())
            for k in keys:
                if "cuda_param" in k:
                    new_k = k.replace("cuda_param", "sigma")
                    data['model_state_dict'][new_k] = data['model_state_dict'][k]
                    del data['model_state_dict'][k]
            model.load_state_dict(data['model_state_dict'])
        except RuntimeError as e:
            print(e)
            print("Unable to load state dict, continuing")
            return stats
    
    # Add exponential layer depending on training regime
    poisson = ("lossfxn" not in hyps and "lossfxn_name" not in hyps) or\
              ("lossfxn" in hyps and hyps['lossfxn'] == "PoissonNLLLoss") or\
              ("lossfxn_name" in hyps and hyps['lossfxn_name'] == "PoissonNLLLoss")
    print("poissonloss:", poisson)
    log_poisson = "log_poisson" not in hyps or hyps['log_poisson'] == "True" or hyps['log_poisson'] == True
    print("log_poisson:", log_poisson)
    softplus = model.sequential[-1]._get_name() == "Softplus"
    noexp_seq = None
    if poisson and log_poisson and softplus:
        new_seq = [m for m in model.sequential] + [Exponential()]
        noexp_seq = model.sequential
        noexp_seq.eval()
        # Old model is model without final exponential
        # Models during inference have an exponential
        print("Old model:") 
        print(noexp_seq)
        model.sequential = nn.Sequential(*new_seq)

    model = model.to(DEVICE)
    model.eval()
    print(model)
    
    try:
        chans = [int(x) for x in hyps['chans'].replace("[","").replace("]","").split(",")]
    except KeyError as e:
        chans=[8,8]
    
    # Load data
    try:
        cells = hyps['cells']
        dataset = hyps['dataset']
        stim_type = hyps['stim_type']
    except KeyError as e:
        cells = "all"
        dataset = "15-10-07"
        stim_type = "naturalscene"
    try:
        norm_stats = [temp['norm_stats']['mean'], temp['norm_stats']['std']]
    except:
        norm_stats = [51.49175, 53.62663279042969]

    test_data = loadexpt(dataset,cells,stim_type,'test',40,0, norm_stats=norm_stats)
    test_x = torch.from_numpy(test_data.X)
    
    stats['FinalLoss'] = losses[-1]
    stats['FinalVal'] = val_losses[-1]
    stats['ValAcc'] = val_accs[-1]
    
    if(math.isnan(losses[-1]) or math.isnan(val_losses[-1]) or math.isnan(val_accs[-1])):
        print("NaN results, continuing...\n")
        return stats
    
    with torch.no_grad():
        model_response = bc.batch_compute_model_response(test_data.X, model, batch_compute_size, 
                                                     insp_keys=set(insp_layers))

    # Collect test data pearson correlation
    test_accs = [scipy.stats.pearsonr(model_response['output'][:, i], test_data.y[:, i])[0] 
                                                    for i in range(test_data.y.shape[-1])]
    avg_test_acc = np.mean(test_accs)
    for i, cell in enumerate(test_data.cells):
        stats["cell"+str(cell)] = test_accs[i]

    print("Final Test Acc:", avg_test_acc)
    stats['FinalTestAcc'] = avg_test_acc

    # Compare to non-exponentiated outputs
    if noexp_seq is not None: 
        noexp_modresp = bc.batch_compute_model_response(test_data.X, noexp_seq, batch_compute_size, 
                                                                                insp_keys=set())
        test_accs = [scipy.stats.pearsonr(noexp_modresp['output'][:, i], test_data.y[:, i])[0] 
                                                        for i in range(test_data.y.shape[-1])]
        noexp_avg_acc = np.mean(test_accs)
        print("Old test acc:", noexp_avg_acc)
        print("New minus old:", avg_test_acc - noexp_avg_acc)
        stats['OldTestAcc'] = noexp_avg_acc
        if math.isnan(avg_test_acc):
            avg_test_acc = noexp_avg_acc
        del noexp_modresp
        del noexp_seq
    

    if math.isnan(avg_test_acc):
        print("NaN results, continuing...\n")
        return stats

    if avg_test_acc <= .45:
        print("Results too low, continuing...\n")
        return stats
    
    # Plot firing rate sample
    fig = plt.figure()
    plt.plot(normalize(model_response['output'][:400, 0]))
    plt.plot(normalize(test_data.y[:400,0]), alpha=.7)
    plt.legend(["model", "data"])
    plt.title("Firing Rate")
    plt.savefig(os.path.join(folder, "firing_rate_sample.png"))
    #
    ## Plot loss curve
    #fig = plt.figure()
    #plt.plot(losses)
    #plt.plot(val_losses)
    #plt.legend(["TrainLoss", "ValLoss"])
    #plt.title("Loss Curves")
    #plt.savefig(os.path.join(folder, "loss_curve.png"))
    
    # Get retinal phenomena plots
    figs, fig_names, metrics = retinal_phenomena_figs(model)

    for fig, name in zip(figs, fig_names):
        save_name = name + ".png"
        fig.savefig(os.path.join(folder, save_name))
    oms_ratios = metrics['oms']
    for i, cell in enumerate(test_data.cells):
        stats["oms_ratio_cell"+str(cell)] = oms_ratios[i]
    stats['avg_oms_ratio'] = np.mean(oms_ratios)
    stats['std_oms_ratio'] = np.std(oms_ratios)
    
    print("Calculating interneuron model responses...")
    # Computes the model responses for each stimulus 
    # and interneuron type labels y_true (0 for bipolar, 1 for amacrine, 2 for horizontal)
    y_true = []
    filter_length = 40
    model_responses = dict()
    for i in tqdm(range(len(intrnrn_files))):
        file_name = intrnrn_files[i]
        if 'bipolar' in file_name:
            for j in range(num_pots[i]):
                y_true.append(0)
        elif 'amacrine' in file_name:
            for j in range(num_pots[i]):
                y_true.append(1)
        else:
            for j in range(num_pots[i]):
                y_true.append(2)
        for k in stims.keys():
            stim = stims[k][i]
            padded_stim = intracellular.pad_to_edge(scipy.stats.zscore(stim))
            if k not in model_responses:
                model_responses[k] = []
            model_responses[k].append(bc.batch_compute_model_response(stimuli.concat(padded_stim),
                                                                      model,batch_compute_size, 
                                                                      insp_keys=set(insp_layers)))
            # Reshape potentially flat layers
            for j,cl in enumerate(insp_layers):
                if len(model_responses[k][-1][cl].shape) <= 2:
                    try:
                        model_responses[k][-1][cl] = model_responses[k][-1][cl].reshape((-1,chans[0],36,36))
                    except:
                        model_responses[k][-1][cl] = model_responses[k][-1][cl].reshape((-1,chans[1],26,26))
    
    # uses classify to get the most correlated cell/layer/subtype for each interneuron recording. 
    # Stored in all_cell_info. y_pred does a baseline "classification": record the convolutional 
    # layer that the most correlated cell is in.
    # See intracellular.py for more info
    # y_pred holds the layer that was maximally correlated between the insp_layers 
    print("Calculating intracellular correlations...\n")
    all_cell_info = dict()
    y_pred = dict()
    intrnrn_info = []
    cell_types = ["bipolar", "amacrine", "horizontal"]

    for i in tqdm(range(len(intrnrn_files))):
        for k in stims.keys():
            model_response = model_responses[k][i]
            stim = stims[k][i]
            for j in range(mem_pots[k][i].shape[0]):
                potential = mem_pots[k][i][j]
                cell_info = intracellular.classify(potential, model_response, stim.shape[0], 
                                                                    layer_keys=insp_layers)
                if k not in all_cell_info:
                    all_cell_info[k] = []
                    y_pred[k] = []
                all_cell_info[k].append(cell_info)
                # Determines index of layer of  max correlation
                y_pred[k].append(analysis.index_of(cell_info[0], insp_layers)) 

                # Record info
                layer, channel, (row, col), cor_coef = cell_info
                info = dict()
                info['stim_type'] = k
                info['cellfile'] = intrnrn_files[i]
                info['cell_type'] = cell_types[y_true[i]]
                info['save_folder'] = folder
                info['cell_idx'] = j
                info['layer'] = layer
                info['channel'] = channel
                info['row'] = row
                info['col'] = col
                info['correlation'] = cor_coef
                intrnrn_info.append(info)
    
    stats['all_cell_info'] = all_cell_info
    stats['intrnrn_info'] = intrnrn_info

    # Collect intracellular correlations 
    idx_to_cell = {i:t for i,t in enumerate(cell_types)}
    cors = {"bipolar":[], "amacrine":[], "horizontal":[]}
    for i in range(len(y_true)):
        idx = y_true[i]
        cell_type = idx_to_cell[idx]
        # Add correlations from all stimulus types for this particular interneuron
        cors[cell_type].extend([all_cell_info[k][i][-1] for k in all_cell_info.keys()])
    # Collect average correlation coefficient for each cell type
    for key in cors.keys():
        stats[key+"_intr_cor"] = np.mean(cors[key])
        print(key, "intracellular:", stats[key+"_intr_cor"])

    # Collect total average correlation (and other statistics)
    avg_intr_cor = np.mean(np.asarray([[all_cell_info[k][i][-1] for i in range(len(all_cell_info[k]))] for k in all_cell_info.keys()]))
    print("Mean intracellular:", avg_intr_cor)
    stats['Meanintracellular'] = avg_intr_cor

    std = np.std(np.asarray([[all_cell_info[k][i][-1] for i in range(len(all_cell_info[k]))] for k in all_cell_info.keys()]))
    stats['STDintracellular'] = std
    m = np.min(np.asarray([[all_cell_info[k][i][-1] for i in range(len(all_cell_info[k]))] for k in all_cell_info.keys()]))
    stats["Minintracellular"] = m
    m = np.max(np.asarray([[all_cell_info[k][i][-1] for i in range(len(all_cell_info[k]))] for k in all_cell_info.keys()]))
    stats["Maxintracellular"] = m
    
    stim_type = 'boxes'
    # Make example correlation map
    model_response = model_responses[stim_type][-1]
    potential = mem_pots[stim_type][-1][-1]
    layer, k, (i,j), r = all_cell_info[stim_type][-1]
    #fig = plt.figure()
    #plt.imshow(intracellular.correlation_map(potential, model_response[layer][:, k]))
    #plt.savefig(os.path.join(folder, "correlation_map.png"))
    
    keys = cell_types
    layer_dict = {}
    # Tally layers for maximally correlated cell
    # y_true holds the true neuron type for each interneuron (0 is bipolar, 1 is amacrine, 2 is horizontal)
    for i in range(len(y_true)):
        if y_true[i] not in layer_dict:
            layer_dict[y_true[i]] = [0 for i in range(len(insp_layers))]
        for k in y_pred.keys():
            layer_dict[y_true[i]][y_pred[k][i]] += 1 # tallys for each neuron type the layer of max correlation
    
    width = 0.5
    lkeys = list(layer_dict.keys())
    ind = np.arange(0,len(insp_layers))
    #for i,k in enumerate(lkeys):
    #    fig = plt.figure()
    #    plt.bar(ind, [count for count in layer_dict[k]], width)
    #    plt.xticks(ind,insp_layers)
    #    plt.title("Layer of unit with max correlation for "+keys[i])
    #    plt.savefig(os.path.join(folder, keys[i]+"_max_cor.png"))
    
    layer_dict = {keys[k]:v for k,v in layer_dict.items()}
    # layer dict now is {"bipolar":[layer1count, layer2count], "amacrine":...}
    for k in layer_dict.keys():
        stats[k+"_maxcorratio"] = str(layer_dict[k][0])+":"+str(layer_dict[k][1])
    
    print("Completed in", time.time()-starttime, "seconds")
    print("\n\n\n\n")
    
    return stats

def analyze_models(model_folders):
    """
    Does the initial model analysis over all the models in a hyperparameter search. 

    grand_folder: string
        the name of the folder that contains the model folders

    """
    # Get Table Headers
    #with open('models.csv') as f:
    #    model_headers = f.readline().strip().split("!")
    model_headers = [
        "exp_name","exp_num","dataset","l2","ValAcc","n_epochs","model_type",
        "bias","chans","results_file","lr","n_output_units","starting_exp_num",
        "l1","adapt_gauss","FinalTestAcc","OldTestAcc","architecture","Meanintracellular",
        "cells","ValLoss","noise","Minintracellular","stim_type","poor_results",
        "TrainLoss","Maxintracellular","FinalLoss","batch_size","FinalVal",
        "save_folder","cell0","cell1","cell2","cell3","cell4","cell5","cell6",
        "cell7","cell8","cell9","cell10","cell11","cell12","cell13","cell14",
        "cell16","cell17","cell18","cell20","cell21","cell22","cell23","cell24",
        "cell25","cell27","cell31","bnorm_momentum","linear_bias","shift",
        "lossfxn","abs_bias","log_poisson","softplus","bipolar_intr_cor", 
        "amacrine_intr_cor","horizontal_intr_cor",'avg_oms_ratio','std_oms_ratio'
    ]
    intrnrn_headers = [
        "cellfile", "cell_idx", "cell_type", "stim_type", "save_folder", 
        "correlation", "layer", "channel", "row", "col"
    ]

    #Load data
    intrnrn_data = load_interneuron_data()
    cells = "all"
    dataset = "15-10-07"
    stim_type = "naturalscene"
    norm_stats = [51.49175, 53.62663279042969]
    test_data = loadexpt(dataset,cells,stim_type,'test',40,0, norm_stats=norm_stats)
    
    # Analysis loop
    for fcount, folder in enumerate(model_folders):
        model_stats = dict()
        print("Analyzing", folder, " -- ", len(model_folders) - fcount, "folders left...")
        model_stats[folder] = analyze_model(folder, intrnrn_data, test_data=test_data)

        # Record intrnrn data in table
        table_path = os.path.join(grand_folder, "intrnrn_data.csv")
        write_header = not os.path.exists(table_path)
        intrnrn_frame = analysis.make_intrnrn_frame(model_stats, intrnrn_headers)
        intrnrn_frame = intrnrn_frame.reindex(intrnrn_headers, axis=1)
        intrnrn_frame.to_csv(table_path, header=write_header, mode='a', sep="!", index=False)

        # Record model data in table
        table_path = os.path.join(grand_folder, "model_data.csv")
        write_header = not os.path.exists(table_path)
        model_frame = analysis.make_model_frame(model_stats, model_headers)
        model_frame = model_frame.reindex(model_headers, axis=1)
        model_frame.to_csv(table_path, header=write_header, mode='a', sep="!", index=False)

if __name__ == "__main__":
    start_idx = None
    if len(sys.argv) >= 2:
        try:
            start_idx = int(sys.argv[1])
            grand_folders = sys.argv[2:]
        except:
            grand_folders = sys.argv[1:]
    DEVICE = torch.device("cuda:0")
    torch.cuda.empty_cache()
    for grand_folder in grand_folders:
        exp_folder = os.path.join("../training_scripts/",grand_folder)
        _, model_folders, _ = next(os.walk(exp_folder))
        for i,folder in enumerate(model_folders):
            model_folders[i] = os.path.join(grand_folder,folder)

        # Sort model folders and select folders above start_idx if argued
        try:
            model_folders = sorted(model_folders, key=lambda x: int(x.split("_")[1]))
            if start_idx is not None:
                for i in range(len(model_folders)):
                    if i == start_idx:
                        model_folders = model_folders[i:]
                        break
            print("Model Folders:")
            print("\n".join(model_folders))
        except IndexError as e:
            print("index error for", grand_folder)
            print("Using model_folders:", model_folders)
        analyze_models(model_folders)
    
    
    
