import os
import json
import numpy as np
import matplotlib.pyplot as plt
import torch
from scipy.stats import pearsonr

def plot_evals(names):
    saved_path = '/home/xhding/saved_model'
    fig, ax = plt.subplots()
    ax.set(xlabel='epoch', ylabel='correlation')
    for name in names:
        epochs = []
        pearsons = []
        eval_path = os.path.join(saved_path, name, 'eval.json')
        with open(eval_path, 'r') as f:
            history = json.load(f)
            for item in history:
                epochs.append(item['epoch'])
                pearsons.append(item['pearson'])
            ax.plot(epochs, pearsons, 'o-', label=name)
            ax.legend()
            
def plot_temperal_filters(path, layer):
    for checkpoint_path in os.scandir(path):
        if checkpoint_path.name.endswith("pth"):
            checkpoint = torch.load(os.path.join(path, checkpoint_path.name))
            filter_w = checkpoint['model_state_dict'][layer].cpu().numpy().squeeze()
            plt.plot(np.arange(filter_w.shape[0]), filter_w)
    plt.show()
    
def corr_matrix(num_cells, response):
    corr_matrix = np.zeros((num_cells, num_cells))
    for i in range(num_cells):
        for j in range(num_cells):
            corr_matrix[i, j] = pearsonr(response[:,i], response[:,j])[0]
    return corr_matrix

def single_trial_corr_matrix(num_cells, num_trials, single_trial):
    result = []
    for trial in range(num_trials):
        result.append(corr_matrix(num_cells, single_trial[trial]))
    result = np.array(result).mean(axis=0)
    return result

def stimuli_corr_matrix(num_cells, num_trials, single_trial):
    corr_matrix = np.zeros((num_cells, num_cells))
    for i in range(num_cells):
        for j in range(num_cells):
            r = []
            for trial1 in range(num_trials):
                for trial2 in range(num_trials):
                    if trial1 != trial2:    
                        r.append(pearsonr(single_trial[trial1,:,i], single_trial[trial2,:,j])[0])
            corr_matrix[i, j] = np.array(r).mean()
    return corr_matrix

def correlation_matrix(pred, targ, single_trial, binary, thre=1, num_cells=5, num_trials=5, poisson=False):
    if poisson:
        pred = np.random.poisson(pred)
    if binary:
        targ_binary = np.zeros_like(targ)
        targ_binary[targ > thre] = 1
        pred_binary = np.zeros_like(pred)
        pred_binary[pred > thre] = 1
        single_trial_binary = np.zeros_like(single_trial)
        single_trial_binary[single_trial > thre] = 1
        return corr_matrix(num_cells, pred_binary), corr_matrix(num_cells, targ_binary), single_trial_corr_matrix(num_cells, num_trials, single_trial_binary)
    else:
        return corr_matrix(num_cells, pred), corr_matrix(num_cells, targ), single_trial_corr_matrix(num_cells, num_trials, single_trial)
    
    