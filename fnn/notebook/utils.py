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
    
def Fano(single_trial):
    fano = np.nanmean(np.var(0.01*single_trial, axis=0)/np.mean(0.01*single_trial, axis=0))
    return fano
    
def Noises(model, data, single_corr, device, n_repeats=5, n_cells=5, poisson=[None, None, None], gaussian=[0, 0, 0, 0]):
    model = model.to(device)
    with torch.no_grad():
        pred_corr = []
        pred_single_trial = []
        for n in range(n_repeats):
            val_pred = []
            val_targ = []
            for x,y in data:
                x = x.to(device)
                out = noise_model(x, model, device, poisson, gaussian)
                val_pred.append(out.detach().cpu().numpy())
                val_targ.append(y.detach().numpy())
            val_pred = np.concatenate(val_pred, axis=0)
            val_targ = np.concatenate(val_targ, axis=0)
            pred_corr.append(corr_matrix(n_cells, val_pred))
            pred_single_trial.append(val_pred)
        pred_corr = np.stack(pred_corr).mean(axis=0)
        pred_single_trial = np.stack(pred_single_trial)
    error = np.abs(single_corr-pred_corr).sum()/single_corr.sum()
    pearsons = []
    pred_mean = pred_single_trial.mean(axis=0)
    for cell in range(n_cells):
        pearsons.append(pearsonr(pred_mean[:,cell],val_targ[:,cell])[0])
    accuracy = np.array(pearsons).mean()
    fano = Fano(pred_single_trial)
    stim_corr = stimuli_corr_matrix(n_cells, n_repeats, pred_single_trial)
    noise_corr = pred_corr - stim_corr
    diagonal_idxs = list(range(0, n_cells*n_cells, n_cells+1))
    mean_stim_corr = np.delete(stim_corr.flatten(), diagonal_idxs).mean()
    mean_noise_corr = np.delete(noise_corr.flatten(), diagonal_idxs).mean()
    return error, accuracy, fano, mean_stim_corr, mean_noise_corr

def noise_model(x, model, device, poisson=[None, None, None], gaussian=[0, 0, 0, 0]):
    with torch.no_grad():
        noise = gaussian[0] * torch.randn(x.size()).to(device)
        out = x + noise
        out = model.bipolar[:3](out)
        noise = gaussian[1] * torch.randn(out.size()).to(device)
        out = out + noise
        if poisson[0] != None:
            out = torch.poisson(poisson[0]*model.bipolar[3:](out))/poisson[0]
        else:
            out = model.bipolar[3:](out)
        out = model.amacrine[:4](out)
        noise = gaussian[2] * torch.randn(out.size()).to(device)
        out = out + noise
        if poisson[1] != None:
            out = torch.poisson(poisson[1]*model.amacrine[4:](out))/poisson[1]
        else:
            out = model.amacrine[4:](out)
        out = model.ganglion[:2](out)
        noise = gaussian[3] * torch.randn(out.size()).to(device)
        out = out + noise
        if poisson[2] != None:
            out = torch.poisson(poisson[2]*model.ganglion[2:](out))/poisson[2]
        else:
            out = model.ganglion[2:](out)
    return out 
    
def single_noise_plot(para_name, noise_list, error_list, accuracy_list, fano_list, mean_stim_corr_list, mean_noise_corr_list,
               recorded_fano=0.091, recorded_mean_stim_corr=0.1638, recorded_mean_noise_corr=0.0127):
    plt.plot(noise_list, error_list, 'bo-')
    plt.xlabel(para_name)
    plt.ylabel('error')
    plt.show()
    plt.plot(noise_list, accuracy_list, 'bo-')
    plt.xlabel(para_name)
    plt.ylabel('pearson correlation')
    plt.show()
    plt.plot(noise_list, fano_list, 'bo-', label='model')
    plt.xlabel(para_name)
    plt.ylabel('Fano factor')
    plt.axhline(y=recorded_fano, color='r', label='data')
    plt.legend()
    plt.show()
    plt.plot(noise_list, mean_stim_corr_list, 'bo-', label='model: stimuli correlation')
    plt.plot(noise_list, mean_noise_corr_list, 'ro-', label='model: noise correlation')
    plt.axhline(y=recorded_mean_stim_corr, color='b', label='data: stimuli correlation')
    plt.axhline(y=recorded_mean_noise_corr, color='r', label='data: noise correlation')
    plt.xlabel(para_name)
    plt.ylabel('correlation')
    plt.legend()
    plt.show()