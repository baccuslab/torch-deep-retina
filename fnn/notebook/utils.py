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
    
def poisson_error(model, data, single_corr, device, n_repeats=5, k1=None, k2=None, k3=None):
    with torch.no_grad():
        pred_corr = []
        pred_single_trial = []
        for n in range(n_repeats):
            val_pred = []
            val_targ = []
            for x,y in data:
                x = x.to(device)
                if k1 != None:
                    out = torch.poisson(k1*model.bipolar(x))/k1
                else:
                    out = model.bipolar(x)
                if k2 != None:
                    out = torch.poisson(k2*model.amacrine(out))/k2
                else:
                    out = model.amacrine(out)
                if k3 != None:
                    out = torch.poisson(k3*model.ganglion(out))/k3
                else:
                    out = model.ganglion(out)
                val_pred.append(out.detach().cpu().numpy())
                val_targ.append(y.detach().numpy())
            val_pred = np.concatenate(val_pred, axis=0)
            val_targ = np.concatenate(val_targ, axis=0)
            pred_corr.append(corr_matrix(5, val_pred))
            pred_single_trial.append(val_pred)
        pred_corr = np.stack(pred_corr).mean(axis=0)
        pred_single_trial = np.stack(pred_single_trial)
    error = np.abs(single_corr-pred_corr).sum()/single_corr.sum()
    pearsons = []
    pred_mean = pred_single_trial.mean(axis=0)
    for cell in range(5):
        pearsons.append(pearsonr(pred_mean[:,cell],val_targ[:,cell])[0])
    accuracy = np.array(pearsons).mean()
    fano = Fano(pred_single_trial)
    stim_corr = stimuli_corr_matrix(5, n_repeats, pred_single_trial)
    noise_corr = pred_corr - stim_corr
    diagonal_idxs = [0, 6, 12, 18, 24]
    mean_stim_corr = np.delete(stim_corr.flatten(), diagonal_idxs).mean()
    mean_noise_corr = np.delete(noise_corr.flatten(), diagonal_idxs).mean()
    return error, accuracy, fano, mean_stim_corr, mean_noise_corr

def gaussian_error(model, data, single_corr, device, n_repeats=5, std1=0, std2=0):
    with torch.no_grad():
        pred_corr = []
        pred_single_trial = []
        for n in range(n_repeats):
            val_pred = []
            val_targ = []
            for x,y in data:
                x = x.to(device)
                model.bipolar[3].sigma.data = torch.tensor(std1).to(device)
                model.bipolar[3].training = True
                model.amacrine[4].sigma.data = torch.tensor(std2).to(device)
                model.amacrine[4].training = True
                out = model(x)
                val_pred.append(out.detach().cpu().numpy())
                val_targ.append(y.detach().numpy())
            val_pred = np.concatenate(val_pred, axis=0)
            val_targ = np.concatenate(val_targ, axis=0)
            pred_corr.append(corr_matrix(5, val_pred))
            pred_single_trial.append(val_pred)
        pred_corr = np.stack(pred_corr).mean(axis=0)
        pred_single_trial = np.stack(pred_single_trial)
    error = np.abs(single_corr-pred_corr).sum()/single_corr.sum()
    pearsons = []
    pred_mean = pred_single_trial.mean(axis=0)
    for cell in range(5):
        pearsons.append(pearsonr(pred_mean[:,cell],val_targ[:,cell])[0])
    accuracy = np.array(pearsons).mean()
    fano = Fano(pred_single_trial)
    stim_corr = stimuli_corr_matrix(5, n_repeats, pred_single_trial)
    noise_corr = pred_corr - stim_corr
    diagonal_idxs = [0, 6, 12, 18, 24]
    mean_stim_corr = np.delete(stim_corr.flatten(), diagonal_idxs).mean()
    mean_noise_corr = np.delete(noise_corr.flatten(), diagonal_idxs).mean()
    return error, accuracy, fano, mean_stim_corr, mean_noise_corr
    