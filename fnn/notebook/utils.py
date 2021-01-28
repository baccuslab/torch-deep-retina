import os
import json
import numpy as np
import matplotlib.pyplot as plt
import torch
from scipy.stats import pearsonr
import torchdeepretina.stimuli as stim

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

def Fano2(single_trial):
    fano = np.var(0.01*single_trial.sum(axis=1), axis=0)/np.mean(0.01*single_trial.sum(axis=1), axis=0)
    fano = fano.mean()
    return fano

def Pearsonr_std(single_trial, pred_single_trial, n_cells):
    stds = single_trial.std(0)
    pred_stds = pred_single_trial.std(0)
    pearsons = []
    for cell in range(n_cells):
        pearsons.append(pearsonr(pred_stds[:,cell],stds[:,cell])[0])
    mean_pearson = np.array(pearsons).mean()
    return mean_pearson

def STD_error(single_trial, pred_single_trial):
    pred_single_trial = pred_single_trial * single_trial.mean() / pred_single_trial.mean()
    stds = single_trial.std(0)
    pred_stds = pred_single_trial.std(0)
    error = np.abs(pred_stds - stds).sum()/stds.sum()
    return error
    
def Noises(model, data, single_corr, single_trial, device, n_repeats=5, 
           n_cells=5, poisson=[None, None, None], gaussian=[0, 0, 0, 0], thre=0):
    model = model.to(device)
    with torch.no_grad():
        pred_corr = []
        pred_single_trial = []
        for n in range(n_repeats):
            val_pred = []
            val_targ = []
            for x,y in data:
                x = x.to(device)
                out = noise_model(x, model, device, poisson, gaussian, thre)
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
    std_error = STD_error(single_trial, pred_single_trial)
    fano = Fano(pred_single_trial)
    stim_corr = stimuli_corr_matrix(n_cells, n_repeats, pred_single_trial)
    noise_corr = pred_corr - stim_corr
    diagonal_idxs = list(range(0, n_cells*n_cells, n_cells+1))
    mean_stim_corr = np.delete(stim_corr.flatten(), diagonal_idxs).mean()
    mean_noise_corr = np.delete(noise_corr.flatten(), diagonal_idxs).mean()
    return error, accuracy, fano, mean_stim_corr, mean_noise_corr, std_error

def noise_model(x, model, device, poisson=[None, None, None], gaussian=[0, 0, 0, 0], thre=0):
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
        out = model.ganglion[2:](out)
        out[out < thre] = 0
        if poisson[2] != None:
            out = torch.poisson(poisson[2]*out)/poisson[2]
    return out 
    
def single_noise_plot(para_name, noise_list, error_list, accuracy_list, fano_list, mean_stim_corr_list, mean_noise_corr_list,
                      std_error_list, recorded_fano=0.091, recorded_mean_stim_corr=0.1638, recorded_mean_noise_corr=0.0127):
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
    plt.plot(noise_list, std_error_list, 'bo-')
    plt.xlabel(para_name)
    plt.ylabel('std error')
    plt.show()
    
def model_single_trial(model, data, device, n_repeats=5, n_cells=5, poisson=[None, None, None], 
                       gaussian=[0, 0, 0, 0], thre=0):
    
    model = model.to(device)
    with torch.no_grad():
        pred_single_trial = []
        for n in range(n_repeats):
            val_pred = []
            val_targ = []
            for x,y in data:
                x = x.to(device)
                out = noise_model(x, model, device, poisson, gaussian, thre)
                val_pred.append(out.detach().cpu().numpy())
                val_targ.append(y.detach().numpy())
            val_pred = np.concatenate(val_pred, axis=0)
            val_targ = np.concatenate(val_targ, axis=0)
            pred_single_trial.append(val_pred)
        pred_single_trial = np.stack(pred_single_trial)
    return pred_single_trial

def variance_plot(single_trial, pred_single_trial):
    
    noise = single_trial - single_trial.mean(0)
    pred_noise = pred_single_trial - pred_single_trial.mean(0)
    
    zero_idx = np.where(single_trial.flatten() == 0)[0]
    plt.hist(np.delete(noise.var(0).flatten(), zero_idx), bins=50, range=(0,500), color='r', label='data')
    plt.xlabel('Variance over trials')
    plt.ylabel('Histogram')
    plt.legend()
    plt.show()
    zero_idx = np.where(pred_single_trial.flatten() < 0.69)[0]
    plt.hist(np.delete(pred_noise.var(0).flatten(), zero_idx), bins=50, range=(0,500), color='b', label='model')
    plt.xlabel('Variance over trials')
    plt.ylabel('Histogram')
    plt.legend()
    plt.show()
    
    plt.plot(single_trial.mean(0).flatten(), noise.var(0).flatten(), 'ro', label='data')
    plt.ylim(0, 1000)
    plt.xlabel('mean firing rate')
    plt.ylabel('variance over trials')
    plt.legend()
    plt.show()
    plt.plot(pred_single_trial.mean(0).flatten(), pred_noise.var(0).flatten(), 'bo', label='model')
    plt.ylim(0, 1000)
    plt.xlabel('mean firing rate')
    plt.ylabel('variance over trials')
    plt.legend()
    plt.show()
    
def correlation_plot(single_trial, pred_single_trial, num_trials=5, num_cells=5):
    
    diagonal_idxs = list(range(0, num_cells*num_cells, num_cells+1))
    
    corr = single_trial_corr_matrix(num_cells, 5, single_trial)
    pred_corr = single_trial_corr_matrix(num_cells, num_trials, pred_single_trial)
    
    stim_corr = stimuli_corr_matrix(num_cells, 5, single_trial)
    noise_corr = corr - stim_corr
    pred_stim_corr = stimuli_corr_matrix(num_cells, num_trials, pred_single_trial)
    pred_noise_corr = pred_corr - pred_stim_corr
    
    ave_corr = corr_matrix(num_cells, single_trial.mean(0))
    pred_ave_corr = corr_matrix(num_cells, pred_single_trial.mean(0))
    
    corr = np.delete(corr.flatten(), diagonal_idxs)
    pred_corr = np.delete(pred_corr.flatten(), diagonal_idxs)
    stim_corr = np.delete(stim_corr.flatten(), diagonal_idxs)
    pred_stim_corr = np.delete(pred_stim_corr.flatten(), diagonal_idxs)
    noise_corr = np.delete(noise_corr.flatten(), diagonal_idxs)
    pred_noise_corr = np.delete(pred_noise_corr.flatten(), diagonal_idxs)
    ave_corr = np.delete(ave_corr.flatten(), diagonal_idxs)
    pred_ave_corr = np.delete(pred_ave_corr.flatten(), diagonal_idxs)
    
    plt.plot(corr, pred_corr, 'bo')
    plt.plot(corr, corr, 'r-')
    plt.xlabel('data')
    plt.ylabel('model')
    plt.title('pairwise correlation')
    plt.show()
    plt.plot(stim_corr, pred_stim_corr, 'bo')
    plt.plot(stim_corr, stim_corr, 'r-')
    plt.xlabel('data')
    plt.ylabel('model')
    plt.title('stimulus correlation')
    plt.show()
    plt.plot(noise_corr, pred_noise_corr, 'bo')
    plt.plot(noise_corr, noise_corr, 'r-')
    plt.xlabel('data')
    plt.ylabel('model')
    plt.title('noise correlation')
    plt.show()
    plt.plot(ave_corr, pred_ave_corr, 'bo')
    plt.plot(ave_corr, ave_corr, 'r-')
    plt.xlabel('data')
    plt.ylabel('model')
    plt.title('trial-averaged correlation')
    plt.show()
    
def fano_contrast(model, device, contrast, n_repeats=5, 
                  poisson=[None, None, None], gaussian=[0, 0, 0, 0], stim_type='fullfield', length=3000):
    
    if stim_type == 'fullfield':
        stimuli = np.random.randn(length, 1, 1) * np.ones((length, 50, 50)) * contrast + 1
    elif stim_type == 'checkboard':
        stimuli = np.random.randn(length, 50, 50) * contrast + 1
    stimuli = (stimuli - stimuli.mean())/stimuli.std()
    stimuli = torch.from_numpy(stim.rolling_window(stimuli, 40, time_axis=0)).float().to(device)
    
    model = model.to(device)
    with torch.no_grad():
        pred_single_trial = []
        for _ in range(n_repeats):
            val_pred = []
            for i in range(stimuli.shape[0]):
                x = stimuli[i:i+1]
                out = noise_model(x, model, device, poisson, gaussian)
                val_pred.append(out.detach().cpu().numpy())
            val_pred = np.concatenate(val_pred, axis=0)
            pred_single_trial.append(val_pred)
    pred_single_trial = np.stack(pred_single_trial)
    fano = Fano(pred_single_trial)
    
    return fano
    
    
    