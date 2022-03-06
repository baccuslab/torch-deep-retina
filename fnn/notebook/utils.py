import os
import json
import random
import numpy as np
import matplotlib.pyplot as plt
import torch
from scipy.stats import pearsonr
from scipy.optimize import curve_fit
import torchdeepretina.stimuli as stim
from fnn.distributions import *

def corr_matrix(response):
    num_cells = response.shape[-1]
    corr_matrix = np.zeros((num_cells, num_cells))
    for i in range(num_cells):
        for j in range(num_cells):
            corr_matrix[i, j] = pearsonr(response[:,i], response[:,j])[0]
    return corr_matrix

def single_trial_corr_matrix(single_trial):
    num_trials = single_trial.shape[0]
    result = []
    for trial in range(num_trials):
        result.append(corr_matrix(single_trial[trial]))
    result = np.array(result).mean(axis=0)
    return result

def stim_corr(single_trial):
    num_cells = single_trial.shape[-1]
    num_trials = single_trial.shape[0]
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

def noise_corr2(single_trial):
    noise = single_trial - single_trial.mean(0)
    cov_noise = (np.expand_dims(noise, -1) * np.expand_dims(noise, -2)).mean((0,1))
    V = ((single_trial - single_trial.mean((0,1)))**2).mean((0,1))
    noise_corr = cov_noise / np.sqrt(np.expand_dims(V, -1) * np.expand_dims(V, -2))
    return noise_corr

def stim_corr2(single_trial):
    mean_resp = single_trial.mean(0) - single_trial.mean((0,1))
    cov_stim = (np.expand_dims(mean_resp, -1) * np.expand_dims(mean_resp, -2)).mean(0)
    V = ((single_trial - single_trial.mean((0,1)))**2).mean((0,1))
    stim_corr = cov_stim / np.sqrt(np.expand_dims(V, -1) * np.expand_dims(V, -2))
    return stim_corr

def noise_model_pre(x, model, device, gaussian=[0, 0, 0, 0]):
    with torch.no_grad():
        noise = gaussian[0] * torch.randn(x.size()).to(device)
        out = x + noise
        out = model.bipolar[:3](out)
        noise = gaussian[1] * torch.randn(out.size()).to(device)
        out = out + noise
        out = model.bipolar[3:](out)
        out = model.amacrine[:4](out)
        noise = gaussian[2] * torch.randn(out.size()).to(device)
        out = out + noise
        out = model.amacrine[4:](out)
        out = model.ganglion[:2](out)
        noise = gaussian[3] * torch.randn(out.size()).to(device)
        out = out + noise
        out = model.ganglion[2:](out)
    return out

def stim_noise(std, shape, history=40):
    noise = std * np.random.randn(shape[0]+history, shape[-2], shape[-1])
    noise = stim.rolling_window(noise, history, 0)
    return noise

def correlated_noise(std, shape, p, phi, sigma):
    noise = std * np.random.randn(*shape)
    out = np.zeros((shape[0]+p, *shape[1:]))
    for i in range(shape[0]):
        out[p+i] = phi.dot(out[i:p+i]) + sigma * noise[i]
    out = out[p:]
    return out

def noise_model_pre2(x, model, device, gaussian=[0, 0, 0, 0]):
    with torch.no_grad():
        noise = stim_noise(gaussian[0], x.shape)
        noise = torch.from_numpy(noise).float().to(device)
        out = x + noise
        out = model.bipolar[:3](out)
        noise = gaussian[1] * torch.randn(out.size()).to(device)
        out = out + noise
        out = model.bipolar[3:](out)
        out = model.amacrine[:4](out)
        noise = gaussian[2] * torch.randn(out.size()).to(device)
        out = out + noise
        out = model.amacrine[4:](out)
        out = model.ganglion[:2](out)
        noise = gaussian[3] * torch.randn(out.size()).to(device)
        out = out + noise
        out = model.ganglion[2:](out)
    return out

def noise_model_pre3(x, model, device, p, phi, sigma, gaussian=[0, 0, 0, 0]):
    with torch.no_grad():
        noise = stim_noise(gaussian[0], x.shape)
        noise = torch.from_numpy(noise).float().to(device)
        out = x + noise
        out = model.bipolar[:3](out)
        noise = gaussian[1] * torch.randn(out.size()).to(device)
        out = out + noise
        out = model.bipolar[3:](out)
        out = model.amacrine[:4](out)
        noise = correlated_noise(gaussian[2], out.shape, p, phi, sigma)
        noise = torch.from_numpy(noise).float().to(device)
        out = out + noise
        out = model.amacrine[4:](out)
        out = model.ganglion[:2](out)
        noise = gaussian[3] * torch.randn(out.size()).to(device)
        out = out + noise
        out = model.ganglion[2:](out)
    return out

def model_single_trial_pre(model, data, device, n_repeats=15, gaussian=[0, 0, 0, 0], seed=None):
    
    if seed != None:
        torch.manual_seed(seed)
    model = model.to(device)
    with torch.no_grad():
        pred_single_trial = []
        for n in range(n_repeats):
            val_pred = []
            for x, _ in data:
                x = x.to(device)
                out = noise_model_pre(x, model, device, gaussian)
                val_pred.append(out)
            val_pred = torch.cat(val_pred, dim=0)
            pred_single_trial.append(val_pred)
        pred_single_trial = torch.stack(pred_single_trial)
    pred_single_trial = pred_single_trial.detach().cpu().numpy()
    return pred_single_trial

def model_single_trial_pre2(model, data, device, n_repeats=15, gaussian=[0, 0, 0, 0], seed=None):
    
    if seed != None:
        torch.manual_seed(seed)
    model = model.to(device)
    with torch.no_grad():
        val_pred = []
        for x, _ in data:
            x = x.to(device).repeat(n_repeats,1,1,1)
            out = noise_model_pre(x, model, device, gaussian)
            out = out.reshape(n_repeats,-1,out.shape[-1])
            val_pred.append(out)
        val_pred = torch.cat(val_pred, dim=1)
    pred_single_trial = val_pred.detach().cpu().numpy()
    return pred_single_trial

def model_single_trial_pre_try(model, data, device, p, phi, sigma, n_repeats=15, gaussian=[0, 0, 0, 0], seed=None):
    
    if seed != None:
        torch.manual_seed(seed)
    model = model.to(device)
    with torch.no_grad():
        pred_single_trial = []
        for n in range(n_repeats):
            val_pred = []
            for x, _ in data:
                x = x.to(device)
                out = noise_model_pre3(x, model, device, p, phi, sigma, gaussian)
                val_pred.append(out)
            val_pred = torch.cat(val_pred, dim=0)
            pred_single_trial.append(val_pred)
        pred_single_trial = torch.stack(pred_single_trial)
    pred_single_trial = pred_single_trial.detach().cpu().numpy()
    return pred_single_trial

def poly_para_fit(recording, pred_single_trial_pre, t_list):
    def poly(x,a,b,c,d):
        return a*x**4+b*x**3+c*x**2+d*x
    poly_paras = []
    num_cells = pred_single_trial_pre.shape[-1]
    for cell in range(num_cells):
        means = []
        rates = []
        for rate in range((t_list[cell]-1)*100):
            mean, var, _, _ = recording.stats_rate(pred_single_trial_pre.mean(0), cell=cell, rate=rate)
            means.append(mean)
            rates.append(rate/100)
        para = curve_fit(poly, np.array(rates)[~np.isnan(means)], np.array(means)[~np.isnan(means)])[0]
        poly_paras.append(np.append(para, 0.))
    return poly_paras

def poly_para_fit2(recording, pred_single_trial_pre, t_list):
    poly_paras = []
    num_cells = pred_single_trial_pre.shape[-1]
    for cell in range(num_cells):
        means = []
        rates = []
        for rate in range((t_list[cell]-1)*100):
            mean, var, _, _ = recording.stats_rate(pred_single_trial_pre.mean(0), cell=cell, rate=rate)
            means.append(mean)
            rates.append(rate/100)
        poly_para = np.polyfit(np.array(rates)[~np.isnan(means)], np.array(means)[~np.isnan(means)], 4, full=True)[0]
        poly_paras.append(poly_para)
    return poly_paras

def model_single_trial_post(pred_single_trial_pre, binomial_para, t_list, poly_paras, pred, thre=3, seed=None):
    
    if seed != None:
        np.random.seed(seed)
        
    pred_single_trial = np.zeros(pred_single_trial_pre.shape)
    num_cells = pred_single_trial_pre.shape[-1]
    for cell in range(num_cells):
        dist = distribution(t_list[cell])
        pred_scale = np.polyval(poly_paras[cell], pred_single_trial_pre[:, :, cell]/100)*100
        for rate in range((t_list[cell]-1)*100):
            indices = np.where((pred_scale>=rate-0.5)*(pred_scale<rate+0.5))
            num = indices[0].shape[0]
            if num == 0:
                continue
            r = dist.rate2para('binomial_scale', binomial_para[cell], rate)
            p = [dist.binomial_scale(i, r, binomial_para[cell]) for i in range(t_list[cell])]
            spikes = np.random.choice(t_list[cell], num, p=p)
            pred_single_trial[:, :, cell][indices] = spikes
        indices = np.where(pred_scale>=(t_list[cell]-1)*100-0.5)
        num = indices[0].shape[0]
        if num == 0:
            continue
        r = dist.rate2para('binomial_scale', binomial_para[cell], (t_list[cell]-1)*100)
        p = [dist.binomial_scale(i, r, binomial_para[cell]) for i in range(t_list[cell])]
        spikes = np.random.choice(t_list[cell], num, p=p)
        pred_single_trial[:, :, cell][indices] = spikes

    pred_single_trial[:, pred<thre] = 0
    pred_single_trial = pred_single_trial.astype(np.int8)
    return pred_single_trial

def model_single_trial(model, data, device, t_list, binomial_para, pred, recording,
                       n_repeats=15, gaussian=[0, 0, 0, 0], thre=3, seed1=None, seed2=None):
    pred_single_trial_pre = model_single_trial_pre(model, data, device, n_repeats, gaussian, seed1)
    poly_paras = poly_para_fit(recording, pred_single_trial_pre, t_list)
    pred_single_trial = model_single_trial_post(pred_single_trial_pre, binomial_para, t_list, poly_paras, pred, thre, seed2)
    return pred_single_trial

def model_single_trial_post_multi(pred_single_trial_pre, binomial_para, t_list, poly_paras, pred, n_repeats=5, thre=3, seed=None):
    
    if seed != None:
        np.random.seed(seed)
        
    pred_single_trial = np.zeros((n_repeats, *pred_single_trial_pre.shape))
    num_cells = pred_single_trial_pre.shape[-1]
    for cell in range(num_cells):
        dist = distribution(t_list[cell])
        pred_scale = np.polyval(poly_paras[cell], pred_single_trial_pre[:, :, cell]/100)*100
        for rate in range((t_list[cell]-1)*100):
            indices = np.where((pred_scale>=rate-0.5)*(pred_scale<rate+0.5))
            num = indices[0].shape[0]
            if num == 0:
                continue
            r = dist.rate2para('binomial_scale', binomial_para[cell], rate)
            p = [dist.binomial_scale(i, r, binomial_para[cell]) for i in range(t_list[cell])]
            spikes = np.random.choice(t_list[cell], num*n_repeats, p=p).reshape((n_repeats, num))
            for n in range(n_repeats):
                pred_single_trial[n, :, :, cell][indices] = spikes[n]
        indices = np.where(pred_scale>=(t_list[cell]-1)*100-0.5)
        num = indices[0].shape[0]
        if num != 0:
            r = dist.rate2para('binomial_scale', binomial_para[cell], (t_list[cell]-1)*100)
            p = [dist.binomial_scale(i, r, binomial_para[cell]) for i in range(t_list[cell])]
            spikes = np.random.choice(t_list[cell], num*n_repeats, p=p).reshape((n_repeats, num))
            for n in range(n_repeats):
                pred_single_trial[n, :, :, cell][indices] = spikes[n]
        #pred_single_trial[:, :, pred_scale.mean(0)<thre, cell] = 0

    pred_single_trial[:, :, pred<thre] = 0
    #pred_single_trial[:, :, pred_single_trial.mean((0,1))<thre/100] = 0
    pred_single_trial = pred_single_trial.astype(np.int8)
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
    
def correlation_plot(single_trial, pred_single_trial, ignore_idxs=[]):
    
    num_cells = single_trial.shape[-1]
    diagonal_idxs = list(range(0, num_cells*num_cells, num_cells+1))
    noise_idxs = diagonal_idxs + ignore_idxs
    
    recorded_corr = single_trial_corr_matrix(single_trial)
    pred_corr = single_trial_corr_matrix(pred_single_trial)
    
    recorded_stim_corr = stim_corr(single_trial)
    recorded_noise_corr = recorded_corr - recorded_stim_corr
    pred_stim_corr = stim_corr(pred_single_trial)
    pred_noise_corr = pred_corr - pred_stim_corr
    
    ave_corr = corr_matrix(single_trial.mean(0))
    pred_ave_corr = corr_matrix(pred_single_trial.mean(0))
    
    recorded_corr = np.delete(recorded_corr.flatten(), diagonal_idxs)
    pred_corr = np.delete(pred_corr.flatten(), diagonal_idxs)
    recorded_trial_corr = recorded_stim_corr.flatten()[diagonal_idxs]
    pred_trial_corr = pred_stim_corr.flatten()[diagonal_idxs]
    recorded_stim_corr = np.delete(recorded_stim_corr.flatten(), diagonal_idxs)
    pred_stim_corr = np.delete(pred_stim_corr.flatten(), diagonal_idxs)
    recorded_noise_corr = np.delete(recorded_noise_corr.flatten(), noise_idxs)
    pred_noise_corr = np.delete(pred_noise_corr.flatten(), noise_idxs)
    ave_corr = np.delete(ave_corr.flatten(), diagonal_idxs)
    pred_ave_corr = np.delete(pred_ave_corr.flatten(), diagonal_idxs)
    recorded_fano = np.nanmean(np.var(single_trial, axis=0)/np.mean(single_trial, axis=0), axis=0)
    pred_fano = np.nanmean(np.var(pred_single_trial, axis=0)/np.mean(pred_single_trial, axis=0), axis=0)
    
    plt.plot(recorded_corr, pred_corr, 'bo')
    plt.plot(recorded_corr, recorded_corr, 'r-')
    plt.xlabel('data')
    plt.ylabel('model')
    plt.title('pairwise correlation')
    plt.show()
    plt.plot(recorded_stim_corr, pred_stim_corr, 'bo')
    plt.plot(recorded_stim_corr, recorded_stim_corr, 'r-')
    plt.xlabel('data')
    plt.ylabel('model')
    plt.title('stimulus correlation')
    plt.show()
    plt.plot(recorded_noise_corr, pred_noise_corr, 'bo')
    plt.plot(recorded_noise_corr, recorded_noise_corr, 'r-')
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
    x = np.arange(num_cells)
    width = 0.35
    fig, ax = plt.subplots()
    rects1 = ax.bar(x - width/2, recorded_trial_corr, width, label='data')
    rects2 = ax.bar(x + width/2, pred_trial_corr, width, label='model')
    ax.set_ylabel('correlation')
    ax.set_xlabel('cells')
    ax.set_title('trial-to-trial correlation')
    ax.set_xticks(x)
    ax.legend()
    plt.show()
    x = np.arange(num_cells)
    width = 0.35
    fig, ax = plt.subplots()
    rects1 = ax.bar(x - width/2, recorded_fano, width, label='data')
    rects2 = ax.bar(x + width/2, pred_fano, width, label='model')
    ax.set_ylabel('Fano Factor')
    ax.set_xlabel('cells')
    ax.set_xticks(x)
    ax.legend()
    plt.show()
    
def correlation_plot_2(single_trial, pred_single_trial):
    
    num_cells = single_trial.shape[-1]
    diagonal_idxs = list(range(0, num_cells*num_cells, num_cells+1))
    
    recorded_corr = single_trial_corr_matrix(single_trial)
    pred_corr = single_trial_corr_matrix(pred_single_trial)
    
    recorded_stim_corr = stim_corr2(single_trial)
    recorded_trial_corr = recorded_stim_corr.flatten()[diagonal_idxs]
    recorded_noise_corr = noise_corr2(single_trial)
    pred_stim_corr = stim_corr2(pred_single_trial)
    pred_trial_corr = pred_stim_corr.flatten()[diagonal_idxs]
    pred_noise_corr = noise_corr2(pred_single_trial)
    
    recorded_ave_corr = corr_matrix(single_trial.mean(0))
    pred_ave_corr = corr_matrix(pred_single_trial.mean(0))
    
    recorded_corr = np.delete(recorded_corr.flatten(), diagonal_idxs)
    pred_corr = np.delete(pred_corr.flatten(), diagonal_idxs)
    recorded_stim_corr = np.delete(recorded_stim_corr.flatten(), diagonal_idxs)
    pred_stim_corr = np.delete(pred_stim_corr.flatten(), diagonal_idxs)
    recorded_noise_corr = np.delete(recorded_noise_corr.flatten(), diagonal_idxs)
    pred_noise_corr = np.delete(pred_noise_corr.flatten(), diagonal_idxs)
    recorded_ave_corr = np.delete(recorded_ave_corr.flatten(), diagonal_idxs)
    pred_ave_corr = np.delete(pred_ave_corr.flatten(), diagonal_idxs)
    
    recorded_fano = np.nanmean(np.var(single_trial, axis=0)/np.mean(single_trial, axis=0), axis=0)
    pred_fano = np.nanmean(np.var(pred_single_trial, axis=0)/np.mean(pred_single_trial, axis=0), axis=0)
    
    plt.plot(recorded_corr, pred_corr, 'bo')
    plt.plot(recorded_corr, recorded_corr, 'r-')
    plt.xlabel('data')
    plt.ylabel('model')
    plt.title('pairwise correlation')
    plt.show()
    plt.plot(recorded_stim_corr, pred_stim_corr, 'bo')
    plt.plot(recorded_stim_corr, recorded_stim_corr, 'r-')
    plt.xlabel('data')
    plt.ylabel('model')
    plt.title('stimulus correlation')
    plt.show()
    plt.plot(recorded_noise_corr, pred_noise_corr, 'bo')
    plt.plot(recorded_noise_corr, recorded_noise_corr, 'r-')
    plt.xlabel('data')
    plt.ylabel('model')
    plt.title('noise correlation')
    plt.show()
    plt.plot(recorded_ave_corr, pred_ave_corr, 'bo')
    plt.plot(recorded_ave_corr, recorded_ave_corr, 'r-')
    plt.xlabel('data')
    plt.ylabel('model')
    plt.title('trial-averaged correlation')
    plt.show()
    x = np.arange(num_cells)
    width = 0.35
    fig, ax = plt.subplots()
    rects1 = ax.bar(x - width/2, recorded_trial_corr, width, label='data')
    rects2 = ax.bar(x + width/2, pred_trial_corr, width, label='model')
    ax.set_ylabel('correlation')
    ax.set_xlabel('cells')
    ax.set_title('trial-to-trial correlation')
    ax.set_xticks(x)
    ax.legend()
    plt.show()
    x = np.arange(num_cells)
    width = 0.35
    fig, ax = plt.subplots()
    rects1 = ax.bar(x - width/2, recorded_fano, width, label='data')
    rects2 = ax.bar(x + width/2, pred_fano, width, label='model')
    ax.set_ylabel('Fano Factor')
    ax.set_xlabel('cells')
    ax.set_xticks(x)
    ax.legend()
    plt.show()
    
def variability_error(single_trial, pred_single_trial):
    
    num_cells = single_trial.shape[-1]
    diagonal_idxs = list(range(0, num_cells*num_cells, num_cells+1))
    
    recorded_stim_corr = stim_corr2(single_trial)
    recorded_trial_corr = recorded_stim_corr.flatten()[diagonal_idxs]
    pred_stim_corr = stim_corr2(pred_single_trial)
    pred_trial_corr = pred_stim_corr.flatten()[diagonal_idxs]
    
    recorded_fano = np.nanmean(np.var(single_trial, axis=0)/np.mean(single_trial, axis=0), axis=0)
    pred_fano = np.nanmean(np.var(pred_single_trial, axis=0)/np.mean(pred_single_trial, axis=0), axis=0)
    
    error = (np.abs(recorded_trial_corr - pred_trial_corr)/np.abs(recorded_trial_corr)).sum()
    error += (np.abs(recorded_fano - pred_fano)/np.abs(recorded_fano)).sum()
    return error
    
    
def fano_contrast(model, device, contrast, n_repeats=5, poisson=[None, None, None], 
                  gaussian=[0, 0, 0, 0], thre=0, stim_type='fullfield', length=3000):
    
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
                out = noise_model(x, model, device, poisson, gaussian, thre)
                val_pred.append(out.detach().cpu().numpy())
            val_pred = np.concatenate(val_pred, axis=0)
            pred_single_trial.append(val_pred)
    pred_single_trial = np.stack(pred_single_trial)
    fano = Fano(pred_single_trial)
    
    return fano

def error_corr(single_trial, pred_single_trial, ignore_idxs=[]):
    
    n_cells = single_trial.shape[-1]
    diagonal_idxs = list(range(0, n_cells*n_cells, n_cells+1))
    noise_idxs = diagonal_idxs + ignore_idxs
    
    pred_corr = single_trial_corr_matrix(pred_single_trial)
    pred_stim_corr = stim_corr(pred_single_trial)
    pred_noise_corr = pred_corr - pred_stim_corr
    pred_corr = np.delete(pred_corr.flatten(), diagonal_idxs)
    pred_noise_corr = np.delete(pred_noise_corr.flatten(), noise_idxs)
        
    recorded_corr = single_trial_corr_matrix(single_trial)
    recorded_stim_corr = stim_corr(single_trial)
    recorded_noise_corr = recorded_corr - recorded_stim_corr
    recorded_corr = np.delete(recorded_corr.flatten(), diagonal_idxs)
    recorded_noise_corr = np.delete(recorded_noise_corr.flatten(), noise_idxs)
        
    error_corr = np.abs(pred_corr - recorded_corr).sum() / recorded_corr.sum()
    error_noise = np.abs(pred_noise_corr - recorded_noise_corr).sum() / np.abs(recorded_noise_corr).sum()
    error_noise_l2 = ((pred_noise_corr - recorded_noise_corr)**2).sum()
    
    return error_noise_l2, error_noise, error_corr

def error_corr2(single_trial, pred_single_trial, ignore_idxs=[]):
    
    n_cells = single_trial.shape[-1]
    diagonal_idxs = list(range(0, n_cells*n_cells, n_cells+1))
    noise_idxs = diagonal_idxs + ignore_idxs
    
    pred_noise_corr= noise_corr2(pred_single_trial)
    pred_noise_corr = np.delete(pred_noise_corr.flatten(), noise_idxs)
        
    recorded_noise_corr = noise_corr2(single_trial)
    recorded_noise_corr = np.delete(recorded_noise_corr.flatten(), noise_idxs)
        
    error_noise_l2 = ((pred_noise_corr - recorded_noise_corr)**2).sum()
    
    return error_noise_l2


def log_likelihood(recording, t_list, optimum_para):
    
    num_cells = len(t_list)
    lls = {'gaussian':[], 'poisson1':[], 'poisson2':[], 'binomial':[]}
    
    for cell in range(num_cells):
        dist = distribution(t_list[cell])

        ll = dist.log_likelihood('truncated_gaussian', optimum_para['gaussian'][cell], recording.single_trial_bin, cell)
        lls['gaussian'].append(ll)

        ll = dist.log_likelihood('truncated_poisson', optimum_para['poisson1'][cell], recording.single_trial_bin, cell, p_version=1)
        lls['poisson1'].append(ll)

        ll = dist.log_likelihood('truncated_poisson', optimum_para['poisson2'][cell], recording.single_trial_bin, cell, p_version=2)
        lls['poisson2'].append(ll)

        ll = dist.log_likelihood('binomial_scale', optimum_para['binomial'][cell], recording.single_trial_bin, cell)
        lls['binomial'].append(ll)
    
    return lls

def log_likelihood_plot(lls, mean=False):
    
    if mean:
        fig, ax = plt.subplots()
        bottom = 26000
        labels = lls.keys()
        x = np.arange(len(labels))
        y = np.array([sum(lls[key]) for key in lls.keys()]) + bottom
        rect = ax.bar(x, y, color='#5DA39D', width=0.5, bottom = -bottom)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=13)
        ax.set_ylabel('Log-likelihood', fontsize=13)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.set_title('Total log-likelihood', fontsize=16)
        plt.show()
    else:
        fig, ax = plt.subplots(2, 3, figsize=(20,10))
        for i in range(6):
            bottom = - min([lls[key][i] for key in lls.keys()]) + 100
            labels = lls.keys()
            x = np.arange(len(labels))
            y = np.array([lls[key][i] for key in lls.keys()]) + bottom
            rect = ax[i//3, i%3].bar(x, y, color='#5DA39D', width=0.5, bottom = -bottom)
            ax[i//3, i%3].set_xticks(x)
            ax[i//3, i%3].set_xticklabels(labels, fontsize=13)
            ax[i//3, i%3].set_ylabel('Log-likelihood', fontsize=13)
            ax[i//3, i%3].spines['right'].set_visible(False)
            ax[i//3, i%3].spines['top'].set_visible(False)
            ax[i//3, i%3].set_title('log-likelihood of cell {}'.format(i), fontsize=16)
        plt.show()

def kullback_leibler(recording, t_list, optimum_para):
    
    num_cells = len(t_list)
    kls = {'gaussian':[], 'poisson1':[], 'poisson2':[], 'binomial':[]}
    
    for cell in range(num_cells):
        dist = distribution(t_list[cell])

        kl = dist.KL('truncated_gaussian', optimum_para['gaussian'][cell], recording, cell)
        kls['gaussian'].append(kl)

        kl = dist.KL('truncated_poisson', optimum_para['poisson1'][cell], recording, cell, p_version=1)
        kls['poisson1'].append(kl)

        kl = dist.KL('truncated_poisson', optimum_para['poisson2'][cell], recording, cell, p_version=2)
        kls['poisson2'].append(kl)

        kl = dist.KL('binomial_scale', optimum_para['binomial'][cell], recording, cell)
        kls['binomial'].append(kl)
    
    return kls

def kullback_leibler_plot(kls, mean=False):
    if mean:
        fig, ax = plt.subplots()
        labels = kls.keys()
        x = np.arange(len(labels))
        y = np.array([sum(kls[key])/6 for key in kls.keys()])
        rect = ax.bar(x, y, color='#9E696D', width=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=13)
        ax.set_ylabel('KL divergence', fontsize=13)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.set_title('Mean KL divergence', fontsize=16)
        plt.show()
    else:
        fig, ax = plt.subplots(2, 3, figsize=(20,10))
        for i in range(6):
            labels = kls.keys()
            x = np.arange(len(labels))
            y = np.array([kls[key][i] for key in kls.keys()])
            rect = ax[i//3, i%3].bar(x, y, color='#9E696D', width=0.5)
            ax[i//3, i%3].set_xticks(x)
            ax[i//3, i%3].set_xticklabels(labels, fontsize=13)
            ax[i//3, i%3].set_ylabel('KL divergence', fontsize=13)
            ax[i//3, i%3].spines['right'].set_visible(False)
            ax[i//3, i%3].spines['top'].set_visible(False)
            ax[i//3, i%3].set_title('KL divergence of cell {}'.format(i), fontsize=16)
        plt.show()
    
def variance_mean(recording, t_list, optimum_para):
    
    num_cells = len(t_list)
    stats = []
    for cell in range(num_cells):
        dist = distribution(t_list[cell])
        means = []
        means_dis = []
        variances = []
        vars_dis = []
        weights = []
        for rate in range(100*t_list[cell]):
            mean, var, em, w = recording.stats_rate(100*recording.single_trial_bin.mean(0), cell, rate)
            if not np.isnan(mean):
                means.append(rate/100)
                variances.append(var)
                weights.append(w)
            try:
                r = dist.rate2para('binomial_scale', optimum_para['binomial'][cell], rate)
                var = dist.var('binomial_scale', r, optimum_para['binomial'][cell])
                means_dis.append(rate/100)
                vars_dis.append(var)
            except:
                pass
        stats.append({'means':means, 'variances':variances, 'weights':weights, 'means_dis':means_dis, 'vars_dis':vars_dis})
    return stats
        
    
def variance_mean_plot(stats, save=None, dpi=300):
    fig, ax = plt.subplots(2, 3, figsize=(16.67,10))
    num_cells = 6
    for cell in range(num_cells):
        ax[cell//3, cell%3].xaxis.set_major_locator(plt.MaxNLocator(4))
        ax[cell//3, cell%3].yaxis.set_major_locator(plt.MaxNLocator(4))
        p = ax[cell//3, cell%3].scatter(stats[cell]['means'], stats[cell]['variances'], c=stats[cell]['weights'], marker='x', cmap='cool', vmax=150, label='empirical')
        ax[cell//3, cell%3].plot(stats[cell]['means_dis'], stats[cell]['vars_dis'], '-', color='#2E8B57', label='binomial')
        ax[cell//3, cell%3].plot([0, 1], [0, 1], 'k')
        if cell == 0:
            ax[cell//3, cell%3].legend(fontsize=15, loc='best', frameon=False)
        if cell%3 == 0:
            ax[cell//3, cell%3].set_ylabel('variance', fontsize=15)
        if cell//3 == 1:
            ax[cell//3, cell%3].set_xlabel('mean', fontsize=15)
        ax[cell//3, cell%3].set_title('cell {}'.format(cell), fontsize=18)
        ax[cell//3, cell%3].spines['right'].set_visible(False)
        ax[cell//3, cell%3].spines['top'].set_visible(False)
        ax[cell//3, cell%3].set_xlim([0, max(stats[cell]['means'])+0.1])
        ax[cell//3, cell%3].tick_params(axis='both', which='major', labelsize=13)
    fig.subplots_adjust(hspace=0.3)
    fig.colorbar(p, ax=ax)
    if save != None:
        plt.savefig('/home/xhding/workspaces/torch-deep-retina/fnn/notebook/figs/'+save+'.png', dpi=dpi, bbox_inches = "tight")
    plt.show()
    
def second_stats_plot(single_trial, pred_single_trial, save=None, dpi=300):
    
    num_cells = single_trial.shape[-1]
    diagonal_idxs = list(range(0, num_cells*num_cells, num_cells+1))
    
    recorded_corr = single_trial_corr_matrix(single_trial)
    pred_corr = single_trial_corr_matrix(pred_single_trial)
    
    recorded_stim_corr = stim_corr2(single_trial)
    recorded_trial_corr = recorded_stim_corr.flatten()[diagonal_idxs]
    recorded_noise_corr = noise_corr2(single_trial)
    pred_stim_corr = stim_corr2(pred_single_trial)
    pred_trial_corr = pred_stim_corr.flatten()[diagonal_idxs]
    pred_noise_corr = noise_corr2(pred_single_trial)
    
    recorded_ave_corr = corr_matrix(single_trial.mean(0))
    pred_ave_corr = corr_matrix(pred_single_trial.mean(0))
    
    recorded_corr = np.delete(recorded_corr.flatten(), diagonal_idxs)
    pred_corr = np.delete(pred_corr.flatten(), diagonal_idxs)
    recorded_stim_corr = np.delete(recorded_stim_corr.flatten(), diagonal_idxs)
    pred_stim_corr = np.delete(pred_stim_corr.flatten(), diagonal_idxs)
    recorded_noise_corr = np.delete(recorded_noise_corr.flatten(), diagonal_idxs)
    pred_noise_corr = np.delete(pred_noise_corr.flatten(), diagonal_idxs)
    recorded_ave_corr = np.delete(recorded_ave_corr.flatten(), diagonal_idxs)
    pred_ave_corr = np.delete(pred_ave_corr.flatten(), diagonal_idxs)
    
    recorded_fano = np.nanmean(np.var(single_trial, axis=0)/np.mean(single_trial, axis=0), axis=0)
    pred_fano = np.nanmean(np.var(pred_single_trial, axis=0)/np.mean(pred_single_trial, axis=0), axis=0)
    
    fig, ax = plt.subplots(1, 3, figsize=(18, 5))
    
    ax[0].plot(recorded_corr, pred_corr, 'bo', alpha=0.7)
    ax[0].plot(recorded_corr, recorded_corr, 'r-', alpha=0.7)
    ax[0].set_xlabel('data', fontsize=16)
    ax[0].set_ylabel('model', fontsize=16)
    ax[0].set_title('total correlation', fontsize=18)
    ax[1].plot(recorded_stim_corr, pred_stim_corr, 'bo', alpha=0.7)
    ax[1].plot(recorded_stim_corr, recorded_stim_corr, 'r-', alpha=0.7)
    ax[1].set_xlabel('data', fontsize=16)
    #ax[1].set_ylabel('model', fontsize=13)
    ax[1].set_title('stimulus correlation', fontsize=18)
    ax[2].plot(recorded_noise_corr, pred_noise_corr, 'bo', alpha=0.7)
    ax[2].plot(recorded_noise_corr, recorded_noise_corr, 'r-', alpha=0.7)
    ax[2].set_xlabel('data', fontsize=16)
    #ax[2].set_ylabel('model', fontsize=13)
    ax[2].set_title('noise correlation', fontsize=18)
    for i in range(3):
        ax[i].spines['right'].set_visible(False)
        ax[i].spines['top'].set_visible(False)
        ax[i].xaxis.set_major_locator(plt.MaxNLocator(4))
        ax[i].yaxis.set_major_locator(plt.MaxNLocator(4))
        ax[i].tick_params(axis='both', which='major', labelsize=13)
    if save != None:
        plt.savefig('/home/xhding/workspaces/torch-deep-retina/fnn/notebook/figs/'+save[0]+'.png', dpi=dpi, bbox_inches = "tight")
    plt.show()
    
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    x = np.arange(num_cells)
    width = 0.35
    rects1 = ax[0].bar(x - width/2, recorded_trial_corr, width, label='data', color='#66A9C9')
    rects2 = ax[0].bar(x + width/2, pred_trial_corr, width, label='model', color='#c06f98')
    ax[0].set_ylabel('trial-to-trial correlation', fontsize=16)
    ax[0].set_xlabel('cells', fontsize=16)
    ax[0].set_xticks(x)
    #ax[0].legend(fontsize=16, loc='upper left', frameon=False)

    x = np.arange(num_cells)
    width = 0.35
    rects1 = ax[1].bar(x - width/2, recorded_fano, width, label='data', color='#66A9C9')
    rects2 = ax[1].bar(x + width/2, pred_fano, width, label='model', color='#c06f98')
    ax[1].set_ylabel('Fano factor', fontsize=16)
    ax[1].set_xlabel('cells', fontsize=16)
    ax[1].set_xticks(x)
    ax[1].legend(fontsize=13, loc='upper center', ncol=2, frameon=False)
    for i in range(2):
        ax[i].spines['right'].set_visible(False)
        ax[i].spines['top'].set_visible(False)
        ax[i].yaxis.set_major_locator(plt.MaxNLocator(4))
        ax[i].tick_params(axis='both', which='major', labelsize=13)
    if save != None:
        plt.savefig('/home/xhding/workspaces/torch-deep-retina/fnn/notebook/figs/'+save[1]+'.png', dpi=dpi, bbox_inches = "tight")
    plt.show()