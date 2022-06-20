import os
import json
import random
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from scipy.stats import pearsonr
from scipy.optimize import curve_fit
import torchdeepretina.stimuli as stim
from torchdeepretina.torch_utils import Flatten
from fnn.distributions import *

def noise_model_pre_all(x, model, device, gaussian=[0, 0, 0, 0], noise_locs=[3, 4, 5]):
    with torch.no_grad():
        noise = gaussian[0] * torch.randn(x.size()).to(device)
        out = x + noise
        out = model.bipolar[:noise_locs[0]](out)
        noise = gaussian[1] * torch.randn(out.size()).to(device)
        out = out + noise
        out = model.bipolar[noise_locs[0]:](out)
        out = model.amacrine[:noise_locs[1]](out)
        noise = gaussian[2] * torch.randn(out.size()).to(device)
        out = out + noise
        out = model.amacrine[noise_locs[1]:](out)
        out = model.ganglion[:noise_locs[2]](out)
        noise = gaussian[3] * torch.randn(out.size()).to(device)
        out = out + noise
        out = model.ganglion[noise_locs[2]:-1](out)
    return out

def model_single_trial_pre_all(model, data, device, n_repeats=15, gaussian=[0, 0, 0, 0], seed=None, noise_locs=[3, 4, 5]):
    
    if seed != None:
        torch.manual_seed(seed)
    model = model.to(device)
    with torch.no_grad():
        pred_single_trial = []
        for n in range(n_repeats):
            val_pred = []
            for x, _ in data:
                x = x.to(device)
                out = noise_model_pre_all(x, model, device, gaussian, noise_locs)
                val_pred.append(out)
            val_pred = torch.cat(val_pred, dim=0)
            pred_single_trial.append(val_pred)
        pred_single_trial = torch.stack(pred_single_trial)
    pred_single_trial = pred_single_trial.detach().cpu().numpy()
    return pred_single_trial

def model_single_trial_post_multi_all(pred_single_trial_pre_all, binomial_para, t_list, poly_paras, pred, n_repeats=5, thre=1, seed=None):
    
    if seed != None:
        np.random.seed(seed)
        
    pred_single_trial = np.zeros((n_repeats, *pred_single_trial_pre_all.shape)).astype(np.int8)
    num_cells = pred_single_trial_pre_all.shape[2]
    for cell in range(num_cells):
        dist = distribution(t_list[cell])
        pred_scale = np.polyval(poly_paras[cell], pred_single_trial_pre_all[:, :, cell, :, :]/100)*100
        for rate in range((t_list[cell]-1)*100):
            indices = np.where((pred_scale>=rate-0.5)*(pred_scale<rate+0.5))
            num = indices[0].shape[0]
            if num == 0:
                continue
            r = dist.rate2para('binomial_scale', binomial_para[cell], rate)
            p = [dist.binomial_scale(i, r, binomial_para[cell]) for i in range(t_list[cell])]
            spikes = np.random.choice(t_list[cell], num*n_repeats, p=p).reshape((n_repeats, num)).astype(np.int8)
            for n in range(n_repeats):
                pred_single_trial[n, :, :, cell, :, :][indices] = spikes[n]
        indices = np.where(pred_scale>=(t_list[cell]-1)*100-0.5)
        num = indices[0].shape[0]
        if num != 0:
            r = dist.rate2para('binomial_scale', binomial_para[cell], (t_list[cell]-1)*100)
            p = [dist.binomial_scale(i, r, binomial_para[cell]) for i in range(t_list[cell])]
            spikes = np.random.choice(t_list[cell], num*n_repeats, p=p).reshape((n_repeats, num)).astype(np.int8)
            for n in range(n_repeats):
                pred_single_trial[n, :, :, cell, :, :][indices] = spikes[n]
        #pred_single_trial[:, :, pred_scale.mean(0)<thre, cell] = 0

    pred_single_trial[:, :, pred<thre, :, :] = 0
    #pred_single_trial[:, :, pred_single_trial.mean((0,1))<thre/100] = 0
    pred_single_trial = pred_single_trial.astype(np.int8)
    return pred_single_trial

def noise_cov_big(single_trial):
    noise = single_trial - single_trial.mean(0)
    shape = noise.shape
    noise = np.swapaxes(noise.reshape(shape[0]*shape[1], shape[2]), 0, 1)
    cov = np.cov(noise)
    return cov

def noise_model_pre_repeats_all(model, x, device, n_repeats=15, gaussian=[0, 0, 0, 0], seed=None, noise_locs=[3, 4, 5]):
    
    if seed != None:
        torch.manual_seed(seed)
    model = model.to(device)
    with torch.no_grad():
        pred_single_trial = []
        for n in range(n_repeats):
            x = x.to(device)
            out = noise_model_pre_all(x, model, device, gaussian, noise_locs)
            pred_single_trial.append(out)
        pred_single_trial = torch.stack(pred_single_trial)
    pred_single_trial = pred_single_trial.detach().cpu().numpy()
    return pred_single_trial

def compute_cov_eigen(model, x, device, binomial_para, t_list, poly_paras, pred,
                      n_repeats=1500, gaussian=[0.,0.,0.,0.], thre=1, seed=None):
    
    x = x.to(device)
    resp_pre = noise_model_pre_repeats_all(model, x, device, n_repeats=n_repeats, gaussian=gaussian, seed=seed, noise_locs=[3, 4, 5])
    resp = model_single_trial_post_multi_all(resp_pre, binomial_para, t_list, poly_paras, pred, n_repeats=1, seed=seed, thre=thre)[0]
    noise_cov = noise_cov_big(resp.reshape(n_repeats, 1, -1))
    w, v = np.linalg.eigh(noise_cov)
    return w, v

def compute_grad_norm(model, x, device, v):
    
    n_units = model.n_units
    
    for p in model.parameters():
        p.requires_grad = False
    
    x.requires_grad = True

    modules = []
    modules.append(Flatten())
    modules.append(nn.Linear(n_units*18*18, 1, bias=False))
    mode = nn.Sequential(*modules)

    out = model.bipolar(x)
    out = model.amacrine(out)
    out = model.ganglion[:-1](out)

    norm_grads = []
    for idx in range(n_units*18*18):
        mode[1].weight.data[0, :] = torch.from_numpy(v[:, idx])
        mode.to(device)
        for p in mode.parameters():
            p.requires_grad = False

        mode_out = mode(out)[0, 0]

        x.grad = None
        mode_out.backward(retain_graph=True)
        norm_grads.append(torch.sqrt((x.grad**2).sum()).cpu().numpy())

    norm_grads = np.array(norm_grads)
    return norm_grads

def compute_grads(model, x, device, v):
    
    n_units = model.n_units
    
    for p in model.parameters():
        p.requires_grad = False

    x.requires_grad = True

    modules = []
    modules.append(Flatten())
    modules.append(nn.Linear(n_units*18*18, 1, bias=False))
    mode = nn.Sequential(*modules)

    out = model.bipolar(x)
    out = model.amacrine(out)
    out = model.ganglion[:-1](out)

    grads = []
    for idx in range(n_units*18*18):
        mode[1].weight.data[0, :] = torch.from_numpy(v[:, idx])
        mode.to(device)
        for p in mode.parameters():
            p.requires_grad = False

        mode_out = mode(out)[0, 0]

        x.grad = None
        mode_out.backward(retain_graph=True)
        grad = x.grad.data
        grads.append(grad.cpu().numpy())
    grads = np.array(grads).squeeze()
    return grads

def mode_inst_RF(model, x, device, mode_vec):
    
    n_units = model.n_units
    
    for p in model.parameters():
        p.requires_grad = False
    
    x.requires_grad = True

    modules = []
    modules.append(Flatten())
    modules.append(nn.Linear(n_units*18*18, 1, bias=False))
    mode = nn.Sequential(*modules)

    out = model.bipolar(x)
    out = model.amacrine(out)
    out = model.ganglion[:-1](out)

    mode[1].weight.data[0, :] = torch.from_numpy(mode_vec)
    mode.to(device)
    for p in mode.parameters():
        p.requires_grad = False

    mode_out = mode(out)[0, 0]

    x.grad = None
    mode_out.backward()
    return x.grad.cpu().numpy()[0]

def cosines(model, x, device, v, step=1e-3):
    
    n_units = model.n_units
    
    for p in model.parameters():
        p.requires_grad = False

    x.requires_grad = True

    modules = []
    modules.append(Flatten())
    modules.append(nn.Linear(n_units*18*18, 1, bias=False))
    mode = nn.Sequential(*modules)

    out = model.bipolar(x)
    out = model.amacrine(out)
    out = model.ganglion[:-1](out)

    cosines = []
    for idx in range(n_units*18*18):
        mode[1].weight.data[0, :] = torch.from_numpy(v[:, idx])
        mode.to(device)
        for p in mode.parameters():
            p.requires_grad = False

        mode_out = mode(out)[0, 0]

        x.grad = None
        mode_out.backward(retain_graph=True)
        grad = x.grad.data
        with torch.no_grad():
            x_new = x + grad * step
            out_new = model.bipolar(x_new)
            out_new = model.amacrine(out_new)
            out_new = model.ganglion[:-1](out_new)
            dev = (out_new - out).data.cpu().numpy().flatten()
        dev /= np.linalg.norm(dev, 2)
        cosines.append(np.abs(v[:, idx].dot(dev)))
    cosines = np.array(cosines)
    return cosines

def ortho_component_norm(grads, top_num=25):
    
    num_modes = grads.shape[0]
    grad_mat = np.swapaxes(grads.reshape(num_modes, -1), 0, 1)
    grad_mat_top = grad_mat[:, -top_num:]
    pure_grads = []
    for mode_idx in range(grad_mat_top.shape[1]):
        other_grads = np.delete(grad_mat_top, mode_idx, 1)
        q = np.linalg.qr(other_grads)[0]
        pure_grad = grad_mat_top[:, mode_idx].copy()
        for idx in range(q.shape[1]):
            pure_grad -= grad_mat_top[:, mode_idx].dot(q[:,-idx-1]) * q[:,-idx-1]

        pure_grads.append(np.linalg.norm(pure_grad, 2))
    pure_grads = np.array(pure_grads)
    
    return pure_grads

def coses_grads(grads, top_num=10): 
    coses = []
    for i in range(top_num):
        for j in range(i+1, top_num):
            a = grads[-i-1].flatten()
            b = grads[-j-1].flatten()
            a /= np.linalg.norm(a, 2)
            b /= np.linalg.norm(b, 2)
            cos = a.dot(b)
            coses.append(cos)
    coses = np.array(coses)
    return coses
