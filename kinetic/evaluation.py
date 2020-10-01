import torch
import numpy as np
from scipy.stats import pearsonr
from kinetic.utils import *

def pearsonr_eval(model, data, n_units, device, I20=None, start_idx=0):
    model = model.to(device)
    model.eval()
    hs = get_hs(model, 1, device, I20)
    with torch.no_grad():
        pearsons = []
        val_pred = []
        val_targ = []
        for idx, (x,y) in enumerate(data):
            x = x.to(device)
            out, hs = model(x, hs)
            if idx >= start_idx:
                val_pred.append(out.detach().cpu().numpy().squeeze(0))
                val_targ.append(y.detach().numpy().squeeze(0))
        val_pred = np.stack(val_pred, axis=0)
        val_targ = np.stack(val_targ, axis=0)
        for cell in range(n_units):
            pearsons.append(pearsonr(val_pred[:,cell],val_targ[:,cell])[0])
    model.train()
    return np.array(pearsons).mean()

def pearsonr_eval_LNK(model, data, device, I20=None, start_idx=0):
    model = model.to(device)
    model.eval()
    hs = get_hs_LNK(model, 1, device, I20)
    with torch.no_grad():
        pearsons = []
        val_pred = []
        val_targ = []
        for idx, (x,y) in enumerate(data):
            x = x.to(device)
            out, hs = model(x, hs)
            if idx >= start_idx:
                val_pred.append(out.detach().cpu().numpy().squeeze(0))
                val_targ.append(y.detach().numpy().squeeze(0))
        val_pred = np.stack(val_pred, axis=0)
        val_targ = np.stack(val_targ, axis=0)

        pearsons = pearsonr(val_pred[:, 0],val_targ[:, 0])[0]
    model.train()
    return pearsons

def pearsonr_eval_2(model, data, n_units, device):
    model = model.to(device)
    model.eval()
    hs = get_hs_2(model, 1, device)
    with torch.no_grad():
        pearsons = []
        val_pred = []
        val_targ = []
        for idx, (x,y) in enumerate(data):
            x = x.to(device)
            out, hs = model(x, hs)
            val_pred.append(out.detach().cpu().numpy().squeeze(0))
            val_targ.append(y.detach().numpy().squeeze(0))
        val_pred = np.stack(val_pred, axis=0)
        val_targ = np.stack(val_targ, axis=0)
        for cell in range(n_units):
            pearsons.append(pearsonr(val_pred[:,cell],val_targ[:,cell])[0])
    model.train()
    return np.array(pearsons).mean()

def pearsonr_eval_with_responses(model, data, n_units, device):
    model = model.to(device)
    model.eval()
    hs = get_hs(model, 1, device)
    with torch.no_grad():
        pearsons = []
        val_pred = []
        val_targ = []
        for idx, (x,y) in enumerate(data):
            x = x.to(device)
            out, hs = model(x, hs)
            val_pred.append(out.detach().cpu().numpy().squeeze())
            val_targ.append(y.detach().numpy().squeeze())
        val_pred = np.stack(val_pred, axis=0)
        val_targ = np.stack(val_targ, axis=0)
        for cell in range(n_units):
            pearsons.append(pearsonr(val_pred[:,cell],val_targ[:,cell])[0])
    model.train()
    return np.array(pearsons).mean(), val_pred, val_targ

def pearsonr_eval_LNK_with_responses(model, data, device, I20=None):
    model = model.to(device)
    model.eval()
    hs = get_hs_LNK(model, 1, device, I20)
    with torch.no_grad():
        pearsons = []
        val_pred = []
        val_targ = []
        for idx, (x,y) in enumerate(data):
            x = x.to(device)
            out, hs = model(x, hs)
            val_pred.append(out.detach().cpu().numpy().squeeze(0))
            val_targ.append(y.detach().numpy().squeeze(0))
        val_pred = np.stack(val_pred, axis=0)
        val_targ = np.stack(val_targ, axis=0)

        pearsons = pearsonr(val_pred[:, 0],val_targ[:, 0])[0]
    model.train()
    return pearsons, val_pred, val_targ
