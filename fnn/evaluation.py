import torch
import torch.nn as nn
import numpy as np
from scipy.stats import pearsonr
from fnn.utils import *

def pearsonr_batch_eval(model, data, n_units, device, cfg):
    model = model.to(device)
    model.eval()
    loss_fn = nn.PoissonNLLLoss(log_input=False).to(device)
    with torch.no_grad():
        pearsons = []
        val_pred = []
        val_targ = []
        loss = 0
        for x,y in data:
            x = x.to(device)
            out = model(x)
            loss += loss_fn(out.double(), y.double().to(device), )
            val_pred.append(out.detach().cpu().numpy())
            val_targ.append(y.detach().numpy())
        val_pred = np.concatenate(val_pred, axis=0)
        val_targ = np.concatenate(val_targ, axis=0)
        for cell in range(n_units):
            pearsons.append(pearsonr(val_pred[:,cell],val_targ[:,cell])[0])
        model.train()
        loss = loss / cfg.Data.val_size * cfg.Data.batch_size
        return np.array(pearsons).mean(), loss.item(), val_pred, val_targ
    
def pearsonr_batch_eval_cut_tail(model, data, n_units, device):
    model = model.to(device)
    model.eval()
    with torch.no_grad():
        pearsons = []
        val_pred = []
        val_targ = []
        for x,y in data:
            x[:, 0:-40] = 0.
            x = x.to(device)
            out = model(x)
            val_pred.append(out.detach().cpu().numpy())
            val_targ.append(y.detach().numpy())
        val_pred = np.concatenate(val_pred, axis=0)
        val_targ = np.concatenate(val_targ, axis=0)
        for cell in range(n_units):
            pearsons.append(pearsonr(val_pred[:,cell],val_targ[:,cell])[0])
        model.train()
        return np.array(pearsons).mean()
    
def pearsonr_eval(model, data, n_units, device):
    model = model.to(device)
    model.eval()
    with torch.no_grad():
        pearsons = []
        val_pred = []
        val_targ = []
        for x,y in data:
            x = x.to(device)
            out = model(x)
            val_pred.append(out.detach().cpu().numpy().squeeze())
            val_targ.append(y.detach().numpy().squeeze())
        val_pred = np.stack(val_pred, axis=0)
        val_targ = np.stack(val_targ, axis=0)
        for cell in range(n_units):
            pearsons.append(pearsonr(val_pred[:,cell],val_targ[:,cell])[0])
        model.train()
        return np.array(pearsons).mean()
    
def pearsonr_eval_cell(model, data, n_units, device):
    model = model.to(device)
    model.eval()
    with torch.no_grad():
        pearsons = []
        val_pred = []
        val_targ = []
        for x,y in data:
            x = x.to(device)
            out = model(x)
            val_pred.append(out.detach().cpu().numpy().squeeze())
            val_targ.append(y.detach().numpy().squeeze())
        val_pred = np.concatenate(val_pred, axis=0)
        val_targ = np.concatenate(val_targ, axis=0)
        if n_units == 1:
            return pearsonr(val_pred,val_targ)[0]
        for cell in range(n_units):
            pearsons.append(pearsonr(val_pred[:,cell],val_targ[:,cell])[0])
        model.train()
        return pearsons