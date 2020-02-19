import torch
import numpy as np
from scipy.stats import pearsonr
from kinetic.utils import *

def pearsonr_eval(model, data, n_units, reset_int, device):
    model = model.to(device)
    model.eval()
    with torch.no_grad():
        pearsons = []
        val_pred = []
        val_targ = []
        for idx, (x,y) in enumerate(data):
            if idx % reset_int == 0:
                hs = get_hs(model, 1, device)
            x = x.to(device)
            out, hs = model(x, hs)
            val_pred.append(out.detach().cpu().numpy().squeeze())
            val_targ.append(y.detach().numpy().squeeze())
        val_pred = np.stack(val_pred, axis=0)
        val_targ = np.stack(val_targ, axis=0)
        for cell in range(n_units):
            pearsons.append(pearsonr(val_pred[:,cell],val_targ[:,cell])[0])
    #model.train()
    return np.array(pearsons).mean()

def pearsonr_eval_with_responses(model, data, n_units, reset_int, device):
    model = model.to(device)
    model.eval()
    with torch.no_grad():
        pearsons = []
        val_pred = []
        val_targ = []
        for idx, (x,y) in enumerate(data):
            if idx % reset_int == 0:
                hs = get_hs(model, 1, device)
            x = x.to(device)
            out, hs = model(x, hs)
            val_pred.append(out.detach().cpu().numpy().squeeze())
            val_targ.append(y.detach().numpy().squeeze())
        val_pred = np.stack(val_pred, axis=0)
        val_targ = np.stack(val_targ, axis=0)
        for cell in range(n_units):
            pearsons.append(pearsonr(val_pred[:,cell],val_targ[:,cell])[0])
    #model.train()
    return np.array(pearsons).mean(), val_pred, val_targ
