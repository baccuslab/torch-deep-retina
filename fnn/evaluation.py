import torch
import numpy as np
from scipy.stats import pearsonr
from utils import *

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
        return np.array(pearsons).mean()
