import torch
import numpy as np
from scipy.stats import pearsonr
from kinetic.utils import *

def pearsonr_eval(model, data, n_units, device, I20=None, start_idx=0, hs_type='single', with_responses=False):
    train_status = model.training
    model = model.to(device)
    model.eval()
    hs = get_hs(model, 1, device, I20, hs_type)
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
    pearson = np.array(pearsons).mean()
    model.train(train_status)
    if with_responses:
        return pearson, val_pred, val_targ
    else:
        return pearson
