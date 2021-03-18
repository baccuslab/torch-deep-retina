import sys
sys.path.append('/home/xhding/workspaces/torch-deep-retina')
import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import h5py
from  torch.utils.data.dataset import Dataset
from torch.utils.data.dataloader import DataLoader
from scipy.stats import pearsonr
from scipy.optimize import minimize
from fnn.evaluation import *
from fnn.utils import select_model
from fnn.config import get_custom_cfg
from fnn.data import TestDataset, ValidationDataset
from fnn.notebook.utils import *
from torchdeepretina.datas import loadexpt

def loss_func(params):
    
    loss = error2(model, test_data, device, single_trial, recorded_fano=0.09188, 
                  n_repeats=15, n_cells=5, poisson=[None, None, params[0]], gaussian=[params[1], params[2], params[3], 0], thre=params[4])
    
    return loss

def main():
    
    global device
    global model
    global test_data
    global single_trial
    
    device = torch.device('cuda:1')
    cfg = get_custom_cfg('bn_cnn_stack')
    model = select_model(cfg, device)
    checkpoint_path = '/home/xhding/saved_model/BN_CNN_Stack/epoch_045_loss_-3.54_pearson_0.6417.pth'
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    with h5py.File('/home/TRAIN_DATA/15-10-07/naturalscene.h5', 'r') as f:
        single_trial = []
        for cell in ['cell01', 'cell02', 'cell03', 'cell04', 'cell05']:
            single_trial.append(f['test']['repeats'][cell])
        single_trial = np.stack(single_trial)
    single_trial = np.swapaxes(single_trial,0,1)
    single_trial = np.swapaxes(single_trial,1,2)[:,40:5996,:]
    
    test_data = DataLoader(TestDataset(cfg), batch_size=10000)
    
    x0 = (0.6, 0., 0., 0.34, 9.)
    #x0 = (0.57294902, 0.76393202, 0.14589803, 0.23606798, 8.81966011)
    options={'maxfev': 500, 'disp': True, 'return_all': True}
    bounds = ((0., 1.0), (-0.5, 0.5), (-0.5, 0.5), (-0.5 ,0.5), (5., 15.))
    res = minimize(loss_func, x0, method='Powell', bounds=bounds, options=options)
    
    print(res.allvecs)
    
    return

if __name__ == "__main__":
    main()