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
from fnn.evaluation import *
from fnn.utils import select_model
from fnn.config import get_custom_cfg
from fnn.data import TestDataset, ValidationDataset
from fnn.notebook.utils import *
from torchdeepretina.datas import loadexpt

def main():
    
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
    
    test_data = DataLoader(TestDataset(cfg), batch_size=512)
    
    inits = {}
    inits['k3'] = 0.3
    inits['std2'] = 0.5
    inits['thre'] = 10.
    steps = {}
    steps['k3'] = 0.1
    steps['std2'] = 0.1
    steps['thre'] = 0.5
    temps = np.geomspace(0.2, 0.02, 100)
    results, error = simulated_annealing(model, test_data, device, inits, steps, temps, single_trial)
    print(results, error)
    
    return

if __name__ == "__main__":
    main()