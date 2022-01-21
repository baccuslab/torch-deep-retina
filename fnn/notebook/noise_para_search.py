import os
import argparse
import numpy as np
import torch
import h5py
import itertools
from  torch.utils.data.dataset import Dataset
from torch.utils.data.dataloader import DataLoader
from fnn.evaluation import *
from fnn.utils import select_model
from fnn.config import get_custom_cfg
from fnn.data import TestDataset
from fnn.distributions import *
from fnn.notebook.utils import *


parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, required=True)
opt = parser.parse_args()

def noise_para_search():
    
    f = open('./errors.txt', 'w')
    f.close()
    
    file_path = '/home/xhding/tem_stim/21-03-15/naturalscene.h5'
    cells = [0,1,2,3,4,6]
    t_list = [3,3,3,3,2,3]
    recording = recording_stats(file_path, cells)
    single_trial_bin = recording.single_trial_bin
    
    device = torch.device('cuda:'+str(opt.gpu))
    cfg = get_custom_cfg('bn_cnn_stack_try')
    model = select_model(cfg, device)
    checkpoint_path = '/home/xhding/saved_model/BN_CNN_Stack/epoch_070_loss_-3.54_pearson_0.6845.pth'
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    test_data = DataLoader(TestDataset(cfg), batch_size=10000)
    test_pc, _, pred, targ = pearsonr_batch_eval(model, test_data, 6, device, cfg)
    
    g0_range = np.linspace(0.,1.4, 11)
    g1_range = np.linspace(0.,0.2, 11)
    g2_range = np.linspace(0.,0.3, 11)
    
    for g0, g1, g2 in itertools.product(g0_range, g1_range, g2_range):
        
        binomial_para = [2.17, 2.85, 2.5, 2.5, 1.0, 0.47]
        error_stats_post = []
        pred_single_trial_pre = model_single_trial_pre(model, test_data, device, 15, [g0, g1, g2, 0])
        poly_paras = poly_para_fit(recording, pred_single_trial_pre, t_list)
        pred_single_trial_multi = model_single_trial_post_multi(pred_single_trial_pre, binomial_para, t_list, poly_paras, pred, n_repeats=100)

        min_error = 10

        for i in range(100):
            pred_single_trial = pred_single_trial_multi[i]
            error = error_corr2(single_trial_bin, pred_single_trial, [15,20])
            if error < min_error:
                min_error = error
        f = open('./errors.txt', 'a')
        f.write(str((g0,g1,g2))+' '+str(min_error)+'\n')
        f.close()

if __name__ == "__main__":
    noise_para_search()