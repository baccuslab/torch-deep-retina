import os
import argparse
import numpy as np
import torch
import h5py
import itertools
from torch.utils.data.dataloader import DataLoader
from fnn.evaluation import *
import fnn.models as models
from fnn.data import TestDataset
from fnn.distributions import *
from fnn.notebook.utils import *
from fnn.noise_search_config import get_custom_cfg


parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, required=True)
parser.add_argument('--hyper', type=str, required=True)
opt = parser.parse_args()

def noise_para_search(cfg):
    
    f = open(cfg.save_path, 'w')
    f.close()
    
    cells = cfg.Data.cells
    t_list = cfg.Model.t_list
    recording = recording_stats(cfg.Data.repeats_path, cells)
    single_trial_bin = recording.single_trial_bin
    
    device = torch.device('cuda:'+str(opt.gpu))
    model_func = getattr(models, cfg.Model.name)
    model_kwargs = dict(cfg.Model)
    model = model_func(**model_kwargs).to(device)
    checkpoint_path = cfg.Model.checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    test_data = DataLoader(TestDataset(cfg), batch_size=cfg.Data.batch_size)
    _, pred, _ = pearsonr_batch_eval(model, test_data, cfg.Model.n_units, device)
    
    g0_range = np.linspace(cfg.Eval.range[0], cfg.Eval.range[1], cfg.Eval.num_points[0])
    g1_range = np.linspace(cfg.Eval.range[2], cfg.Eval.range[3], cfg.Eval.num_points[1])
    g2_range = np.linspace(cfg.Eval.range[4], cfg.Eval.range[5], cfg.Eval.num_points[2])
    
    for count, (g0, g1, g2) in enumerate(itertools.product(g0_range, g1_range, g2_range)):
        
        print(count)
        try:
            binomial_para = cfg.Model.binomial_para
            pred_single_trial_pre = model_single_trial_pre(model, test_data, device, cfg.Data.num_trials, [g0, g1, g2, 0], noise_locs=cfg.Model.noise_locs)
            poly_paras = poly_para_fit(recording, pred_single_trial_pre, pred, thre=cfg.Model.thre, threshold=cfg.Model.curve_thre, intv=cfg.Model.intv, sigma=cfg.Model.sigma)
            pred_single_trial_multi = model_single_trial_post_multi(pred_single_trial_pre, binomial_para, t_list, poly_paras, pred, n_repeats=cfg.Eval.num_repeats, thre=cfg.Model.thre)

            min_error = 10

            for i in range(cfg.Eval.num_repeats):
                pred_single_trial = pred_single_trial_multi[i]
                error, stim_error, noise_error = error_corr3(single_trial_bin, pred_single_trial, weight=cfg.Eval.weight, ignore_idxs=cfg.Eval.ignore_idxs)
                if error < min_error:
                    min_error = error
                    min_error_stim = stim_error
                    min_error_noise = noise_error
                    pred_single_trial_try = pred_single_trial
            val_error = variability_error(single_trial_bin, pred_single_trial_try)

            f = open(cfg.save_path, 'a')
            f.write(str((g0,g1,g2))+' '+str(min_error_stim)+' '+str(min_error_noise)+' '+str(val_error)+'\n')
            f.close()
        except:
            print("Something is wrong")

if __name__ == "__main__":
    cfg = get_custom_cfg(opt.hyper)
    print(cfg)
    noise_para_search(cfg)