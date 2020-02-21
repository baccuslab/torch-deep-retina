import os
import json
import torch
from collections import deque
from kinetic.models import *

def get_hs(model, batch_size, device):
    hs = []
    hs.append(torch.zeros(batch_size, *model.h_shapes[0]).to(device))
    hs[0][:,0] = 1
    hs.append(deque([],maxlen=model.seq_len))
    for i in range(model.seq_len):
        hs[1].append(torch.zeros(batch_size, *model.h_shapes[1]).to(device))
    return hs

def select_model(cfg, device):
    
    if cfg.Model.name == 'KineticsChannelModel':
        model = KineticsChannelModel(drop_p=cfg.Model.drop_p, scale_kinet=cfg.Model.scale_kinet, 
                                  recur_seq_len=cfg.Model.recur_seq_len, n_units=cfg.Model.n_units, 
                                  noise=cfg.Model.noise, bias=cfg.Model.bias, 
                                  linear_bias=cfg.Model.linear_bias, chans=cfg.Model.chans, 
                                  bn_moment=cfg.Model.bn_moment, softplus=cfg.Model.softplus, 
                                  img_shape=cfg.img_shape, ksizes=cfg.Model.ksizes).to(device)
    if cfg.Model.name == 'KineticsChannelModelFilter':
        model = KineticsChannelModelFilter(drop_p=cfg.Model.drop_p, scale_kinet=cfg.Model.scale_kinet, 
                                  recur_seq_len=cfg.Model.recur_seq_len, n_units=cfg.Model.n_units, 
                                  noise=cfg.Model.noise, bias=cfg.Model.bias, 
                                  linear_bias=cfg.Model.linear_bias, chans=cfg.Model.chans, 
                                  bn_moment=cfg.Model.bn_moment, softplus=cfg.Model.softplus, 
                                  img_shape=cfg.img_shape, ksizes=cfg.Model.ksizes).to(device)
    if cfg.Model.name == 'KineticsChannelModelFilterNoClamp':
        model = KineticsChannelModelFilterNoClamp(drop_p=cfg.Model.drop_p, scale_kinet=cfg.Model.scale_kinet, 
                                  recur_seq_len=cfg.Model.recur_seq_len, n_units=cfg.Model.n_units, 
                                  noise=cfg.Model.noise, bias=cfg.Model.bias, 
                                  linear_bias=cfg.Model.linear_bias, chans=cfg.Model.chans, 
                                  bn_moment=cfg.Model.bn_moment, softplus=cfg.Model.softplus, 
                                  img_shape=cfg.img_shape, ksizes=cfg.Model.ksizes).to(device)
    if cfg.Model.name == 'KineticsChannelModelNoBatchnorm':
        model = KineticsChannelModelNoBatchnorm(drop_p=cfg.Model.drop_p, scale_kinet=cfg.Model.scale_kinet, 
                                  recur_seq_len=cfg.Model.recur_seq_len, n_units=cfg.Model.n_units, 
                                  noise=cfg.Model.noise, bias=cfg.Model.bias, 
                                  linear_bias=cfg.Model.linear_bias, chans=cfg.Model.chans, 
                                  softplus=cfg.Model.softplus, 
                                  img_shape=cfg.img_shape, ksizes=cfg.Model.ksizes).to(device)
    if cfg.Model.name == 'KineticsChannelModelLayerNorm':
        model = KineticsChannelModelLayerNorm(drop_p=cfg.Model.drop_p, scale_kinet=cfg.Model.scale_kinet, 
                                  recur_seq_len=cfg.Model.recur_seq_len, n_units=cfg.Model.n_units, 
                                  bias=cfg.Model.bias, 
                                  linear_bias=cfg.Model.linear_bias, chans=cfg.Model.chans, 
                                  softplus=cfg.Model.softplus, 
                                  img_shape=cfg.img_shape, ksizes=cfg.Model.ksizes).to(device)
    if cfg.Model.name == 'KineticsChannelModelInstanceNorm':
        model = KineticsChannelModelInstanceNorm(drop_p=cfg.Model.drop_p, scale_kinet=cfg.Model.scale_kinet, 
                                  recur_seq_len=cfg.Model.recur_seq_len, n_units=cfg.Model.n_units, 
                                  bias=cfg.Model.bias, 
                                  linear_bias=cfg.Model.linear_bias, chans=cfg.Model.chans, 
                                  softplus=cfg.Model.softplus, 
                                  img_shape=cfg.img_shape, ksizes=cfg.Model.ksizes).to(device)
    if cfg.Model.name == 'KineticsModel':
        model = KineticsModel(drop_p=cfg.Model.drop_p, scale_kinet=cfg.Model.scale_kinet, 
                          recur_seq_len=cfg.Model.recur_seq_len, n_units=cfg.Model.n_units, 
                          noise=cfg.Model.noise, bias=cfg.Model.bias, 
                          linear_bias=cfg.Model.linear_bias, chans=cfg.Model.chans, 
                          bn_moment=cfg.Model.bn_moment, softplus=cfg.Model.softplus, 
                          img_shape=cfg.img_shape, ksizes=cfg.Model.ksizes).to(device)
        
    return model

def update_eval_history(cfg, epoch, pearson, epoch_loss):
    eval_history_path = os.path.join(cfg.save_path, cfg.exp_id, 'eval.json')
    if not os.path.exists(eval_history_path):
        eval_history = []
    else: 
        with open(eval_history_path, 'r') as f:
            eval_history = json.load(f)
    eval_history.append({'epoch' : epoch, 'pearson': pearson, 'loss': epoch_loss})
    with open(eval_history_path, 'w') as f:
            json.dump(eval_history, f)