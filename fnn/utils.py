import os
import json
import torch
from collections import deque
from fnn.models import *

def get_hs(model, batch_size, device):
    hs = []
    hs.append(torch.zeros(batch_size, *model.h_shapes[0]).to(device))
    hs[0][:,0] = 1
    hs.append(deque([],maxlen=model.seq_len))
    for i in range(model.seq_len):
        hs[1].append(torch.zeros(batch_size, *model.h_shapes[1]).to(device))
    return hs

def select_model(cfg, device):
    if cfg.Model.name == 'BN_CNN_Net':
        model = BN_CNN_Net(n_units=cfg.Model.n_units, noise=cfg.Model.noise, chans=cfg.Model.chans, 
                       bn_moment=cfg.Model.bn_moment, softplus=cfg.Model.softplus, 
                       img_shape=cfg.img_shape, ksizes=cfg.Model.ksizes).to(device)
    if cfg.Model.name == 'BN_CNN_Stack':
        model = BN_CNN_Stack(n_units=cfg.Model.n_units, noise=cfg.Model.noise, chans=cfg.Model.chans, 
                       bn_moment=cfg.Model.bn_moment, softplus=cfg.Model.softplus, 
                       img_shape=cfg.img_shape, ksizes=cfg.Model.ksizes).to(device)
    if cfg.Model.name == 'BNCNN_3D':
        model = BNCNN_3D(n_units=cfg.Model.n_units, noise=cfg.Model.noise, chans=cfg.Model.chans, 
                         bn_moment=cfg.Model.bn_moment, softplus=cfg.Model.softplus, 
                         img_shape=cfg.img_shape, ksizes=cfg.Model.ksizes, 
                         strides=cfg.Model.strides).to(device)
    if cfg.Model.name == 'BNCNN_3D2':
        model = BNCNN_3D2(n_units=cfg.Model.n_units, noise=cfg.Model.noise, chans=cfg.Model.chans, 
                         bn_moment=cfg.Model.bn_moment, softplus=cfg.Model.softplus, 
                         img_shape=cfg.img_shape, ksizes=cfg.Model.ksizes, 
                         strides=cfg.Model.strides).to(device)
    if cfg.Model.name == 'BNCNN_3D2_Stack':
        model = BNCNN_3D2_Stack(n_units=cfg.Model.n_units, 
                                noise=cfg.Model.noise, chans=cfg.Model.chans, 
                                bn_moment=cfg.Model.bn_moment, softplus=cfg.Model.softplus, 
                                img_shape=cfg.img_shape, ksizes=cfg.Model.ksizes, 
                                strides=cfg.Model.strides, filter_mod=cfg.Model.filter_mod).to(device)
    return model

def update_eval_history(cfg, epoch, pearson, epoch_loss, eval_loss):
    eval_history_path = os.path.join(cfg.save_path, cfg.exp_id, 'eval.json')
    if not os.path.exists(eval_history_path):
        eval_history = []
    else: 
        with open(eval_history_path, 'r') as f:
            eval_history = json.load(f)
    eval_history.append({'epoch' : epoch, 'pearson': pearson, 'loss': epoch_loss, 'eval_loss': eval_loss})
    with open(eval_history_path, 'w') as f:
            json.dump(eval_history, f)