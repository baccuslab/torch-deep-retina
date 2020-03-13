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
    if cfg.Model.name == 'KineticsChannelModelFilterBipolar':
        model = KineticsChannelModelFilterBipolar(drop_p=cfg.Model.drop_p, scale_kinet=cfg.Model.scale_kinet, 
                                  recur_seq_len=cfg.Model.recur_seq_len, n_units=cfg.Model.n_units, 
                                  noise=cfg.Model.noise, bias=cfg.Model.bias, 
                                  linear_bias=cfg.Model.linear_bias, chans=cfg.Model.chans, 
                                  bn_moment=cfg.Model.bn_moment, softplus=cfg.Model.softplus, 
                                  img_shape=cfg.img_shape, ksizes=cfg.Model.ksizes).to(device)
    if cfg.Model.name == 'KineticsChannelModelFilterBipolarNoNorm':
        model = KineticsChannelModelFilterBipolarNoNorm(drop_p=cfg.Model.drop_p, scale_kinet=cfg.Model.scale_kinet, 
                                  recur_seq_len=cfg.Model.recur_seq_len, n_units=cfg.Model.n_units, 
                                  noise=cfg.Model.noise, bias=cfg.Model.bias, 
                                  linear_bias=cfg.Model.linear_bias, chans=cfg.Model.chans, 
                                  softplus=cfg.Model.softplus, img_shape=cfg.img_shape, ksizes=cfg.Model.ksizes).to(device)
    if cfg.Model.name == 'KineticsChannelModelFilterAmacrine':
        model = KineticsChannelModelFilterAmacrine(drop_p=cfg.Model.drop_p, scale_kinet=cfg.Model.scale_kinet, 
                                  recur_seq_len=cfg.Model.recur_seq_len, n_units=cfg.Model.n_units, 
                                  noise=cfg.Model.noise, bias=cfg.Model.bias, 
                                  linear_bias=cfg.Model.linear_bias, chans=cfg.Model.chans, 
                                  softplus=cfg.Model.softplus, img_shape=cfg.img_shape, ksizes=cfg.Model.ksizes).to(device)
    if cfg.Model.name == 'KineticsModel':
        model = KineticsModel(drop_p=cfg.Model.drop_p, scale_kinet=cfg.Model.scale_kinet, 
                          recur_seq_len=cfg.Model.recur_seq_len, n_units=cfg.Model.n_units, 
                          noise=cfg.Model.noise, bias=cfg.Model.bias, 
                          linear_bias=cfg.Model.linear_bias, chans=cfg.Model.chans, 
                          bn_moment=cfg.Model.bn_moment, softplus=cfg.Model.softplus, 
                          img_shape=cfg.img_shape, ksizes=cfg.Model.ksizes).to(device)
    if cfg.Model.name == 'KineticsOnePixelChannel':
        model = KineticsOnePixelChannel(recur_seq_len=cfg.Model.recur_seq_len, n_units=cfg.Model.n_units, dt=0.01,
                                    bias=cfg.Model.bias, linear_bias=cfg.Model.linear_bias, chans=cfg.Model.chans, 
                                    softplus=cfg.Model.softplus, img_shape=cfg.img_shape).to(device)
        
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
            
def LinearStack(conv_weights):
    
    for i in range(len(conv_weights)):
        if i == 0:
            current_weight = conv_weights[i].copy()
        else:
            next_weight = np.zeros((*conv_weights[0].shape[:2], conv_weights[0].shape[-1] + i * 2, conv_weights[0].shape[-1] + i * 2))
            for x in range(conv_weights[i].shape[2]):
                for y in range(conv_weights[i].shape[3]):
                    next_weight[:,:,x:x+current_weight.shape[2],y:y+current_weight.shape[3]] += np.tensordot(conv_weights[i][:,:,x,y], current_weight, axes=[-1, 0])
            current_weight = next_weight
    return current_weight

def OnePixelModel(cfg, state_dict, dt, device):
    
    model = KineticsOnePixelChannel(recur_seq_len=cfg.Model.recur_seq_len, n_units=cfg.Model.n_units, dt=dt,
                                    bias=cfg.Model.bias, linear_bias=cfg.Model.linear_bias, chans=cfg.Model.chans, 
                                    softplus=cfg.Model.softplus, img_shape=cfg.img_shape).to(device)
    
    conv_weights = []
    for i in range((cfg.Model.ksizes[0]-1)//2):
        conv_weights.append(state_dict['bipolar.0.convs.{}.weight'.format(i)].cpu().numpy())
    model.bipolar_weight.data = torch.from_numpy(LinearStack(conv_weights).sum(axis=(-1,-2))).to(device)
    model.bipolar_bias.data = state_dict['bipolar.0.convs.6.bias'].to(device)
    
    model.kinetics.ksi.data = state_dict['kinetics.ksi'].to(device)
    model.kinetics.ksr.data = state_dict['kinetics.ksr'].to(device)
    model.kinetics.ka.data = state_dict['kinetics.ka'].to(device)
    model.kinetics.kfi.data = state_dict['kinetics.kfi'].to(device)
    model.kinetics.kfr.data = state_dict['kinetics.kfr'].to(device)
    
    model.amacrine_filter.filter.data = state_dict['amacrine.1.filter'].to(device).squeeze(dim=-1)
    
    conv_weights = []
    for i in range((cfg.Model.ksizes[1]-1)//2):
        conv_weights.append(state_dict['amacrine.2.convs.{}.weight'.format(i)].cpu().numpy())
    model.amacrine_weight.data = torch.from_numpy(LinearStack(conv_weights).sum(axis=(-1,-2))).to(device)
    model.amacrine_bias.data = state_dict['amacrine.2.convs.4.bias'].to(device)
    
    model.ganglion[0].weight.data = state_dict['ganglion.0.weight'].view(cfg.Model.n_units, cfg.Model.chans[1], -1).sum(-1).to(device)
    
    model.float()
    
    return model
    
    
    