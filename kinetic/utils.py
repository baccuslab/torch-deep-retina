import os
import json
import torch
import torch.nn as nn
from scipy import signal
from collections import deque
from kinetic.models import *
from torchdeepretina.intracellular import load_interneuron_data, max_correlation
import torchdeepretina.stimuli as tdrstim

def get_hs(model, batch_size, device):
    hs = []
    hs.append(torch.zeros(batch_size, *model.h_shapes[0]).to(device))
    hs[0][:,0] = 1
    hs.append(deque([],maxlen=model.seq_len))
    for i in range(model.seq_len):
        hs[1].append(torch.zeros(batch_size, *model.h_shapes[1]).to(device))
    return hs

def get_hs_2(model, batch_size, device):
    hs = []
    hs.append(torch.zeros(batch_size, *model.h_shapes[0]).to(device))
    hs[0][:,3] = 100.
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
                                        scale_kinet=cfg.Model.scale_kinet, bias=cfg.Model.bias, 
                                        linear_bias=cfg.Model.linear_bias, chans=cfg.Model.chans, 
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
                                    scale_kinet=cfg.Model.scale_kinet, bias=cfg.Model.bias, 
                                    linear_bias=cfg.Model.linear_bias, chans=cfg.Model.chans, 
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
    
    if cfg.Model.scale_kinet:
        model.kinet_scale.scale_param.data = state_dict['kinet_scale.scale_param'].to(device)
        model.kinet_scale.shift_param.data = state_dict['kinet_scale.shift_param'].to(device)
    
    model.amacrine_filter.filter.data = state_dict['amacrine.1.filter'].to(device).squeeze(dim=-1)
    
    conv_weights = []
    for i in range((cfg.Model.ksizes[1]-1)//2):
        conv_weights.append(state_dict['amacrine.2.convs.{}.weight'.format(i)].cpu().numpy())
    model.amacrine_weight.data = torch.from_numpy(LinearStack(conv_weights).sum(axis=(-1,-2))).to(device)
    model.amacrine_bias.data = state_dict['amacrine.2.convs.4.bias'].to(device)
    
    model.ganglion[0].weight.data = state_dict['ganglion.0.weight'].view(cfg.Model.n_units, cfg.Model.chans[1], -1).sum(-1).to(device)
    
    model.float()
    
    return model

def temporal_frequency_normalized_loss(y_pred, y_targ, loss_fn, device, cut_off=4, num_units=1, filter_len=11, dt=0.01):
    
    numtaps = filter_len
    f = cut_off
    fs = int(1./dt)
    conv_filters = {}
    for pass_zero in ['lowpass', 'highpass']:
        lp_filter = signal.firwin(numtaps, f, pass_zero=pass_zero, fs=fs)
        conv_filter = nn.Conv1d(num_units, num_units, filter_len, groups=num_units, bias=False)
        conv_filter.weight.data = torch.from_numpy(np.flip(lp_filter).copy())[None, None, :].repeat(num_units, 1, 1)
        conv_filter.weight.requires_grad = False
        conv_filter = conv_filter.to(device)
        conv_filters[pass_zero] = conv_filter
    
    y_pred_low = conv_filters['lowpass'](y_pred)
    y_pred_high = conv_filters['highpass'](y_pred)
    y_targ_low = conv_filters['lowpass'](y_targ)
    y_targ_high = conv_filters['highpass'](y_targ)
    
    low_std = torch.std(y_targ_low, dim=-1)[:, :, None]
    high_std = torch.std(y_targ_high, dim=-1)[:, :, None]
    
    y_pred_low_norm = y_pred_low / low_std
    y_pred_high_norm = y_pred_high / high_std
    y_targ_low_norm = y_targ_low / low_std
    y_targ_high_norm = y_targ_high / high_std
    
    loss = loss_fn(y_pred_low_norm, y_targ_low_norm)
    loss += loss_fn(y_pred_high_norm, y_targ_high_norm)
    
    return loss
    
def interneuron_correlation_bipolar(model, root_path, files, stim_keys, length, device):
    
    stim_dict, mem_pot_dict, _ = load_interneuron_data(root_path, files, 40, stim_keys)
    intr_cors = {
                "cell_file":[], 
                "stim_type":[],
                "cell_type":[],
                "cor":[]
                }
    for cell_file in stim_dict.keys():
        for stim_type in stim_dict[cell_file].keys():
            print(cell_file, stim_type)
            stim = tdrstim.spatial_pad(stim_dict[cell_file][stim_type], model.img_shape[1])
            stim = tdrstim.rolling_window(stim, model.img_shape[0])[:length].astype(np.float32)

            with torch.no_grad():
                stim_tensor = torch.from_numpy(stim).to(device)
                resp = model.bipolar[0](stim_tensor)
                resp = resp.detach().cpu().numpy()
            pots = mem_pot_dict[cell_file][stim_type][:, :length]
            rnge = range(len(pots))

            for cell_idx in rnge:
                r = max_correlation(pots[cell_idx], resp)
                intr_cors['stim_type'].append(stim_type)
                cell_type = cell_file.split("/")[-1].split("_")[0][:-1]
                intr_cors['cell_type'].append(cell_type) # amacrine or bipolar
                intr_cors['cor'].append(r)
    return intr_cors
    
    
def slow_parameters_solver(A_l, A_h, decay_rate, kfi, kfr, ka, u_l, u_h):
    
    I1_l = kfi / kfr * A_l
    I1_h = kfi / kfr * A_h
    delta_I2_l = 1 - I1_l * (kfi*u_l*ka + kfr*u_l*ka + kfr*kfi) / (kfi*u_l*ka)
    delta_I2_h = 1 - I1_h * (kfi*u_h*ka + kfr*u_h*ka + kfr*kfi) / (kfi*u_h*ka)
    #alpha = u_h * I1_l / u_l / I1_h
    alpha = I1_l / I1_h
    I20 = (delta_I2_l - alpha*delta_I2_h) / (alpha - 1)
    ksi = decay_rate / ((kfi*u_h*ka)/(kfi*u_h*ka + kfr*u_h*ka + kfr*kfi)+I1_h/(I20 + delta_I2_h))
    ksr = I1_h / (I20 + delta_I2_h) * ksi
    return ksi, ksr, I20
    