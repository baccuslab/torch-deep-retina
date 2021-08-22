import os
import json
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from scipy import signal
from collections import deque
import pyret.filtertools as ft
from pyret.stimulustools import slicestim
from pyret.utils import flat2d
from pyret.nonlinearities import Binterp, Sigmoid
from kinetic.models import *
from torchdeepretina.intracellular import load_interneuron_data, max_correlation
import torchdeepretina.stimuli as tdrstim
from kinetic.custom_modules import Weighted_Poisson_MSE

def get_hs(model, batch_size, device, I20=None, mode='single'):
    if mode == 'single':
        hs = torch.zeros(batch_size, *model.h_shapes).to(device)
        hs[:,0] = 1
        if isinstance(I20, np.ndarray):
            hs[:,3] = torch.from_numpy(I20)[:,None].to(device)
    elif mode == 'multiple':
        hs = []
        hs.append(torch.zeros(batch_size, *model.h_shapes[0]).to(device))
        hs[0][:,0] = 1
        if isinstance(I20, np.ndarray):
            hs[0][:,3] = torch.from_numpy(I20)[:,None].to(device)
        hs.append(deque([],maxlen=model.seq_len))
        for i in range(model.seq_len):
            hs[1].append(torch.zeros(batch_size, *model.h_shapes[1]).to(device))
    elif mode == 'double':
        hs1 = torch.zeros(batch_size, *model.h_shapes).to(device)
        hs1[:,0] = 1
        if isinstance(I20[0], np.ndarray):
            hs1[:,3] = torch.from_numpy(I20[0])[:,None].to(device)
        h_shapes = list(model.h_shapes)
        h_shapes[1] = 1
        h_shapes = tuple(h_shapes)
        hs2 = torch.zeros(batch_size, *h_shapes).to(device)
        hs2[:,0] = 1
        if isinstance(I20[1], np.ndarray):
            hs2[:,3] = torch.from_numpy(I20[1])[:,None].to(device)
        hs = (hs1, hs2)
    else:
        raise Exception('Invalid mode')
    return hs

def detach_hs(hs, mode='single', seq_len=None):
    if mode == 'single':
        hs_new = hs.detach()
    elif mode == 'multiple':
        hs_new = []
        hs_new.append(hs[0].detach())
        hs_new.append(deque([h.detach() for h in hs[1]], seq_len))
    elif mode == 'double':
        hs_new = (hs[0].detach(), hs[1].detach())
    return hs_new

def select_lossfn(loss='poisson'):
    if loss == 'poisson':
        return nn.PoissonNLLLoss(log_input=False)
    if loss == 'mse':
        return nn.MSELoss()
    if loss == 'mix':
        return Weighted_Poisson_MSE()
    
def init_params(model, device):
    if model.name == 'LNK':
        model.bias.data = -4 * torch.ones(1).to(device)
    if model.name == 'KineticsModel' or model.name == 'KineticsChannelModelFilterBipolarNoNorm':
        model.bipolar[0].convs[6].bias.data = -4. * torch.ones(model.chans[0]).to(device)
        
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

def OnePixelModelMulti(cfg, state_dict, device):
    
    model_kwargs = dict(cfg.Model)
    model = KineticsOnePixelChannel(**model_kwargs).to(device)
    
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

def OnePixelModel(cfg, state_dict, device):
    
    model_kwargs = dict(cfg.Model)
    model = KineticsOnePixel(**model_kwargs).to(device)
    
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
    if model.ka_offset:
        model.kinetics.ka_2.data = state_dict['kinetics.ka_2'].to(device)
    if model.ksr_gain:
        model.kinetics.ksr_2.data = state_dict['kinetics.ksr_2'].to(device)
    
    model.kinetics_w.data = state_dict['kinetics_w'].to(device)
    model.kinetics_b.data = state_dict['kinetics_b'].to(device)
    
    conv_weights = []
    for i in range((cfg.Model.ksizes[1]-1)//2):
        conv_weights.append(state_dict['amacrine.1.convs.{}.weight'.format(i)].cpu().numpy())
    model.amacrine_weight.data = torch.from_numpy(LinearStack(conv_weights).sum(axis=(-1,-2))).to(device)
    model.amacrine_bias.data = state_dict['amacrine.1.convs.4.bias'].to(device)
    
    model.ganglion[0].weight.data = state_dict['ganglion.0.weight'].view(cfg.Model.n_units, cfg.Model.chans[1], -1).sum(-1).to(device)
    
    model.float()
    
    return model

def OneDimModel(cfg, state_dict, device):
    
    model_kwargs = dict(cfg.Model)
    model = KineticsModel1D(**model_kwargs).to(device)
    
    conv_weights = []
    for i in range((cfg.Model.ksizes[0]-1)//2):
        conv_weights.append(state_dict['bipolar.0.convs.{}.weight'.format(i)].cpu().numpy())
    model.bipolar[0].weight.data = torch.from_numpy(LinearStack(conv_weights).sum(axis=(-2))).to(device)
    model.bipolar[0].bias.data = state_dict['bipolar.0.convs.6.bias'].to(device)
    
    model.kinetics.ksi.data = state_dict['kinetics.ksi'].to(device)
    model.kinetics.ksr.data = state_dict['kinetics.ksr'].to(device)
    model.kinetics.ka.data = state_dict['kinetics.ka'].to(device)
    model.kinetics.kfi.data = state_dict['kinetics.kfi'].to(device)
    model.kinetics.kfr.data = state_dict['kinetics.kfr'].to(device)
    
    if model.ka_offset:
        model.kinetics.ka_2.data = state_dict['kinetics.ka_2'].to(device)
    if model.ksr_gain:
        model.kinetics.ksr_2.data = state_dict['kinetics.ksr_2'].to(device)
    
    model.kinetics_w.data = state_dict['kinetics_w'].to(device)
    model.kinetics_b.data = state_dict['kinetics_b'].to(device)
    
    conv_weights = []
    for i in range((cfg.Model.ksizes[1]-1)//2):
        conv_weights.append(state_dict['amacrine.1.convs.{}.weight'.format(i)].cpu().numpy())
    model.amacrine[1].weight.data = torch.from_numpy(LinearStack(conv_weights).sum(axis=(-2))).to(device)
    model.amacrine[1].bias.data = state_dict['amacrine.1.convs.4.bias'].to(device)
    
    shape0 = int(np.sqrt(state_dict['ganglion.0.weight'].shape[1] // cfg.Model.chans[1]))
    model.ganglion[0].weight.data = state_dict['ganglion.0.weight'].view(cfg.Model.n_units, cfg.Model.chans[1], shape0, shape0).sum(-2).view(cfg.Model.n_units, cfg.Model.chans[1] * shape0).to(device)
    
    model.float()
    
    return model

def temporal_frequency_normalized_loss(y_pred, y_targ, loss_fn, device, cut_off=8, num_units=1, filter_len=25, dt=0.01):
    
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
    
    low_std[low_std < 5.] = 5.
    high_std[high_std < 5.] = 5.
    
    y_pred_low_norm = y_pred_low / (low_std)
    y_pred_high_norm = y_pred_high / (high_std)
    y_targ_low_norm = y_targ_low / (low_std)
    y_targ_high_norm = y_targ_high / (high_std)
    
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

def interneuron_correlation_occupancy(model, root_path, files, stim_keys, length, device, I20=None):
    
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
                hs = get_hs(model, 1, device, I20, 'multiple')
                layer_outs = inspect_rnn(model, stim_tensor, hs, ['kinetics'])
                kinetics_history = [h[1].detach().cpu().numpy().mean(-1) for h in layer_outs['kinetics']]
                kinetics_history = np.concatenate(kinetics_history, axis=0)
                resp = kinetics_history[:, 1]
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

def adaptive_index(response, l_start=1960):
    re = response[l_start+50:l_start+300].mean(0)
    rl = response[-250:].mean(0)
    return (re - rl) / (re + rl)

def adaptive_index_bipolar(layer_outs):
    responses = layer_outs['spiking_block2']
    shape = responses.shape
    responses = responses.reshape((shape[0], shape[1], 36, 36))
    fig, axs = plt.subplots(2, 4, figsize=(10,5))
    for i in range(shape[1]//2):
        axs[0, i].imshow(adaptive_index(responses[:, i, :, :]), vmin=-0.1, vmax=0.1, cmap='bwr')
        axs[1, i].imshow(adaptive_index(responses[:, i+shape[1]//2, :, :]), vmin=-0.1, vmax=0.1, cmap='bwr')
        
def adaptive_index_amacrine(layer_outs):
    responses = layer_outs['amacrine']
    shape = responses.shape
    responses = responses.reshape((shape[0], 8, 26, 26))
    shape = responses.shape
    fig, axs = plt.subplots(2, 4, figsize=(10,5))
    for i in range(shape[1]//2):
        axs[0, i].imshow(adaptive_index(responses[:, i, :, :]), vmin=-0.1, vmax=0.1, cmap='bwr')
        axs[1, i].imshow(adaptive_index(responses[:, i+shape[1]//2, :, :]), vmin=-0.1, vmax=0.1, cmap='bwr')
        
def adaptive_index_ganglion(layer_outs):
    responses = layer_outs['ganglion']
    fig, axs = plt.subplots()
    axs.imshow(adaptive_index(responses[:,0,:,:]), vmin=-0.1, vmax=0.1, cmap='bwr')
    
def normalize_filter2(sta, stimulus):
    '''Enforces filtered stimulus to have the same standard deviation
    as the stimulus.'''
    theta = stimulus.std() / ft.linear_response(sta, stimulus).std()
    return theta * sta
    
def later_early_segments(stimulus, grad, output, segments=[(1100,1500),(1500,2000),(2100,2500),(2500,3000)], filt_depth=40):
    
    result = {}
    for i, seg in enumerate(['he', 'hl', 'le', 'll']):
        start = segments[i][0]
        end = segments[i][1]
        result[seg] = (stimulus[start:end+filt_depth-1], grad[start:end].mean(0), output[start:end])
    return result

def gradient_filter(data_list, fullfield=False):
    
    filters = {}
    for key in ['he', 'hl', 'le', 'll']:
        stimuli = []
        filtered = []
        sta = 0
        for i in range(len(data_list)):
            sta += np.flip(data_list[i][key][1], axis=0)
            stimuli.append(data_list[i][key][0])
        if fullfield:
            sta = np.ones(sta.shape) * np.expand_dims(sta.mean(axis=(-1,-2)), axis=(1,2))
        for stimulus in stimuli:
            resp = linear_response_no_pad(sta, stimulus)
            filtered.append(resp)
        theta = np.array(stimuli).std() / np.array(filtered).std()
        filters[key] = theta * sta
        
    return filters

def nonlinearities(data_list, filters):
    
    result = {}
    for key in ['he', 'hl', 'le', 'll']:
        sta = filters[key]
        filtered = []
        responses = []
        for i in range(len(data_list)):
            stimulus = data_list[i][key][0]
            filtered.append(linear_response_no_pad(sta, stimulus))
            responses.append(data_list[i][key][2])
        
        filtered = np.array(filtered).flatten()
        responses = np.array(responses).flatten()
        nonlinearity = Sigmoid(peak=100.)
        nonlinearity.fit(filtered, responses, maxfev=50000)

        x = np.linspace(np.min(filtered), np.max(filtered), 10)
        nonlinear_prediction = nonlinearity.predict(x)
        
        result[key] = (x, nonlinear_prediction)
    return result

def linear_response_no_pad(filt, stim, nsamples_after=0):
    
    slices = np.fliplr(slicestim(stim, filt.shape[0] - nsamples_after, nsamples_after))
    return np.einsum('tx,x->t', flat2d(slices), filt.ravel())

def random_from_envelope(envelope, repeat=3):
    
    if repeat > 1:
        timelen = envelope.shape[0]
        stimulus = np.random.randn(timelen//repeat, *envelope.shape[1:])
        stimulus = np.repeat(stimulus, repeat, axis=0)
        leftover = np.random.randn(timelen - timelen//repeat*repeat,  *envelope.shape[1:])
        stimulus = np.concatenate((leftover, stimulus), axis=0)
    elif repeat == 1:
        stimulus = np.random.randn(*envelope.shape)
        
    assert stimulus.shape == envelope.shape
    stimulus = stimulus * envelope
    
    return stimulus