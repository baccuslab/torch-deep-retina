import torch
import os
import scipy
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data.dataloader import DataLoader
import pyret
from kinetic.evaluation import *
from kinetic.config import get_custom_cfg
from kinetic.data import *
from kinetic.utils import *
import kinetic.models as models
from torchdeepretina.utils import *
import torchdeepretina.stimuli as stim
import torchdeepretina.visualizations as viz
from torchdeepretina.retinal_phenomena import normalize_filter
from pyret.nonlinearities import Binterp, Sigmoid

def contrast_adaptation_kinetic(model, device, insp_keys, hs_mode='single', stim_type='full', I20=None,
                                c0=0.05, c1=0.35, duration=1000, delay=1000, nsamples=3000, nrepeats=10, filt_depth=40, **kwargs):
    """Step change in contrast"""

    # the contrast envelope
    if stim_type == 'full':
        envelope = stim.flash(duration, delay, nsamples, intensity=(c1 - c0))
    elif stim_type == 'one_pixel':
        envelope = stim.flash(duration, delay, nsamples, intensity=(c1 - c0)).squeeze()
    else:
        raise Exception('Invalid hs type')
    envelope += c0

    layer_outs_list  ={key:[] for key in insp_keys}
    with torch.no_grad():
        for _ in range(nrepeats):
            x = np.random.randn(*envelope.shape) * envelope + 1
            x = (x - x.mean())/x.std()
            if stim_type == 'full':
                x = torch.from_numpy(stim.concat(x, nh=filt_depth)).to(device)
            elif stim_type == 'one_pixel':
                x = torch.from_numpy(stim.rolling_window(x, filt_depth, time_axis=0)).to(device)
            else:
                raise Exception('Invalid stimulus type')
                
            hs = get_hs(model, 1, device, I20, hs_mode)
            layer_outs = inspect_rnn(model, x, hs, insp_keys)
            for key in layer_outs_list.keys():
                if key == 'kinetics':
                    kinetics_history = [h[1].detach().cpu().numpy().mean(-1) for h in layer_outs['kinetics']]
                    kinetics_history = np.concatenate(kinetics_history, axis=0)
                    layer_outs_list[key].append(kinetics_history)
                else:
                    layer_outs_list[key].append(layer_outs[key])
    for key in layer_outs_list.keys():
        if isinstance(layer_outs_list[key][0], np.ndarray):
            layer_outs[key] = np.array(layer_outs_list[key]).mean(0)
        else:
            layer_outs[key] = layer_outs_list[key]
    
    response = layer_outs['outputs']
    
    if stim_type == 'full':
        figs = viz.response1D(envelope[filt_depth:, 0, 0], response)
    elif stim_type == 'one_pixel':
        figs = viz.response1D(envelope[filt_depth:], response)
    else:
        raise Exception('Invalid hs type')
    (fig, (ax0,ax1)) = figs

    return (fig, (ax0,ax1)), layer_outs

def contrast_adaptation_LN(model, device, hs_mode='single', stim_type='full', I20=None, cells='all', 
                                c0=0.05, c1=0.35, duration=1000, delay=1000, nsamples=3000, nrepeats=10, filt_depth=40, **kwargs):
    if stim_type == 'full':
        envelope = stim.flash(duration, delay, nsamples, intensity=(c1 - c0))
    elif stim_type == 'one_pixel':
        envelope = stim.flash(duration, delay, nsamples, intensity=(c1 - c0)).squeeze()
    else:
        raise Exception('Invalid hs type')
    envelope += c0

    stimuli = []
    responses = []
    with torch.no_grad():
        for _ in range(nrepeats):
            x = np.random.randn(*envelope.shape) * envelope
            stimuli.append(x.squeeze())
            x = (x - x.mean())/x.std()
            if stim_type == 'full':
                x = torch.from_numpy(stim.concat(x, nh=filt_depth)).to(device)
            elif stim_type == 'one_pixel':
                x = torch.from_numpy(stim.rolling_window(x, filt_depth, time_axis=0)).to(device)
            else:
                raise Exception('Invalid stimulus type')

            hs = get_hs(model, 1, device, I20, hs_mode)
            layer_outs = inspect_rnn(model, x, hs)
            response = np.pad(layer_outs['outputs'], ((filt_depth, 0), (0,0)), 'constant', constant_values=(0,0))
            responses.append(response)
    
    if cells == 'all':
        cells = range(model.n_units)
    for cell in cells:
        _, _, x_he, nonlinear_he = LN_model_multi_trials(stimuli, responses, c1, cell, delay, delay + duration//2)
        _, _, x_hl, nonlinear_hl = LN_model_multi_trials(stimuli, responses, c1, cell, delay + duration//2, delay + duration)
        _, _, x_le, nonlinear_le = LN_model_multi_trials(stimuli, responses, c0, cell, delay + duration, 
                                                         (delay + duration + nsamples)//2)
        _, _, x_ll, nonlinear_ll = LN_model_multi_trials(stimuli, responses, c0, cell, (delay + duration + nsamples)//2, nsamples)
        plt.plot(x_he, nonlinear_he, 'r', label='high early')
        plt.plot(x_hl, nonlinear_hl, 'b', label='high late')
        plt.plot(x_le, nonlinear_le, 'k', label='low early')
        plt.plot(x_ll, nonlinear_ll, 'g', label='low late')
        plt.legend()
        plt.show()
        
    return
    
def LN_model_multi_trials(stimuli, responses, contrast, cell, start_idx, end_idx, filter_len=100):
    sta = 0
    stimulus = []
    resp = []
    for trial in range(len(stimuli)):
        stim_trial = stimuli[trial][start_idx:end_idx]
        resp_trial = responses[trial][start_idx:end_idx, cell]
        stimulus.append(stim_trial)
        resp.append(resp_trial)
        sta_trial, tax = pyret.filtertools.revcorr(stim_trial, resp_trial, 0, filter_len)
        sta += sta_trial
    sta = np.flip(sta, axis=0)
    tax = tax / 100
    sta -= sta.mean()
    stimulus = np.concatenate(stimulus)
    resp = np.concatenate(resp)
    normed_sta, _, _ = normalize_filter(sta, stimulus, contrast)
    
    filtered_stim = pyret.filtertools.linear_response(normed_sta, stimulus)
    nonlinearity = Binterp(80)
    nonlinearity.fit(filtered_stim[filter_len:], resp[filter_len:])

    x = np.linspace(np.min(filtered_stim), np.max(filtered_stim), 40)
    nonlinear_prediction = nonlinearity.predict(x)
    
    return tax, normed_sta, x, nonlinear_prediction

def stimulus_importance_rnn(model, X, gc_idx=None, alpha_steps=5, 
                            seq_len=8, device=torch.device('cuda:1')):
    
    requires_grad(model, False) # Model gradient unnecessary for integrated gradient
    prev_grad_state = torch.is_grad_enabled() # Save current grad calculation state
    torch.set_grad_enabled(True) # Enable grad calculations
    
    if gc_idx is None:
        gc_idx = list(range(model.n_units))
    intg_grad = torch.zeros(seq_len, *model.image_shape)
    curr_hs = get_hs(model, 1, device)
    model = model.to(device)
    model.eval()
    X = torch.FloatTensor(X)
    X.requires_grad = True
    idxs = torch.arange(len(X)).long()
    for start_idx in range(0, len(X) - seq_len):
        linspace = torch.linspace(0,1,alpha_steps)
        idx = idxs[start_idx: start_idx + seq_len]
        with torch.no_grad():
            out, next_hs = model(X[start_idx][None,:,:,:].to(device), curr_hs)
        curr_intg_grad = torch.zeros(seq_len, *model.image_shape)
        for alpha in linspace:
            x = X[idx].to(device) * alpha
            outs = inspect_rnn(model, curr_hs, x)[:,gc_idx]
            grad = torch.autograd.grad(outs.sum(), x)[0]
            grad = grad.detach().cpu().reshape(*intg_grad.shape)
            act = X[idx].detach().cpu()
            curr_intg_grad += grad*act
        intg_grad += torch.mul(curr_intg_grad, curr_intg_grad) / (len(X) - seq_len)
        curr_hs = next_hs
            
    requires_grad(model, True)
    torch.set_grad_enabled(prev_grad_state)
    
    intg_grad = intg_grad.view(seq_len * model.image_shape[0], *model.image_shape[1:])
    intg_grad = torch.mean(intg_grad, dim=(1,2))
    intg_grad = torch.sqrt(intg_grad)
    intg_grad = intg_grad.data.cpu().numpy()
    return intg_grad

def rev_sta(stim, resp, filter_len = 40):

    sta, tax = pyret.filtertools.revcorr(stim, resp, 0, filter_len)
    sta = sta / resp.sum() * 100
    sta = np.flip(sta, axis=0)
    tax = tax / 100
    sta -= sta.mean()
    normed_sta, _, _ = normalize_filter(sta, stim, stim.std())
    
    return normed_sta, tax

def fourier_sta(x, y, M):
    
    N = x.shape[0]
    contrast = x.std() / x.mean()
    x = scipy.stats.zscore(x) * contrast
    #x = x - np.mean(x)
    y = y - np.mean(y)

    offset = 100
    num_pers = np.int(np.floor((N-M)/offset))

    f = np.zeros(M)
    fft_f = np.zeros(M)
    cross_xy = fft_f
    denom = cross_xy

    for i in range(num_pers):
        x_per = x[i*offset:i*offset + M]
        y_per = y[i*offset:i*offset + M]

        auto_x = np.abs(np.fft.fft(x_per))**2
        auto_y = np.abs(np.fft.fft(y_per))**2

        cross_xy = cross_xy + np.conjugate(np.fft.fft(x_per)) * np.fft.fft(y_per)
        denom = denom + auto_x + np.mean(auto_y)*10

    fft_f = cross_xy / denom
    f = np.real(np.fft.ifft(fft_f))

    return f

def LN_model_1d(stim, resp, filter_len = 40, nonlinearity_type = 'bin'):
    
    contrast = stim.std() / stim.mean()
    stim = scipy.stats.zscore(stim) * contrast
    
    normed_sta, tax = rev_sta(stim, resp, filter_len)

    filtered_stim = pyret.filtertools.linear_response(normed_sta, stim)
    if nonlinearity_type == 'bin':
        nonlinearity = Binterp(80)
    else:
        nonlinearity = Sigmoid()
    nonlinearity.fit(filtered_stim[filter_len:], resp[filter_len:])

    x = np.linspace(np.min(filtered_stim), np.max(filtered_stim), 40)
    nonlinear_prediction = nonlinearity.predict(x)
    
    return tax, normed_sta, x, nonlinear_prediction

def contrast_adaption_nonlinear(stimulus, resp, h_start, l_start, contrast_duration=2000, e_start = 0,
                                e_duration=500, l_duration=1000, nonlinearity_type = 'bin', filter_len = 40):
    
    stim_he = stimulus[h_start+e_start:h_start+e_duration+e_start]
    resp_he = resp[h_start+e_start:h_start+e_duration+e_start]
    stim_hl = stimulus[h_start+contrast_duration-l_duration:h_start+contrast_duration]
    resp_hl = resp[h_start+contrast_duration-l_duration:h_start+contrast_duration]
    stim_le = stimulus[l_start+e_start:l_start+e_duration+e_start]
    resp_le = resp[l_start+e_start:l_start+e_duration+e_start]
    stim_ll = stimulus[l_start+contrast_duration-l_duration:l_start+contrast_duration]
    resp_ll = resp[l_start+contrast_duration-l_duration:l_start+contrast_duration]
    _, _, x_he, nonlinear_he = LN_model_1d(stim_he, resp_he)
    _, _, x_hl, nonlinear_hl = LN_model_1d(stim_hl, resp_hl)
    _, _, x_le, nonlinear_le = LN_model_1d(stim_le, resp_le)
    _, _, x_ll, nonlinear_ll = LN_model_1d(stim_ll, resp_ll)
    
    plt.plot(x_he, nonlinear_he, 'r', label='high early')
    plt.plot(x_hl, nonlinear_hl, 'b', label='high late')
    plt.plot(x_le, nonlinear_le, 'k', label='low early')
    plt.plot(x_ll, nonlinear_ll, 'g', label='low late')
    plt.legend()
    plt.show()

    return (x_he, nonlinear_he), (x_hl, nonlinear_hl), (x_le, nonlinear_le), (x_ll, nonlinear_ll)

def analyze_one_pixel(cfg_name, checkpoint_path, stimulus, device, n_units=3, channel=0, cell=0, h_start=2000, l_start=4000):
    
    cfg = get_custom_cfg(cfg_name)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model_func = getattr(models, cfg.Model.name)
    model_kwargs = dict(cfg.Model)
    model = model_func(**model_kwargs).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    data_kwargs = dict(cfg.Data)
    _, layer_outs = contrast_adaptation_kinetic(model, device, insp_keys=['kinetics'], **data_kwargs)
    
    filter_len = model.img_shape[0]

    plt.plot(np.arange(3000 - filter_len),layer_outs['kinetics'][:, 0, channel], label='R')
    plt.plot(np.arange(3000 - filter_len),layer_outs['kinetics'][:, 1, channel], label='A')
    plt.plot(np.arange(3000 - filter_len),layer_outs['kinetics'][:, 2, channel], label='I1')
    plt.plot(np.arange(3000 - filter_len),layer_outs['kinetics'][:, 3, channel], label='I2')
    plt.legend()
    plt.show()
    
    train_dataset = MyDataset(stim_sec='train', **data_kwargs)
    test_data =  DataLoader(dataset=MyDataset(stim_sec='test', stats=train_dataset.stats, **data_kwargs))
    test_pc, pred, _ = pearsonr_eval(model, test_data, n_units, device, with_responses=True)
    pred = np.pad(pred, ((filter_len, 0), (0,0)), 'constant', constant_values=(0,0))
    
    he, hl, le, ll = contrast_adaption_nonlinear(stimulus, pred[:, cell], h_start=h_start, l_start=l_start)
    
    return layer_outs

def analyze(cfg_name, checkpoint_path, checkpoint_path_one_pixel, stimulus, device,
            n_units=3, channel=0, cell=0, h_start=2000, l_start=4000):
    
    cfg = get_custom_cfg(cfg_name)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model_func = getattr(models, cfg.Model.name)
    model_kwargs = dict(cfg.Model)
    model = model_func(**model_kwargs).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    if checkpoint_path_one_pixel != '':
        checkpoint_one_pixel = torch.load(checkpoint_path_one_pixel, map_location=device)
        model.kinetics.ksi.data = checkpoint_one_pixel['model_state_dict']['kinetics.ksi']
        model.kinetics.ksr.data = checkpoint_one_pixel['model_state_dict']['kinetics.ksr']
        if model.ksr_gain:
            try:
                model.kinetics.ksr_2.data = checkpoint_one_pixel['model_state_dict']['kinetics.ksr_2']
            except:
                pass
    model.eval()
    
    filter_len = model.img_shape[0]
    
    data_kwargs = dict(cfg.Data)
    _, layer_outs = contrast_adaptation_kinetic(model, device, insp_keys=['kinetics'], **data_kwargs)

    plt.plot(np.arange(3000 - filter_len),layer_outs['kinetics'][:, 0, channel], label='R')
    plt.plot(np.arange(3000 - filter_len),layer_outs['kinetics'][:, 1, channel], label='A')
    plt.plot(np.arange(3000 - filter_len),layer_outs['kinetics'][:, 2, channel], label='I1')
    plt.plot(np.arange(3000 - filter_len),layer_outs['kinetics'][:, 3, channel], label='I2')
    plt.legend()
    plt.show()
    
    contrast_adaptation_LN(model, device, **data_kwargs)
    
    data_kwargs['stim'] = 'fullfield_whitenoise'
    train_dataset_noise = MyDataset(stim_sec='train', **data_kwargs)
    test_data_noise =  DataLoader(dataset=MyDataset(stim_sec='test', stats=train_dataset_noise.stats, **data_kwargs))
    test_pc_noise, pred_noise, _ = pearsonr_eval(model, test_data_noise, n_units, device, with_responses=True)
    pred_noise = np.pad(pred_noise, ((filter_len, 0), (0,0)), 'constant', constant_values=(0,0))

    he, hl, le, ll = contrast_adaption_nonlinear(stimulus, pred_noise[:, cell], h_start=h_start, l_start=l_start)
    
    data_kwargs['stim'] = 'naturalscene'
    train_dataset_natural = MyDataset(stim_sec='train', **data_kwargs)
    test_data_natural =  DataLoader(dataset=MyDataset(stim_sec='test', stats=train_dataset_natural.stats, **data_kwargs))
    test_pc_natural = pearsonr_eval(model, test_data_natural, n_units, device)
    
    print('white noise prediction correlation :{:.4f}'.format(test_pc_noise))
    print('natural scene prediction correlation: {:.4f}'.format(test_pc_natural))
    
    return test_pc_noise, test_pc_natural, layer_outs