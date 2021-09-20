import torch
import os
import scipy
import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
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
from pyret.filtertools import decompose

def contrast_adaptation(model, device, insp_keys, scale=4.46, fpf=3, c0=0.05, c1=0.35, duration=1000, nrepeats=10, channel=4,
                        cells='all', filt_depth=40, load_stimuli=None, hs_mode='single', stim_type='full', I20=None, **kwargs):
    """Step change in contrast"""

    # the contrast envelope
    if stim_type == 'full':
        envelope = stim.flash(duration, duration, 3 * duration, intensity=(c1 - c0))
    elif stim_type == 'one_pixel':
        envelope = stim.flash(duration, duration, 3 * duration, intensity=(c1 - c0)).squeeze()
    else:
        raise Exception('Invalid hs type')
    envelope += c0
    
    stimuli = []
    responses = []
    
    layer_outs_list = {key:[] for key in insp_keys}
    layer_outs_list['outputs'] = []
    with torch.no_grad():
        for trial in range(nrepeats):
            if load_stimuli == None:
                x = random_from_envelope(envelope, repeat=fpf)
            else:
                x = load_stimuli[trial] - 1
                x = np.expand_dims(x, axis=(-1,-2))
            stimuli.append(x.squeeze())
            x = scale * x
            if stim_type == 'full':
                x = torch.from_numpy(stim.concat(x, nh=filt_depth)).to(device)
            elif stim_type == 'one_pixel':
                x = torch.from_numpy(stim.rolling_window(x, filt_depth, time_axis=0)).to(device)
            else:
                raise Exception('Invalid stimulus type')
                
            hs = get_hs(model, 1, device, I20, hs_mode)
            layer_outs = inspect_rnn(model, x, hs, insp_keys)
            response = np.pad(layer_outs['outputs'], ((filt_depth, 0), (0,0)), 'constant', constant_values=(0,0))
            responses.append(response)
            
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
    
    stimuli = [(stim+1) for stim in stimuli]
    if cells == 'all':
        cells = range(model.n_units)
    for cell in cells:
        ln_he = LN_model(stimuli, responses, c1, cell, duration, duration + 500, sta_type='revcor')
        ln_hl = LN_model(stimuli, responses, c1, cell, 2 * duration - 600, 2 * duration, sta_type='revcor')
        ln_le = LN_model(stimuli, responses, c0, cell, 2 * duration, 2 * duration + 500, sta_type='revcor')
        ln_ll = LN_model(stimuli, responses, c0, cell, 3 * duration - 600, 3 * duration, sta_type='revcor')
        
        LN_plot(ln_he, ln_hl, ln_le, ln_ll, save='LN_'+str(cell))
        
    responses_plot(stimuli, responses, layer_outs, channel, save='response')

    return layer_outs, stimuli, responses

def LN_plot(ln_he, ln_hl, ln_le, ln_ll, save=None, filt_len=50, dpi=300):
    
    sta_he, x_he, nonlinear_he = ln_he
    sta_hl, x_hl, nonlinear_hl = ln_hl
    sta_le, x_le, nonlinear_le = ln_le
    sta_ll, x_ll, nonlinear_ll = ln_ll
    
    fig, axes = plt.subplots(1, 2, figsize=(4.9, 2.1),  constrained_layout=True)
    
    axes[0].plot(np.linspace(0, filt_len/100., filt_len), sta_he[:filt_len], 'r', label=r'$H_{early}$')
    axes[0].plot(np.linspace(0, filt_len/100., filt_len), sta_hl[:filt_len], 'b', label=r'$H_{late}$')
    axes[0].plot(np.linspace(0, filt_len/100., filt_len), sta_le[:filt_len], 'k', label=r'$L_{early}$')
    axes[0].plot(np.linspace(0, filt_len/100., filt_len), sta_ll[:filt_len], 'g', label=r'$L_{late}$')
    axes[0].legend(fontsize=13, loc='best', ncol=2, labelcolor='linecolor', frameon=False, handlelength=0, columnspacing=0.2, labelspacing=0.2, handletextpad=0)
    '''
    axes[0].text(0.18, -13, r'$H_{early}$', color='r', fontsize=13)
    axes[0].text(0.18, -20, r'$H_{late}$', color='b', fontsize=13)
    axes[0].text(0.35, -13, r'$L_{early}$', color='k', fontsize=13)
    axes[0].text(0.35, -20, r'$L_{late}$', color='g', fontsize=13)
    '''
    axes[0].set_xlabel('Delay (s)', fontsize=13)
    axes[0].set_ylabel(r'Filter ($s^{-1}$)', fontsize=13)
    axes[0].set_title('Filter', fontsize=15)
    axes[0].spines['right'].set_visible(False)
    axes[0].spines['top'].set_visible(False)
    axes[0].tick_params(axis='both', which='major', labelsize=10)
        
        
    axes[1].plot(x_he, nonlinear_he, 'r', label=r'$H_{early}$')
    axes[1].plot(x_hl, nonlinear_hl, 'b', label=r'$H_{late}$')
    axes[1].plot(x_le, nonlinear_le, 'k', label=r'$L_{early}$')
    axes[1].plot(x_ll, nonlinear_ll, 'g', label=r'$L_{late}$')
    axes[1].set_xlim((-1, 1))
    axes[1].set_xlabel('Input', fontsize=13)
    axes[1].set_ylabel('Output (Hz)', fontsize=13)
    axes[1].set_title('Nonlinearity', fontsize=15)
    axes[1].spines['right'].set_visible(False)
    axes[1].spines['top'].set_visible(False)
    axes[1].tick_params(axis='both', which='major', labelsize=10)
    
    if save != None:
        plt.savefig('/home/xhding/workspaces/torch-deep-retina/kinetic/notebook/figures/'+save+'.png', dpi=dpi, bbox_inches = "tight")
    plt.show()
    
    return

def responses_plot(stimuli, responses, layer_outs, channel, cell=2, save=None, dpi=300, duration=1000, start=800, filt_len=40):
    
    fig, axes = plt.subplots(3,1, figsize=(5, 6), sharex=True, constrained_layout=True)
    for trial in range(len(stimuli)):
        stimuli[trial][stimuli[trial]>2.] = 2.
        stimuli[trial][stimuli[trial]<0.] = 0.
    axes[0].plot(np.linspace((start-duration)/100, 2*duration/100, (3*duration-start)), 127.5*stimuli[0][start:], color='gray', alpha=0.8)
    axes[0].axvline(x=0, color='black', ls='--')
    axes[0].axvline(x=duration//100, color='black', ls='--')
    axes[0].set_ylabel('Intensity', fontsize=13)
    axes[0].set_title('Stimulus', fontsize=15)
    axes[0].set_ylim((-10, 310))
    axes[0].add_patch(patches.Rectangle((1, 275), 4, 20, color='red'))
    axes[0].add_patch(patches.Rectangle((duration//100-5, 275), 5, 20, color='blue'))
    axes[0].add_patch(patches.Rectangle((1+duration//100, 275), 4, 20, color='black'))
    axes[0].add_patch(patches.Rectangle((2*duration//100-5, 275), 5, 20, color='green'))
    axes[0].spines['right'].set_visible(False)
    axes[0].spines['top'].set_visible(False)
    axes[0].tick_params(axis='both', which='major', labelsize=10)

    axes[1].plot(np.linspace((start-duration)/100, 2*duration/100, (3*duration-start)), np.array(responses).mean(0)[start:, cell], color='r', alpha=0.7)
    axes[1].axvline(x=0, color='black', ls='--')
    axes[1].axvline(x=duration//100, color='black', ls='--')
    axes[1].set_ylabel('Response (Hz)', fontsize=13)
    axes[1].set_title('Response', fontsize=15)
    axes[1].spines['right'].set_visible(False)
    axes[1].spines['top'].set_visible(False)
    axes[1].tick_params(axis='both', which='major', labelsize=10)

    axes[2].plot(np.linspace((start-duration)/100, 2*duration/100, (3*duration-start)), layer_outs['kinetics'][start-filt_len:, 0, channel], label=r'$R$', alpha=0.8)
    axes[2].plot(np.linspace((start-duration)/100, 2*duration/100, (3*duration-start)), layer_outs['kinetics'][start-filt_len:, 1, channel], label=r'$A$', alpha=0.8)
    axes[2].plot(np.linspace((start-duration)/100, 2*duration/100, (3*duration-start)), layer_outs['kinetics'][start-filt_len:, 2, channel], label=r'$I_1$', alpha=0.8)
    axes[2].plot(np.linspace((start-duration)/100, 2*duration/100, (3*duration-start)), layer_outs['kinetics'][start-filt_len:, 3, channel], label=r'$I_2$', alpha=0.8)
    axes[2].legend(fontsize=13, loc='right', ncol=2, labelcolor='linecolor', frameon=False, handlelength=0, columnspacing=0.5, labelspacing=0.5, handletextpad=0)
    axes[2].axvline(x=0, color='black', ls='--')
    axes[2].axvline(x=duration//100, color='black', ls='--')
    axes[2].set_xlabel('Time (s)', fontsize=13)
    axes[2].set_ylabel('Occupancy', fontsize=13)
    axes[2].set_title('Kinetic States', fontsize=15)
    axes[2].spines['right'].set_visible(False)
    axes[2].spines['top'].set_visible(False)
    axes[2].tick_params(axis='both', which='major', labelsize=10)

    if save != None:
        plt.savefig('/home/xhding/workspaces/torch-deep-retina/kinetic/notebook/figures/'+save+'.png', dpi=dpi, bbox_inches = "tight")
    plt.show()
    
    return

def contrast_adaptation_statistics(model, device, hs_mode='single', stim_type='full', I20=None, cells='all', scale=4.46, fpf=3,
                                   c0=0.05, c1=0.35, nsamples=2000, nrepeats=10, filt_depth=40, load_stimuli=None, **kwargs):
    
    envelope = np.ones((nsamples, 1, 1)) * c0
    envelope[nsamples//2:] = c1
    if stim_type == 'one_pixel':
        envelope = envelope.squeeze()

    stimuli = []
    responses = []
    with torch.no_grad():
        for trial in range(nrepeats):
            if load_stimuli == None:
                x = random_from_envelope(envelope, repeat=fpf)
            else:
                x = load_stimuli[trial] - 1
                x = np.expand_dims(x, axis=(-1,-2))
            stimuli.append(x.squeeze())
            x = scale * x
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
    
    stimuli = [(stim+1) for stim in stimuli]
    if cells == 'all':
        cells = range(model.n_units)
    gains = {'he':[], 'hl':[]}
    freqs = []
    for cell in cells:
        gain_he, _ = LN_statistics(stimuli, responses, c1, cell, nsamples//2, nsamples//2 + 500)
        gain_hl, mean_freq = LN_statistics(stimuli, responses, c1, cell, nsamples - 600, nsamples)
        gains['he'].append(gain_he)
        gains['hl'].append(gain_hl)
        freqs.append(mean_freq)
        
    return gains, freqs

def LN_statistics_plot(data, contrasts_l, contrasts_nl, save=None, dpi=300):
    
    fig, axe = plt.subplots(2,1, figsize=(3.5,6), constrained_layout=True)

    axe[0].errorbar(contrasts_l, [np.mean(each_data[2]) for each_data in data if each_data[0] in contrasts_l], fmt='-o', 
                    yerr=[sem(each_data[2]) for each_data in data if each_data[0] in contrasts_l], color='darkviolet', alpha=0.7)
    axe[0].set_ylabel('Mean Frequency (Hz)', fontsize=13)
    axe[0].set_xticks(contrasts_l, minor=True)
    axe[0].set_xticks(contrasts_l[::2])
    axe[0].spines['right'].set_visible(False)
    axe[0].spines['top'].set_visible(False)
    axe[0].tick_params(axis='both', which='major', labelsize=10)
    axe[0].yaxis.set_major_locator(plt.MaxNLocator(4))
    
    axe[1].errorbar(contrasts_nl, [np.mean(each_data[1]['he']) for each_data in data if each_data[0] in contrasts_nl], fmt='-o', 
                    yerr=[sem(each_data[1]['he']) for each_data in data if each_data[0] in contrasts_nl], color='red', alpha=0.7, label=r'$H_{early}$')
    axe[1].errorbar(contrasts_nl, [np.mean(each_data[1]['hl']) for each_data in data if each_data[0] in contrasts_nl], fmt='-o', 
                    yerr=[sem(each_data[1]['hl']) for each_data in data if each_data[0] in contrasts_nl], color='blue', alpha=0.7, label=r'$H_{late}$')
    axe[1].legend(fontsize=13, frameon=False)
    axe[1].set_ylabel('Averaged Gain (Hz / Filtered input)', fontsize=13)

    axe[1].set_xlabel('Contrast', fontsize=13)
    axe[1].set_xticks(contrasts_nl, minor=True)
    axe[1].set_xticks(contrasts_nl[::2])
    axe[1].spines['right'].set_visible(False)
    axe[1].spines['top'].set_visible(False)
    axe[1].tick_params(axis='both', which='major', labelsize=10)
    axe[1].yaxis.set_major_locator(plt.MaxNLocator(4))

    if save != None:
        plt.savefig('/home/xhding/workspaces/torch-deep-retina/kinetic/notebook/figures/'+save+'.png', dpi=dpi, bbox_inches = "tight")
        
    plt.show()
    
    return

def LN_model(stimuli, responses, contrast, cell, start_idx, end_idx, filter_len=100, sta_type='revcor', offset=10):
    sta = 0
    stimulus = []
    resp = []
    for trial in range(len(stimuli)):
        stim_trial = stimuli[trial][start_idx:end_idx]
        resp_trial = responses[trial][start_idx:end_idx, cell]
        stimulus.append(stim_trial)
        resp.append(resp_trial)
        if sta_type == 'revcor':
            sta_trial, _ = pyret.filtertools.revcorr(stim_trial, resp_trial, 0, filter_len)
            sta_trial = np.flip(sta_trial, axis=0)
        elif sta_type == 'fourier':
            sta_trial = fourier_sta(stim_trial, resp_trial, filter_len, offset)
        sta += sta_trial
    sta -= sta.mean()
    stimulus = np.concatenate(stimulus)
    resp = np.concatenate(resp)
    normed_sta = normalize_filter2(sta, stimulus)
    
    filtered_stim = pyret.filtertools.linear_response(normed_sta, stimulus)
    #nonlinearity = Binterp(10)
    nonlinearity = Sigmoid(peak=100.)
    nonlinearity.fit(filtered_stim[filter_len:], resp[filter_len:], maxfev=50000)

    x = np.linspace(np.min(filtered_stim), np.max(filtered_stim), 50)
    nonlinear_prediction = nonlinearity.predict(x)
    
    normed_sta = normed_sta / 0.01
    
    return normed_sta, x, nonlinear_prediction

def LN_statistics(stimuli, responses, contrast, cell, start_idx, end_idx, filter_len=100):
    sta = 0
    stimulus = []
    resp = []
    for trial in range(len(stimuli)):
        stim_trial = stimuli[trial][start_idx:end_idx]
        resp_trial = responses[trial][start_idx:end_idx, cell]
        stimulus.append(stim_trial)
        resp.append(resp_trial)
        sta_trial, _ = pyret.filtertools.revcorr(stim_trial, resp_trial, 0, filter_len)
        sta_trial = np.flip(sta_trial, axis=0)
        sta += sta_trial
    sta -= sta.mean()
    stimulus = np.concatenate(stimulus)
    resp = np.concatenate(resp)
    normed_sta = normalize_filter2(sta, stimulus)
    
    filtered_stim = pyret.filtertools.linear_response(normed_sta, stimulus)
    normed_sta = normed_sta / 0.01
    gain = slope_statistic(filtered_stim[filter_len:], resp[filter_len:])
    
    amps = abs(np.fft.rfft(normed_sta))
    fs = np.fft.rfftfreq(len(normed_sta), 0.01)
    mean_freq = sum([fs * amps for fs,amps in zip(fs, amps)])/sum(amps)
    
    return gain, mean_freq

def LN_model_fourier(stimuli, responses, contrast, cell, start_idx, end_idx, filter_len=100, offset=10):
    stimulus = []
    resp = []
    N = end_idx - start_idx
    
    num_pers = np.int(np.floor((N - filter_len)/offset))
    cross_xy = np.zeros(filter_len)
    denom = np.zeros(filter_len)
    
    for trial in range(len(stimuli)):
        stim_trial = stimuli[trial][start_idx:end_idx]
        resp_trial = responses[trial][start_idx:end_idx, cell]
        stimulus.append(stim_trial)
        resp.append(resp_trial)
        for i in range(num_pers):
            x_per = stim_trial[i*offset:i*offset + filter_len]
            y_per = resp_trial[i*offset:i*offset + filter_len]
            auto_x = np.abs(np.fft.fft(x_per))**2
            auto_y = np.abs(np.fft.fft(y_per))**2
            cross_xy = cross_xy + np.conjugate(np.fft.fft(x_per)) * np.fft.fft(y_per)
            denom = denom + auto_x + np.mean(auto_y)*10
    fft_f = cross_xy / denom
    sta = np.real(np.fft.ifft(fft_f))
    sta -= sta.mean()
    stimulus = np.concatenate(stimulus)
    resp = np.concatenate(resp)
    normed_sta, _, _ = normalize_filter(sta, stimulus, contrast)
    
    filtered_stim = pyret.filtertools.linear_response(normed_sta, stimulus)
    nonlinearity = Binterp(10)
    nonlinearity.fit(filtered_stim[filter_len:], resp[filter_len:])

    x = np.linspace(np.min(filtered_stim), np.max(filtered_stim), 10)
    nonlinear_prediction = nonlinearity.predict(x)
    
    return normed_sta, x, nonlinear_prediction

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

def rev_sta(stim, resp, filter_len = 100):

    sta, _ = pyret.filtertools.revcorr(stim, resp, 0, filter_len)
    sta = sta / resp.sum() * 100
    sta = np.flip(sta, axis=0)
    sta -= sta.mean()
    normed_sta, _, _ = normalize_filter(sta, stim, stim.std())
    
    return normed_sta

def fourier_sta(x, y, filter_len=100, offset=10):
    
    N = x.shape[0]
    
    num_pers = np.int(np.floor((N - filter_len)/offset))

    cross_xy = np.zeros(filter_len)
    denom = np.zeros(filter_len)

    for i in range(num_pers):
        x_per = x[i*offset:i*offset + filter_len]
        y_per = y[i*offset:i*offset + filter_len]

        auto_x = np.abs(np.fft.fft(x_per))**2
        auto_y = np.abs(np.fft.fft(y_per))**2

        cross_xy = cross_xy + np.conjugate(np.fft.fft(x_per)) * np.fft.fft(y_per)
        denom = denom + auto_x + np.mean(auto_y)*10
        #denom = denom + auto_x

    fft_f = cross_xy / denom
    f = np.real(np.fft.ifft(fft_f))
    
    f -= f.mean()
    normed_f, _, _ = normalize_filter(f, x, x.std())

    return normed_f

def performance(model, device, cfg):
    
    n_units = cfg.Model.n_units
    data_kwargs = dict(cfg.Data)

    data_kwargs['stim'] = 'fullfield_whitenoise'
    cfg.Data.start_idx = 4000

    train_dataset_noise = MyDataset(stim_sec='train', **data_kwargs)
    test_data_noise =  DataLoader(dataset=MyDataset(stim_sec='test', stats=train_dataset_noise.stats, **data_kwargs))
    test_pc_noise, error_noise = pearsonr_eval(model, test_data_noise, n_units, device, I20=cfg.Data.I20, start_idx=cfg.Data.start_idx, hs_mode=cfg.Data.hs_mode)

    test_data_noise =  DataLoader(dataset=MyDataset(stim_sec='test', stats=train_dataset_noise.stats, **data_kwargs), shuffle=True)
    test_pc_noise_shuffle, error_noise_shuffle = pearsonr_eval(model, test_data_noise, n_units, device, I20=cfg.Data.I20, start_idx=cfg.Data.start_idx, hs_mode=cfg.Data.hs_mode)

    cfg.Data.start_idx = 0
    data_kwargs['stim'] = 'naturalscene'
    train_dataset_natural = MyDataset(stim_sec='train', **data_kwargs)
    test_data_natural =  DataLoader(dataset=MyDataset(stim_sec='test', stats=train_dataset_natural.stats, **data_kwargs))
    test_pc_natural, error_natural = pearsonr_eval(model, test_data_natural, n_units, device,
                                    I20=cfg.Data.I20, start_idx=cfg.Data.start_idx, hs_mode=cfg.Data.hs_mode)

    test_data_natural =  DataLoader(dataset=MyDataset(stim_sec='test', stats=train_dataset_natural.stats, **data_kwargs), shuffle=True)
    test_pc_natural_shuffle, error_natural_shuffle = pearsonr_eval(model, test_data_natural, n_units, device,
                                    I20=cfg.Data.I20, start_idx=cfg.Data.start_idx, hs_mode=cfg.Data.hs_mode)

    return (test_pc_noise, error_noise), (test_pc_noise_shuffle, error_noise_shuffle), (test_pc_natural, error_natural), (test_pc_natural_shuffle, error_natural_shuffle)

def performance_plot(noise, noise_shuffle, natural, natural_shuffle, corr_natural, corr_noise, save=None, dpi=300):
    
    test_pc_noise, error_noise = noise
    test_pc_noise_shuffle, error_noise_shuffle = noise_shuffle
    test_pc_natural, error_natural = natural
    test_pc_natural_shuffle, error_natural_shuffle = natural_shuffle
    
    fig, axes = plt.subplots(1,2, figsize=(5,5))

    rects1 = axes[0].bar(['full', 'shuffled'], [test_pc_natural, test_pc_natural_shuffle], capsize=5,
                         yerr=[error_natural, error_natural_shuffle], color=['blue', 'green'], alpha=0.7)

    axes[0].axhline(y=corr_natural.mean(), linestyle='--', color='black')
    axes[0].axhspan(ymin=corr_natural.mean()-sem(corr_natural), ymax=corr_natural.mean()+sem(corr_natural), color='gray', alpha=0.5)

    axes[0].text(0, 0.82, 'retinal reliability')
    axes[0].set_ylabel('Pearson Correlation Coefficient')
    axes[0].set_ylim([0, 1])
    axes[0].set_title('natural scene')

    rects1 = axes[1].bar(['full', 'shuffled'], [test_pc_noise, test_pc_noise_shuffle], capsize=5,
                         yerr=[error_noise, error_noise_shuffle], color=['blue', 'green'], alpha=0.7)
    axes[1].axhline(y=corr_noise.mean(), linestyle='--', color='black')
    axes[1].axhspan(ymin=corr_noise.mean()-sem(corr_natural), ymax=corr_noise.mean()+sem(corr_natural), color='gray', alpha=0.5)
    axes[1].set_ylim([0, 1])
    axes[1].set_title('white noise')

    if save != None:
        plt.savefig('/home/xhding/workspaces/torch-deep-retina/kinetic/notebook/figures/'+save+'.png', dpi=dpi, bbox_inches = "tight")
    plt.show()
    
    return

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
    
    contrast_adaptation_LN(model, device, **data_kwargs)
    
    train_dataset = MyDataset(stim_sec='train', **data_kwargs)
    test_data =  DataLoader(dataset=MyDataset(stim_sec='test', stats=train_dataset.stats, **data_kwargs))
    test_pc, pred, _ = pearsonr_eval(model, test_data, n_units, device, with_responses=True)
    pred = np.pad(pred, ((filter_len, 0), (0,0)), 'constant', constant_values=(0,0))
    
    he, hl, le, ll = contrast_adaption_nonlinear(stimulus, pred[:, cell], h_start=h_start, l_start=l_start)
    
    print('white noise prediction correlation :{:.4f}'.format(test_pc_noise))
    
    return layer_outs

def analyze(cfg_name, checkpoint_path, checkpoint_path_one_pixel, device, c0=0.05, c1=0.35, nrepeats=10):
    
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
        try:
            if model.ksr_gain:
                model.kinetics.ksr_2.data = checkpoint_one_pixel['model_state_dict']['kinetics.ksr_2']
        except:
            pass
    model.eval()
    
    data_kwargs = dict(cfg.Data)
    _, _, _ = contrast_adaptation(model, device, insp_keys=['kinetics'], nrepeats=nrepeats, c0=c0, c1=c1, **data_kwargs)
    
    data = []
    contrasts = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    for c1 in contrasts:
        gains, freqs = contrast_adaptation_statistics(model, device, c1=c1, nrepeats=nrepeats)
        data.append((c1, gains, freqs))
    contrasts_l = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35]
    contrasts_nl = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    LN_statistics_plot(data, contrasts_l, contrasts_nl, save='population')
    
    corr_natural = np.array([0.7118, 0.8057, 0.7835, 0.7989])
    corr_noise = np.array([0.7134, 0.6254, 0.7281, 0.7501])
    noise, noise_shuffle, natural, natural_shuffle = performance(model, device, cfg)
    performance_plot(noise, noise_shuffle, natural, natural_shuffle, corr_natural, corr_noise, save='pearson')
    
    return

def contrast_adaptation_fullfield_multi(model, device):
    
    with h5py.File('/home/xhding/tem_stim/21-01-262/fullfield.h5', 'r') as f:
        stimulus =  np.asarray(f['train']['stimulus'][:, 25, 25]).astype('float32')
    
    fullfield_dataset = MyDataset('validation', (40,50,50), '/home/xhding/tem_stim', '21-01-262', 'fullfield', 0, cells=[1,2,7,10])
    fullfield_data =  DataLoader(fullfield_dataset)
    
    pearson, val_pred, val_targ = pearsonr_eval(model, fullfield_data, 4, device, with_responses=True)
    
    stimuli = stimulus[4006:40060].reshape((9,-1))
    responses = val_pred[3966:40020, :].reshape((9, -1, 4))
    
    
    fig,ax = plt.subplots(5,1,figsize=(20,10))
    ax[0].plot(stimulus[:])
    for i in range(4):
        ax[i+1].plot(responses.mean(0)[:, i])
    plt.show()
    
    for cell in [0,1,2,3]:
        _, x_he, nonlinear_he = LN_model(stimuli, responses, 0.35, cell, 0, 500)
        _, x_hl, nonlinear_hl = LN_model(stimuli, responses, 0.35, cell, 1400, 2000)
        _, x_le, nonlinear_le = LN_model(stimuli, responses, 0.05, cell, 2000, 2500)
        _, x_ll, nonlinear_ll = LN_model(stimuli, responses, 0.05, cell, 3400, 4000)
        plt.plot(x_he, nonlinear_he, 'r', label='high early')
        plt.plot(x_hl, nonlinear_hl, 'b', label='high late')
        plt.plot(x_le, nonlinear_le, 'k', label='low early')
        plt.plot(x_ll, nonlinear_ll, 'g', label='low late')
        plt.legend()
        plt.show()
    
    return pearson, stimuli, responses

def gradient_LN(model, device, cell = 0, c0 = 0.15, c1 = 0.35, scale=2.95, shift=0.88, stim_type='checkerboard',
                filt_depth = 40, nrepeats = 10, fpf = 3, I20 = None, hs_mode = 'single'):
    if stim_type == 'checkerboard':
        envelope = np.ones((3040, 50, 50))
        envelope[:1040] = c0
        envelope[1040:2040] = c1
        envelope[2040:] = c0
    elif stim_type == 'fullfield':
        envelope = np.ones((3040, 1, 1))
        envelope[:1040] = c0
        envelope[1040:2040] = c1
        envelope[2040:] = c0

    data_list = []
    for _ in range(nrepeats):
        x = random_from_envelope(envelope, repeat=fpf)
        if stim_type == 'fullfield':
            x = x * np.ones((envelope.shape[0], 50, 50))
        stimulus = x
        x = scale * x + shift

        x = torch.from_numpy(stim.rolling_window(x, filt_depth, time_axis=0)).float().to(device)

        hs = get_hs(model, 1, device, I20, hs_mode)
        grad, output = inspect_grad_rnn(model, x, hs, cell_idx=cell)
        segment_dict = later_early_segments(stimulus, grad, output)
        data_list.append(segment_dict)
    
    if stim_type == 'fullfield':
        fullfield = True
    elif stim_type == 'checkerboard':
        fullfield = False
    filters = gradient_filter(data_list, fullfield)
    nonlinear = nonlinearities(data_list, filters)
    
    for key in nonlinear.keys():
        x, nonlinear_prediction = nonlinear[key]
        plt.plot(x, nonlinear_prediction, label=key)
    plt.legend()
    plt.show()
    
    for key in filters.keys():
        sta = filters[key]
        _, t_filter = decompose(sta)
        plt.plot(t_filter, label=key)
    plt.legend()
    plt.show()
    
    return filters, nonlinear