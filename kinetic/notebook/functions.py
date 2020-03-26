import torch
import os
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data.dataloader import DataLoader
from kinetic.evaluation import *
from kinetic.utils import select_model
from kinetic.config import get_custom_cfg
from kinetic.data import *
from torchdeepretina.utils import *
import torchdeepretina.stimuli as stim
import torchdeepretina.visualizations as viz

def contrast_adaptation_kinetic(model, device, c0, c1, duration=50, delay=50, nsamples=140, nrepeats=10, filt_depth=40):
    """Step change in contrast"""

    # the contrast envelope
    envelope = stim.flash(duration, delay, nsamples, intensity=(c1 - c0))
    envelope += c0

    # generate a bunch of responses to random noise with the given contrast envelope
    responses = []
    with torch.no_grad():
        for _ in range(nrepeats):
            x = np.random.randn(*envelope.shape) * envelope + 1
            x = (x - x.mean())/x.std()
            x = torch.from_numpy(stim.concat(x, nh=filt_depth)).to(device)
  
            hs = get_hs(model, 1, device)
            resps = []
            for i in range(x.shape[0]):
                resp, hs = model(x[i:i+1], hs)
                resps.append(resp)
            resp = torch.cat(resps, dim=0)

            responses.append(resp.cpu().detach().numpy())

    responses = np.asarray(responses)
    figs = viz.response1D(envelope[40:, 0, 0], responses.mean(axis=0))
    (fig, (ax0,ax1)) = figs

    return (fig, (ax0,ax1)), envelope, responses

def contrast_adaptation_kinetic_reset(model, device, c0, c1, duration=50, delay=50, nsamples=140, nrepeats=10, filt_depth=40, reset=600):
    """Step change in contrast"""

    # the contrast envelope
    envelope = stim.flash(duration, delay, nsamples, intensity=(c1 - c0))
    envelope += c0

    # generate a bunch of responses to random noise with the given contrast envelope
    responses = []
    with torch.no_grad():
        for _ in range(nrepeats):
            x = np.random.randn(*envelope.shape) * envelope + 1
            x = (x - x.mean())/x.std()
            x = torch.from_numpy(stim.concat(x, nh=filt_depth)).to(device)

            resps = []
            for i in range(x.shape[0]):
                if i % reset == 0:
                    hs = get_hs(model, 1, device)
                resp, hs = model(x[i:i+1], hs)
                resps.append(resp)
                    
            resp = torch.cat(resps, dim=0)

            responses.append(resp.cpu().detach().numpy())

    responses = np.asarray(responses)
    figs = viz.response1D(envelope[40:, 0, 0], responses.mean(axis=0))
    (fig, (ax0,ax1)) = figs

    return (fig, (ax0,ax1)), envelope, responses

def contrast_adaptation_kinetic_occupancy(model, device, c0, c1, duration=50, delay=50, nsamples=140, nrepeats=10, filt_depth=40):
    """Step change in contrast"""

    # the contrast envelope
    envelope = stim.flash(duration, delay, nsamples, intensity=(c1 - c0))
    envelope += c0

    # generate a bunch of responses to random noise with the given contrast envelope
    responses = []
    Rs = []
    As = []
    I1s = []
    I2s = []
    with torch.no_grad():
        for _ in range(nrepeats):
            x = np.random.randn(*envelope.shape) * envelope + 1
            x = (x - x.mean())/x.std()
            x = torch.from_numpy(stim.concat(x, nh=filt_depth)).to(device)
            hs = get_hs(model, 1, device)
            resps = []
            R = []
            A = []
            I1 = []
            I2 = []
            for i in range(x.shape[0]):
                resp, hs = model(x[i:i+1], hs)
                resps.append(resp)
                R.append(hs[0][0,0].mean().cpu().detach().numpy())
                A.append(hs[0][0,1].mean().cpu().detach().numpy())
                I1.append(hs[0][0,2].mean().cpu().detach().numpy())
                I2.append(hs[0][0,3].mean().cpu().detach().numpy())
            resp = torch.cat(resps, dim=0)

            responses.append(resp.cpu().detach().numpy())
            Rs.append(R)
            As.append(A)
            I1s.append(I1)
            I2s.append(I2)
            
    responses = np.asarray(responses)
    Rs = np.asarray(Rs).mean(axis=0)
    As = np.asarray(As).mean(axis=0)
    I1s = np.asarray(I1s).mean(axis=0)
    I2s = np.asarray(I2s).mean(axis=0)
    figs = viz.response1D(envelope[40:, 0, 0], responses.mean(axis=0))
    (fig, (ax0,ax1)) = figs

    return Rs, As, I1s, I2s

def contrast_adaptation_kinetic_center_occupancy(model, device, c0, c1, duration=50, delay=50, nsamples=140, nrepeats=10, filt_depth=40):
    """Step change in contrast"""

    # the contrast envelope
    envelope = stim.flash(duration, delay, nsamples, intensity=(c1 - c0))
    envelope += c0

    # generate a bunch of responses to random noise with the given contrast envelope
    responses = []
    Rs = []
    As = []
    I1s = []
    I2s = []
    with torch.no_grad():
        for _ in range(nrepeats):
            x = np.random.randn(*envelope.shape) * envelope + 1
            x = (x - x.mean())/x.std()
            x = torch.from_numpy(stim.concat(x, nh=filt_depth)).to(device)
  
            hs = get_hs(model, 1, device)
            resps = []
            R = []
            A = []
            I1 = []
            I2 = []
            for i in range(x.shape[0]):
                resp, hs = model(x[i:i+1], hs)
                resps.append(resp)
                R.append(hs[0][0,0][4,648].cpu().detach().numpy())
                A.append(hs[0][0,1][4,648].cpu().detach().numpy())
                I1.append(hs[0][0,2][4,648].cpu().detach().numpy())
                I2.append(hs[0][0,3][4,648].cpu().detach().numpy())
            resp = torch.cat(resps, dim=0)

            responses.append(resp.cpu().detach().numpy())
            Rs.append(R)
            As.append(A)
            I1s.append(I1)
            I2s.append(I2)
            
    responses = np.asarray(responses)
    Rs = np.asarray(Rs).mean(axis=0)
    As = np.asarray(As).mean(axis=0)
    I1s = np.asarray(I1s).mean(axis=0)
    I2s = np.asarray(I2s).mean(axis=0)
    figs = viz.response1D(envelope[40:, 0, 0], responses.mean(axis=0))
    (fig, (ax0,ax1)) = figs

    return Rs, As, I1s, I2s

def contrast_adaptation_kinetic_occupancy_natural(model, device, c0, c1, duration=50, delay=50, nsamples=140, nrepeats=10, filt_depth=40, stimuli=None):
    """Step change in contrast"""

    
    responses = []
    Rs = []
    As = []
    I1s = []
    I2s = []
    with torch.no_grad():
        hs = get_hs(model, 1, device)
        resps = []
        targs = []
        R = []
        A = []
        I1 = []
        I2 = []
        for i in range(len(stimuli)):
            resp, hs = model(stimuli[i:i+1][0].to(device), hs)
            targs.append(stimuli[i:i+1][1])
            resps.append(resp)
            R.append(hs[0][0,0].mean().cpu().detach().numpy())
            A.append(hs[0][0,1].mean().cpu().detach().numpy())
            I1.append(hs[0][0,2].mean().cpu().detach().numpy())
            I2.append(hs[0][0,3].mean().cpu().detach().numpy())
        resp = torch.cat(resps, dim=0)

        responses = resp.cpu().detach().numpy()
        targets = torch.cat(targs, dim=0).detach().numpy()
        
    Rs = np.asarray(R)
    As = np.asarray(A)
    I1s = np.asarray(I1)
    I2s = np.asarray(I2)

    return Rs, As, I1s, I2s, responses, targets

def inspect_rnn(model, curr_hs, data, device=torch.device('cuda:1')):
    hs = []
    hs.append(curr_hs[0].detach())
    hs.append(deque([h.detach() for h in curr_hs[1]], maxlen=model.seq_len))
    data = data.to(device)
    for x in data:
        x = x[None,:,:,:]
        out, hs = model(x, hs)
    return out

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


def contrast_adaptation_kinetic_where_occupancy(model, device, c0, c1, where, duration=50, delay=50, nsamples=140, nrepeats=10, filt_depth=40):
    """Step change in contrast"""

    # the contrast envelope
    envelope = stim.flash(duration, delay, nsamples, intensity=(c1 - c0))
    envelope += c0

    # generate a bunch of responses to random noise with the given contrast envelope
    responses = []
    Rs = []
    As = []
    I1s = []
    I2s = []
    with torch.no_grad():
        for _ in range(nrepeats):
            x = np.random.randn(*envelope.shape) * envelope + 1
            x = (x - x.mean())/x.std()
            x = torch.from_numpy(stim.concat(x, nh=filt_depth)).to(device)
  
            hs = get_hs(model, 1, device)
            resps = []
            R = []
            A = []
            I1 = []
            I2 = []
            for i in range(x.shape[0]):
                resp, hs = model(x[i:i+1], hs)
                resps.append(resp)
                R.append(hs[0][0,0][where[0],where[1]].cpu().detach().numpy())
                A.append(hs[0][0,1][where[0],where[1]].cpu().detach().numpy())
                I1.append(hs[0][0,2][where[0],where[1]].cpu().detach().numpy())
                I2.append(hs[0][0,3][where[0],where[1]].cpu().detach().numpy())
            resp = torch.cat(resps, dim=0)

            responses.append(resp.cpu().detach().numpy())
            Rs.append(R)
            As.append(A)
            I1s.append(I1)
            I2s.append(I2)
            
    responses = np.asarray(responses)
    Rs = np.asarray(Rs).mean(axis=0)
    As = np.asarray(As).mean(axis=0)
    I1s = np.asarray(I1s).mean(axis=0)
    I2s = np.asarray(I2s).mean(axis=0)
    figs = viz.response1D(envelope[40:, 0, 0], responses.mean(axis=0))
    (fig, (ax0,ax1)) = figs

    return Rs, As, I1s, I2s

def contrast_adaptation_kinetic_average(model, device, c0, c1, duration=50, delay=50, nsamples=140, nrepeats=10, filt_depth=40):
    """Step change in contrast"""

    # the contrast envelope
    envelope = stim.flash(duration, delay, nsamples, intensity=(c1 - c0))
    envelope += c0

    # generate a bunch of responses to random noise with the given contrast envelope
    responses = []
    averages = []
    with torch.no_grad():
        for _ in range(nrepeats):
            x = np.random.randn(*envelope.shape) * envelope + 1
            x = (x - x.mean())/x.std()
            x = torch.from_numpy(stim.concat(x, nh=filt_depth)).to(device)
  
            hs = get_hs(model, 1, device)
            resps = []
            aves = []
            for i in range(x.shape[0]):
                resp, hs = model(x[i:i+1], hs)
                resps.append(resp)
                aves.append(resp * x[i].mean())
            resp = torch.cat(resps, dim=0)
            ave = torch.cat(aves, dim=0)

            responses.append(resp.cpu().detach().numpy())
            averages.append(ave.cpu().detach().numpy())

    responses = np.asarray(responses)
    averages = np.asarray(averages)
    figs = viz.response1D(envelope[40:, 0, 0], responses.mean(axis=0))
    (fig, (ax0,ax1)) = figs

    return averages.mean(axis=0)

def contrast_adaptation_kinetic_occupancy_onepixel(model, device, c0, c1, duration=50, delay=50, nsamples=140, nrepeats=10, filt_depth=40):
    """Step change in contrast"""

    # the contrast envelope
    envelope = stim.flash(duration, delay, nsamples, intensity=(c1 - c0)).squeeze()
    envelope += c0

    # generate a bunch of responses to random noise with the given contrast envelope
    responses = []
    Rs = []
    As = []
    I1s = []
    I2s = []
    with torch.no_grad():
        for _ in range(nrepeats):
            x = np.random.randn(*envelope.shape) * envelope + 1
            x = (x - x.mean())/x.std()
            x = torch.from_numpy(stim.rolling_window(x, filt_depth, time_axis=0)).to(device)
            hs = get_hs(model, 1, device)
            resps = []
            R = []
            A = []
            I1 = []
            I2 = []
            for i in range(x.shape[0]):
                resp, hs = model(x[i:i+1], hs)
                resps.append(resp)
                R.append(hs[0][0,0].mean().cpu().detach().numpy())
                A.append(hs[0][0,1].mean().cpu().detach().numpy())
                I1.append(hs[0][0,2].mean().cpu().detach().numpy())
                I2.append(hs[0][0,3].mean().cpu().detach().numpy())
            resp = torch.cat(resps, dim=0)

            responses.append(resp.cpu().detach().numpy())
            Rs.append(R)
            As.append(A)
            I1s.append(I1)
            I2s.append(I2)
            
    responses = np.asarray(responses)
    Rs = np.asarray(Rs).mean(axis=0)
    As = np.asarray(As).mean(axis=0)
    I1s = np.asarray(I1s).mean(axis=0)
    I2s = np.asarray(I2s).mean(axis=0)
    figs = viz.response1D(envelope[40:, 0, 0], responses.mean(axis=0))
    (fig, (ax0,ax1)) = figs

    return Rs, As, I1s, I2s

def contrast_adaptation_kinetic_occupancy_2(model, device, c0, c1, duration=50, delay=50, nsamples=140, nrepeats=10, filt_depth=40):
    """Step change in contrast"""

    # the contrast envelope
    envelope = stim.flash(duration, delay, nsamples, intensity=(c1 - c0))
    envelope += c0

    # generate a bunch of responses to random noise with the given contrast envelope
    responses = []
    Rs = []
    As = []
    I1s = []
    I2s = []
    with torch.no_grad():
        for _ in range(nrepeats):
            x = np.random.randn(*envelope.shape) * envelope + 1
            x = (x - x.mean())/x.std()
            x = torch.from_numpy(stim.concat(x, nh=filt_depth)).to(device)
            hs = get_hs_2(model, 1, device)
            resps = []
            R = []
            A = []
            I1 = []
            I2 = []
            for i in range(x.shape[0]):
                resp, hs = model(x[i:i+1], hs)
                resps.append(resp)
                R.append(hs[0][0,0].mean().cpu().detach().numpy())
                A.append(hs[0][0,1].mean().cpu().detach().numpy())
                I1.append(hs[0][0,2].mean().cpu().detach().numpy())
                I2.append(hs[0][0,3].mean().cpu().detach().numpy())
            resp = torch.cat(resps, dim=0)

            responses.append(resp.cpu().detach().numpy())
            Rs.append(R)
            As.append(A)
            I1s.append(I1)
            I2s.append(I2)
            
    responses = np.asarray(responses)
    Rs = np.asarray(Rs).mean(axis=0)
    As = np.asarray(As).mean(axis=0)
    I1s = np.asarray(I1s).mean(axis=0)
    I2s = np.asarray(I2s).mean(axis=0)
    figs = viz.response1D(envelope[40:, 0, 0], responses.mean(axis=0))
    (fig, (ax0,ax1)) = figs

    return Rs, As, I1s, I2s