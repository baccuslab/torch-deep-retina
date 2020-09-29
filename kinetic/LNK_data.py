import os
import h5py
import argparse
import numpy as np
from scipy import signal
from scipy.special import erf
from scipy.linalg import orth
import scipy.io as sio
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset
from kinetic.utils import *
import torchdeepretina.stimuli as tdrstimuli


def kinetic(rate, x_0, u, v, dt):
    x_current = x_0.copy()
    x_next = np.zeros(x_0.shape)
    X = np.zeros((u.shape[0], 4))

    for k in range(u.shape[0]):
        X[k, :] = x_current.copy()
        x_next[0] = x_current[0]*(1-dt*u[k]) + x_current[2]*dt*rate['fr']
        x_next[1] = x_current[1]*(1-dt*rate['fi']) + x_current[0]*dt*u[k]
        x_next[2] = x_current[2]*(1-dt*(rate['fr']+rate['si'])) + x_current[1]*dt*rate['fi'] + x_current[3]*dt*v[k]
        x_next[3] = x_current[3]*(1-dt*v[k]) + x_current[2]*dt*rate['si']
        x_current = x_next.copy()
    return X

def LNK(x, data, dt):
    a = data['opt_p1'][0]
    t = np.linspace(0.001, 1, 1000)
    t2 = t*2 - t**2
    m1 = np.outer(t2, np.ones(15))
    m2 = np.outer(np.ones(1000), np.arange(1,16))
    A = np.sin(m1 * m2 * np.pi)
    basis = orth(A)
    linear_weight = a[0:15] * 1000
    filters = basis.dot(linear_weight)
    if dt == 0.01:
        filters = signal.resample(filters, 100)
    after_filter = np.convolve(x, filters, 'valid') * dt
    u = (a[15]**(erf(after_filter+a[16])+1))+a[17]
    v = a[18]*((a[15]**(erf(after_filter+a[16])+1))+a[17])+a[19]
    rate = {'fi':a[21], 'fr':a[23], 'si':a[25]}
    x_0 = np.array([0., 0., 0., 100.])
    out = kinetic(rate, x_0, u, v, dt)
    out = a[26]*out
    return out

def natural_center():
    
    filepath = os.path.join('/home/TRAIN_DATA', '15-10-07', 'naturalscene' + '.h5')
    f = h5py.File(filepath, mode='r')
    stim = np.asarray(f['train']['stimulus']).astype('float32')
    x = stim[:,25,25]
    return x

def white_noise(c0=0.05, c1=0.35, tot_len=300000, duration=20, dt=0.001):
    
    n_repeat = int(duration / dt)
    envelope = np.random.random(tot_len//n_repeat) * (c1-c0) + c0
    envelope = np.repeat(envelope, n_repeat)
    x = ((np.random.randn(*envelope.shape) * envelope + 1) * 3.).astype('float32')
    return x

def LNK_stim(data, dt):
    
    stim = data['stim'][0]
    if dt == 0.001:
        return stim
    if dt == 0.01:
        return signal.resample(stim, stim.shape[0]//10)
    
def organize(stim, resp, history, val_size=30000, dt=0.01):
    stats = {}
    stats['mean'] = stim.mean()
    stats['std'] = stim.std()+1e-7
    stim = (stim-stats['mean'])/stats['std']

    stim_reshaped = tdrstimuli.rolling_window(stim, history, time_axis=0)
    stim_numpy = stim_reshaped[-resp.shape[0]:]
    stim_reshaped = torch.from_numpy(stim_reshaped[-resp.shape[0]:])
    resp = torch.from_numpy(resp)
    train_dataset = TensorDataset(stim_reshaped[:-val_size], resp[:-val_size][:,None])
    val_dataset = TensorDataset(stim_reshaped[-val_size:], resp[-val_size:][:,None])
    
    return train_dataset, val_dataset, stats

def generate(data_path, stimuli, dt):
    
    data = sio.loadmat(data_path)
    
    if stimuli == 'natural_center':
        stim = natural_center()
    if stimuli == 'white_noise':
        stim = white_noise(dt=dt)
    if stimuli == 'LNK_stim':
        stim = LNK_stim(data, dt)
    
    out = LNK(stim, data, dt)
    
    return stim, out
