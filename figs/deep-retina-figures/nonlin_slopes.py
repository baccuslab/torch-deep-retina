"""Fast contrast adaptation figure."""

import os

import stimuli as s
import deepdish as dd

import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns
from scipy.fftpack import fft, fftshift, fftfreq
from scipy.interpolate import interp1d
import pandas as pd

import pyret.filtertools as ft
import torchdeepretina as tdr
import torchdeepretina.stimuli as s
from torchdeepretina.io import load_model

from jetpack import errorplot
from tqdm import tqdm

prepath = "convgc_"
DATAFILE = prepath+'nonlin_slopes_15-10-07.h5'
DEVICE = torch.device('cuda:0')


def signfix(taus):
    """Corrects sign of a temporal kernel."""
    new_taus = []
    for tau in taus:
        if np.mean(tau[25:]) > 0:
            new_taus.append(-tau)
        else:
            new_taus.append(tau)
    return np.stack(new_taus)


def cadapt(contrasts, mdls, ncells):
    """Runs contrast adaptation analysis."""
    
    celldata = []
    for mdl, nc in zip(mdls, ncells):
        mdl.cuda()
        mdl.eval()
        for ci in range(nc):
            taus = []
            for c in tqdm(contrasts):
                X = s.concat(s.white(200, nx=50, contrast=c))
                X = torch.FloatTensor(X).cuda()
                X.requires_grad = True
                pred = mdl(X)[:,ci]
                pred.sum().backward()
                g = X.grad.data.detach().cpu().numpy()
                rf = g.mean(axis=0)
                _, tau = ft.decompose(rf)
                taus.append(tau)
            celldata.append(signfix(np.stack(taus)))
        mdl.cpu()
    return np.stack(celldata)


def center_of_mass(f, p):
    return np.sum(f * p)

def latenc(mdls, intensities):
    rates = []
    for i in tqdm(intensities):
        X = s.concat(s.flash(2, 40, 70, intensity=i))
        X = torch.FloatTensor(X).cuda()
        preds = []
        for mdl in mdls:
            mdl.cuda()
            mdl.eval()
            #pred = mdl(X).detach().cpu().numpy()[:,i:i+1]
            pred = mdl(X).detach().cpu().numpy()
            preds.append(pred)
            mdl.cpu()
        r = np.hstack(preds)
        rates.append(r)
    return np.stack(rates)


def upsample_single(tau, t, t_us):
    return interp1d(t, tau, kind='cubic')(t_us)


def upsample(taus, t, t_us):
    return np.stack([upsample_single(tau, t, t_us) for tau in taus])


def npower(x):
    """Normalized Fourier power spectrum."""
    amp = np.abs(fft(x))
    power = amp ** 2
    return power / power.max()


def get_slopes(model, contrasts):
    model.to(DEVICE)
    table = {"contrast":[], "slope":[]}
    for contr in contrasts:
        for i in range(model.n_units):
            slope = get_nonlin_slope(model, contr, i)
            table['contrast'].append(contr)
            table['slope'].append(slope)
    return table


def get_nonlin_slope(model, contr, unit_idx):
    layer_name = "sequential." + str(len(model.sequential)-1)
    tup = tdr.retinal_phenomena.filter_and_nonlinearity(model, contr, layer_name=layer_name,
                                      unit_index=unit_idx, verbose=True,
                                      nonlinearity_type='bin')
    time, temporal, x, nl, nonlinearity = tup
    slope = 0
    for i in range(1,len(time)):
        slope += (nl[0]-nl[i])/(time[0]-time[i])
    return slope/(len(time)-1)


def generate_data(min_contrast=0.5, max_contrast=2.0, num_contrasts=4):
    """Generates data for the contrast adaptation figure."""
    contrasts = np.linspace(min_contrast, max_contrast, num_contrasts)
    t = np.linspace(-400, 0, 40)
    t_us = np.linspace(-400, 0, 1000)

    path = '/home/grantsrb/src/torch-deep-retina/models/'+prepath+'15-10-07_naturalscene.pt'
    kn1 = load_model(path)
    ns = cadapt(contrasts, (kn1,), (5,))                      # Get temporal kernels
    nss = upsample(ns.reshape(20, -1), t, t_us).reshape(5, 4, 1000) # Upsample kernels
    Fns = np.stack([npower(ni) for ni in nss])                      # FFT analysis
    nat_slopes = get_slopes(kn1, contrasts)

    path = '/home/grantsrb/src/torch-deep-retina/models/'+prepath+'15-10-07_whitenoise.pt'
    km1 = load_model(path)
    wn = cadapt(contrasts, (km1,), (5,))
    wns = upsample(wn.reshape(20, -1), t, t_us).reshape(5, 4, 1000)
    Fwn = np.stack([npower(ni) for ni in wns])
    whit_slopes = get_slopes(km1, contrasts)

    slopes = {"naturalscene":nat_slopes, "whitenoise":whit_slopes}

    freqs = fftfreq(1000, 1e-3 * np.mean(np.diff(t_us)))

    data = {
        'ns': ns,
        'contrasts': contrasts,
        'nss': nss,
        't': t,
        't_us': t_us,
        'wn': wn,
        'wns': wns,
        'Fns': Fns,
        'Fwn': Fwn,
        'freqs': freqs,
        'slopes': slopes
    }
    dd.io.save(DATAFILE, data)
    return data


def main():
    if os.path.exists(DATAFILE):
        data = dd.io.load(DATAFILE)
    else:
        data = generate_data()

    nat_slopes = data['slopes']['naturalscene']
    nat_df = pd.DataFrame(nat_slopes)
    nat_df['Model Type'] = 'Natural Scene'
    whit_slopes = data['slopes']['whitenoise']
    whit_df = pd.DataFrame(whit_slopes)
    whit_df['Model Type'] = 'White Noise'
    df = nat_df.append(whit_df)

    fig = plt.figure(figsize=(10, 6))
    ax = sns.scatterplot(x="contrast",y="slope",hue="Model Type",data=df, linewidth=3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    #ax.spines['left'].set_visible(False)
    #ax.spines['bottom'].set_visible(False)

    #ax.set_xlim(0.25, 2.25)
    #ax.set_ylim(9, 16)
    ax.set_xlabel('Contrast (a.u.)', fontsize=20)
    ax.set_ylabel('Frequency (Hz)', fontsize=20)
    #ax.set_title('Center of mass of frequency response',fontsize=20)
    ax.tick_params(axis='both', which='major', labelsize=35)
    plt.locator_params(nbins=3,fontsize=20)
    plt.tight_layout()
    plt.savefig(prepath+'nonlin_slopes.png')
    plt.savefig(prepath+'nonlin_slopes.pdf')


if __name__ == "__main__":
    main()
