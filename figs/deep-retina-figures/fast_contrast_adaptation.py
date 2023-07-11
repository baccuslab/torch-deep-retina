"""Fast contrast adaptation figure."""

import os

import deepdish as dd

import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy.fftpack import fft, fftshift, fftfreq
from scipy.interpolate import interp1d

import pyret.filtertools as ft
import stimuli as s
import torchdeepretina as tdr

from jetpack import errorplot
from tqdm import tqdm

prepath = "convgc_"
DATAFILE = prepath+'fast_contrast_adaptation-15-10-07.h5'


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



def generate_data(min_contrast=0.5, max_contrast=2.0, num_contrasts=4):
    """Generates data for the contrast adaptation figure."""
    contrasts = np.linspace(min_contrast, max_contrast, num_contrasts)
    t = np.linspace(-400, 0, 40)
    t_us = np.linspace(-400, 0, 1000)

    path = '/home/grantsrb/src/torch-deep-retina/models/'+prepath+'15-10-07_naturalscene.pt'
    kn1 = tdr.io.load_model(path)
    ns = cadapt(contrasts, (kn1,), (5,))                      # Get temporal kernels
    nss = upsample(ns.reshape(20, -1), t, t_us).reshape(5, 4, 1000) # Upsample kernels
    Fns = np.stack([npower(ni) for ni in nss])                      # FFT analysis

    path = '/home/grantsrb/src/torch-deep-retina/models/'+prepath+'15-10-07_whitenoise.pt'
    km1 = tdr.io.load_model(path)
    wn = cadapt(contrasts, (km1,), (5,))
    wns = upsample(wn.reshape(20, -1), t, t_us).reshape(5, 4, 1000)
    Fwn = np.stack([npower(ni) for ni in wns])

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
    }
    dd.io.save(DATAFILE, data)
    return data


def main():
    plt.style.use('deepretina.mplstyle')
    if os.path.exists(DATAFILE):
        data = dd.io.load(DATAFILE)
    else:
        data = generate_data()

    Fw = data['Fwn']
    Fn = data['Fns']
    fr = fftfreq(1000, 4e-4)

    ns_cm = np.zeros((5, 4))
    wn_cm = np.zeros((5, 4))
    for ci in range(5):
        for j in range(4):
            ns_cm[ci, j] = center_of_mass(fr[:7], Fn[ci, j, :7])
            wn_cm[ci, j] = center_of_mass(fr[:7], Fw[ci, j, :7])

    c = data['contrasts']

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)

    errorplot(c, ns_cm.mean(axis=0), ns_cm.std(axis=0) / np.sqrt(5), method='line', fmt='o', color='lightcoral', ax=ax)
    errorplot(c, wn_cm.mean(axis=0), wn_cm.std(axis=0) / np.sqrt(5), method='line', fmt='o', color='gray', ax=ax)

    zs = np.linspace(0.5, 2.0, 1e3)
    Pn = np.polyfit(c, ns_cm.mean(axis=0), 1)
    Pw = np.polyfit(c, wn_cm.mean(axis=0), 1)

    ax.plot(zs, np.polyval(Pn, zs), '--', color='lightcoral')
    ax.plot(zs, np.polyval(Pw, zs), '--', color='gray')

    ax.set_xlim(0.25, 2.25)
    ax.set_ylim(9, 16)
    plt.xticks([])
    plt.yticks(ticks=[9,12,15])
    #ax.set_xlabel('Contrast (a.u.)', fontsize=40)
    #ax.set_ylabel('Frequency (Hz)', fontsize=40)
    #ax.set_title('Center of mass of frequency response',fontsize=20)
    ax.tick_params(axis='both', which='major', labelsize=35)
    #plt.locator_params(nbins=3)
    plt.tight_layout()
    plt.savefig(prepath+'fastcontrastadaptationfig.png')
    plt.savefig(prepath+'fastcontrastadaptationfig.pdf')


if __name__ == "__main__":
    main()
