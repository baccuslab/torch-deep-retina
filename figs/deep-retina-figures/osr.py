"""Helper functions for making figures."""

from itertools import repeat

import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from tqdm import tqdm
import os

import stimuli as s
from torchdeepretina.io import load_model



def sqz(mdls, X):
    """Squeeze predictions from multiple models into one array."""
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
    return r


def block(x, offset, dt=10, us_factor=50, ax=None, alpha=1.0, color='lightgrey'):
    """block plot."""

    # upsample the stimulus
    time = np.arange(x.size) * dt
    time_us = np.linspace(0, time[-1], us_factor * time.size)
    x = interp1d(time, x, kind='nearest')(time_us)
    maxval, minval = x.max(), x.min()

    ax = plt.gca() if ax is None else ax
    ax.fill_between(time_us, np.ones_like(x) * offset, x+offset, color=color, interpolate=True)
    ax.plot(time_us, x+offset, '-', color=color, alpha=alpha)

    return ax


def osr(duration=2, interval=8, nflashes=3, intensity=-1.0, noise_std=0.1):
    """Omitted stimulus response

    Parameters
    ----------
    duration : int
        Length of each flash in samples (default: 2)

    interval : int
        Number of frames between each flash (default: 10)

    nflashes : int
        Number of flashes to show before the omitted flash (default: 5)

    intensity : float
        The intensity (luminance) of each flash (default: -2.0)
    """

    # generate the stimulus
    #single_flash = s.flash(duration, interval, duration+interval, intensity=intensity)
    #omitted_flash = s.flash(duration, interval, duration+interval, intensity=0.0)
    #flash_group = list(repeat(single_flash, nflashes))
    #zero_pad = np.zeros((40-interval, 1, 1))
    #start_pad = np.zeros((interval * (nflashes-1), 1, 1))
    #X = s.concat(start_pad, zero_pad, *flash_group, omitted_flash, zero_pad, nx=50, nh=40)
    #return X
    single_flash = s.flash(duration, interval, duration+interval, intensity=intensity)
    single_flash += noise_std*np.random.randn(*single_flash.shape)
    omitted_flash = s.flash(duration, interval, duration+interval, intensity=0.0)
    omitted_flash += noise_std*np.random.randn(*omitted_flash.shape)
    flash_group = list(repeat(single_flash, nflashes))
    zero_pad = np.zeros((40-interval, 1, 1))
    start_pad = np.zeros((interval * (nflashes-1), 1, 1))
    X = s.concat(start_pad, zero_pad, *flash_group, omitted_flash, zero_pad, nx=50, nh=40)
    omitted_idx = len(start_pad) + len(zero_pad) + nflashes*len(single_flash) + interval - 40
    return X, omitted_idx

def rm_pngs(path=None, verbose=False):
    path = "." if path is None else path
    for d,_,fs in os.walk(path):
        for f in fs:
            if ".png" in f:
                temp = os.path.join(d,f)
                os.system("rm "+temp)
                if verbose:
                    print("Removing", temp)

def main():
    """Generates the OSR figure."""
    rm_pngs("imgs/",verbose=True)
    prepath = "convgc_"

    # Load natural scene models.
    kn1 = load_model('/home/grantsrb/src/torch-deep-retina/models/'+prepath+'15-10-07_naturalscene.pt')
    kn2 = load_model('/home/grantsrb/src/torch-deep-retina/models/'+prepath+'15-11-21a_naturalscene.pt')
    kn3 = load_model('/home/grantsrb/src/torch-deep-retina/models/'+prepath+'15-11-21b_naturalscene.pt')

    # Load white noise models.
    km1 = load_model('/home/grantsrb/src/torch-deep-retina/models/'+prepath+'15-10-07_whitenoise.pt')
    km2 = load_model('/home/grantsrb/src/torch-deep-retina/models/'+prepath+'15-11-21a_whitenoise.pt')
    km3 = load_model('/home/grantsrb/src/torch-deep-retina/models/'+prepath+'15-11-21b_whitenoise.pt')

    #for i in range(1,10):
    #    for d in tqdm(range(1,min(i+5,10))):
    # Generate stimulus.
    dt = 10
    # interval: delay
    # duration: duration of flash
    X,omt_idx = osr(duration=2, interval=6, nflashes=8, intensity=-2, noise_std=0)

    # Responses
    t = np.linspace(0, len(X)*dt, len(X))
    rn = sqz((kn1,kn2, kn3,), X)
    rm = sqz((km1, km2, km3,), X)

    # Plot
    plt.clf()
    plt.style.use('deepretina.mplstyle')
    fig = plt.figure(figsize=(15,8))
    s = np.s_[40:140]
    temp_t = t[s]
    wht = rm.mean(1)[s]
    nat = rn.mean(1)[s]
    nat_color = 'lightcoral'
    whit_color = '#888888'
    plt.plot(temp_t, wht, color=whit_color, label='White Noise',linewidth=4)
    #plt.ylabel("Rate (Hz)", fontsize=40)
    #plt.xlabel("Time from Omission (s)", fontsize=40)
    ax = plt.gca()
    ax.plot(temp_t, nat, color=nat_color , label='Natural Scenes',linewidth=4)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    offset = max(np.max(rn.mean(1)),np.max(rm.mean(1)))
    offset = 11
    #ax.plot(t,X[:,-1,0,0]+offset)
    #plt.title('imgs/temp_intv{}_dur{}.png'.format(i,d))
    block(X[:, -1, 0, 0], offset=offset, dt=dt, us_factor=50, ax=plt.gca(), color="black")
    zero = t[omt_idx]
    neg2 = t[omt_idx-20]
    neg4 = t[omt_idx-40]
    pos2 = t[omt_idx+20]
    ticks = [neg4, neg2, zero, pos2]
    plt.xlim([neg4-20,pos2+20])
    plt.ylim([0,offset])
    plt.yticks([0,5,10])
    #plt.locator_params(nbins=3)
    labels = [-0.4, -0.2, 0, .2]
    plt.xticks(ticks=ticks, labels=labels)
    ax.tick_params(axis='both', which='major', labelsize=45)
    l = plt.legend(fontsize=40,loc='upper right',frameon=False,
                                                handletextpad=-2.0, 
                                                handlelength=0,
                                                markerscale=0)
    colors = [whit_color, nat_color]
    for color,text in zip(colors, l.get_texts()):
        text.set_color(color)
    #ax.plot(t, X[:,-1,0,0]-(2), 'k')

    plt.tight_layout()
    plt.savefig(prepath+'osr.png')
    plt.savefig(prepath+'osr.pdf')


if __name__ == "__main__":
    main()
