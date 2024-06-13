"""F2 Response plot."""

import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fftfreq, fft
from tqdm import tqdm
from jetplot import errorplot
import torchdeepretina as tdr
import os
import torch

#import drone
import stimuli as s

def sqz_predict(mdls):
    """Squeeze predictions from multiple models into one array."""
    def predict(X):
        X = torch.FloatTensor(X).cuda()
        return np.hstack([mdl(X).detach().cpu().numpy() for mdl in mdls])
    return predict


def freq_doubling(predict, ncells, phases, widths, halfperiod=25, nsamples=140):
    F1 = np.zeros((ncells, phases.size, widths.size))
    F2 = np.zeros_like(F1)

    period = (2 * halfperiod) * 0.01
    base_freq = 1 / period
    freqs = fftfreq(nsamples - 40, 0.01)
    i1 = np.where(freqs == base_freq)[0][0]
    i2 = np.where(freqs == base_freq * 2)[0][0]
    print(f'Base frequency: {base_freq}Hz\n\tF1[{i1}]: {freqs[i1]}Hz\n\tF2[{i2}]: {freqs[i2]}Hz\n')

    for p, phase in enumerate(phases):
        for w, width in tqdm(list(enumerate(widths))):
            X = s.reversing_grating(nsamples, halfperiod, width, phase, 'sin')
            stim = s.concat(X)
            r = predict(stim)
            for ci in range(r.shape[1]):
                amp = np.abs(fft(r[:, ci]))
                F1[ci, p, w] = amp[i1]
                F2[ci, p, w] = amp[i2]
    return F1, F2


def main():
    """Generate figure."""

    # Load natural scene models.
    #kn1 = drone.load_model('15-10-07', 'naturalscene')
    #kn2 = drone.load_model('15-11-21a', 'naturalscene')
    #kn3 = drone.load_model('15-11-21b', 'naturalscene')

    prepath = os.path.expanduser("~/src/torch-deep-retina/models/")
    kn1 = tdr.io.load_model(os.path.join(prepath,'convgc_{}_{}.pt'.format('15-10-07', 'naturalscene'))).cuda()
    kn2 = tdr.io.load_model(os.path.join(prepath,'convgc_{}_{}.pt'.format('15-11-21a', 'naturalscene'))).cuda()
    kn3 = tdr.io.load_model(os.path.join(prepath,'convgc_{}_{}.pt'.format('15-11-21b', 'naturalscene'))).cuda()
    ns_predict = sqz_predict((kn1, kn2, kn3))

    ncells = 5 + 4 + 17                 # Total number of cells.
    phases = np.linspace(0, 1, 9)       # Grating phases.
    widths = np.arange(1, 9)            # Bar widths (checkers).
    sz = widths * 55.55                 # Bar widths (microns).
    F1, F2 = freq_doubling(ns_predict, ncells, phases, widths)
    fratio = F2 / F1
    mu = fratio.mean(axis=1)

    # Plot
    fig = plt.figure(figsize=(6,6))
    plt.style.use('deepretina.mplstyle')
    ax = fig.add_subplot(111)
    errorplot(sz, mu.mean(axis=0), mu.std(axis=0) / np.sqrt(ncells), method='line', fmt='o', color='#222222', ax=ax)
    ax.plot([0, 500], [1, 1], '--', color='lightgray', lw=1, zorder=0)
    #ax.set_xlabel('Bar width ($\mu m$)',fontsize=40)
    #ax.set_ylabel('$F_2/F_1$ component ratio',fontsize=40)
    ax.tick_params(axis='both', which='major', labelsize=40)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.xticks(ticks=[0,200,400])
    plt.yticks(ticks=[0,1,5])

    #ax.set_ylim(0, 8.2)
    plt.tight_layout()
    plt.savefig('convgc_f2_response.png')
    plt.savefig('convgc_f2_response.pdf')


if __name__ == "__main__":
    main()
