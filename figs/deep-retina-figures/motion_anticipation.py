"""Motion anticipation figure."""

import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import torchdeepretina as tdr
from torchdeepretina.io import load_model

prepath = "convgc_"

def sqz_predict(mdls):
    """Squeeze predictions from multiple models into one array."""
    def sqz(X):
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
    return sqz


def upsample(x, ys, x_us):
    """Smoothly upsample a firing rate response."""
    return np.stack([interp1d(x, y, kind='cubic')(x_us) for y in ys])


def get_responses(velocity):
    """Get motion anticipation responses."""

    # Load natural scene models.
    kn1 = load_model('/home/grantsrb/src/torch-deep-retina/models/'+prepath+'15-10-07_naturalscene.pt')
    kn1.cuda()
    kn1.eval()
    kn2 = load_model('/home/grantsrb/src/torch-deep-retina/models/'+prepath+'15-11-21a_naturalscene.pt')
    kn2.cuda()
    kn2.eval()
    kn3 = load_model('/home/grantsrb/src/torch-deep-retina/models/'+prepath+'15-11-21b_naturalscene.pt')
    kn3.cuda()
    kn3.eval()

    #ns_predict = sqz_predict((kn1,))
    _, _, (cr, _, rr), (cl, _, rl), (fc, fr) = tdr.retinal_phenomena.motion_anticipation(kn1, scale_factor=55, velocity=velocity, width=2, flash_duration=2, make_fig=True)

    fr_pop = fr.mean(axis=2)
    fcu = np.linspace(fc[6], fc[30], 100)
    z = 55.55 * (fcu - fcu.mean())          # 55.55 microns per checker
    fr_us = upsample(fc, fr_pop.T, fcu)

    # Time of the maximal response.
    peak_index = fr_us.max(axis=1).argmax()

    # Get the peak response.
    flash = fr_us[peak_index]

    rl_us = upsample(cl[40:], rl.T, np.linspace(-19, 5, 100))
    rr_us = upsample(cr[40:], rr.T, np.linspace(-19, 5, 100))

    left = rl_us.mean(axis=0)
    right = rr_us.mean(axis=0)

    maxval = max(left.max(), right.max())
    left /= maxval
    right /= maxval

    return z, flash / flash.max(), left, right


def main():
    """Generate figure."""
    zz, flash, left, right = get_responses(0.176)
    #plt.savefig('temp.png')

    # Plot
    plt.clf()
    fig = plt.figure(figsize=(12, 5))
    red = '#E62538'
    blue = '#0168A2'
    green = '#009F4F'

    ax = fig.add_subplot(121)
    ax.plot(zz, flash, color=red, linewidth=4)
    ax.plot(zz, right, color=blue,linewidth=4)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_xlim(-250, 250)
    ax.set_xticks([-200, 0, 200])
    ax.set_yticks([0, 1])
    ax.set_yticklabels(['0', 'Max'])
    ax.tick_params(axis='both', which='major', labelsize=30)

    ax = fig.add_subplot(122)
    ax.plot(zz, flash, color=red,linewidth=4)
    ax.plot(zz, left, color=green,linewidth=4)
    ax.set_xlim(-250, 250)
    ax.set_xticks([-200,0, 200])
    ax.yaxis.set_label_position("right")
    ax.yaxis.tick_right()
    ax.set_yticks([0, 1])
    ax.set_yticklabels(['0', 'Max'])
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.tick_params(axis='both', which='major', labelsize=30)

    plt.locator_params(nbins=3)
    plt.tight_layout()

    plt.savefig(prepath+'motion_anticipation.png')
    plt.savefig(prepath+'motion_anticipation.pdf')


if __name__ == "__main__":
    main()
