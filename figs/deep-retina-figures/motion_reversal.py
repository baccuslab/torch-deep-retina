"""Motion reversal figure."""

import numpy as np
import torch
import matplotlib.pyplot as plt

from tqdm import tqdm
from jetpack import errorplot

import stimuli as s
from torchdeepretina.io import load_model


def motion_sweep(xstop, speed=0.24):
    """Sweeps a bar across the screen."""
    c, _, X = s.driftingbar(speed=speed, x=(-40, xstop))
    centers = c[40:]
    return centers, s.concat(X)


def sqz(mdls, X):
    """Squeeze predictions from multiple models into one array."""
    X = torch.FloatTensor(X).cuda()
    preds = []
    for mdl in mdls:
        mdl.cuda()
        mdl.eval()
        pred = mdl(X).detach().cpu().numpy()
        preds.append(pred)
        mdl.cpu()
    r = np.hstack(preds)
    return r


def run_motion_reversal(x_locations, models, speed=0.19, clip_n=210, scaling=1):
    """Gets responses to a bar reversing motion."""
    tflips, Xs = zip(*[s.motion_reversal(xi,speed)[1:] for xi in x_locations])
    data = [sqz(models, scaling*X) for X in tqdm(Xs)]
    fps = 0.01
    time = np.linspace(-tflips[0], (len(Xs[0])-tflips[0]), len(Xs[0]))
    time = time*fps
    deltas = np.array(tflips) - np.array(tflips)[0]
    datacut = [data[0]]
    datacut += [data[i][deltas[i]:-deltas[i]] for i in range(1, len(deltas))]
    cn = int((datacut[0].shape[0]-clip_n)//2)
    t = time[cn:-cn]
    data_clipped = [d[cn:-cn] for d in datacut]
    mu = np.hstack(data_clipped).mean(axis=1)
    sig = np.hstack(data_clipped).std(axis=1) / np.sqrt(60)

    clip_n = len(data_clipped[0])
    #offset = 40//2
    #speed = 0.01
    #t = np.linspace(-clip_n//2 * speed , clip_n//2 * speed, clip_n) + offset*speed
    return t, mu, sig, deltas


def run_motion_sweep(x_locations, models, deltas,speed=0.24, clip_n=210):
    """Gets responses to a bar sweeping across the screen (no reversal)."""
    Xs = [motion_sweep(40 + xi,speed=speed)[1] for xi in x_locations]
    rs = [sqz(models, X) for X in tqdm(Xs)]
    rcut = [rs[0]]
    rcut += [rs[i][deltas[i]:-deltas[i]] for i in range(1, len(deltas))]
    #cn = int((rcut[0].shape[0]-clip_n)//2)
    R = np.hstack([r[:clip_n, :] for r in rcut])

    mu = R.mean(axis=1)
    sig = R.std(axis=1) / np.sqrt(60)
    offset = 0
    speed = 0.01
    t = np.linspace((-clip_n//2 + offset) * speed , (clip_n//2 + offset) * speed , clip_n)

    return t, mu, sig


def main():
    prepath = "convgc_"

    # Load natural scene models.
    kn1 = load_model('/home/grantsrb/src/torch-deep-retina/models/'+prepath+'15-10-07_naturalscene.pt')
    # kn2 = drone.load_model('15-11-21a', 'naturalscene')
    # kn3 = drone.load_model('15-11-21b', 'naturalscene')

    # Load white noise models.
    km1 = load_model('/home/grantsrb/src/torch-deep-retina/models/'+prepath+'15-10-07_whitenoise.pt')
    # km2 = drone.load_model('15-11-21a', 'whitenoise')
    # km3 = drone.load_model('15-11-21b', 'whitenoise')

    #for model,stim_type in zip([kn1,km1], ["naturalscene","whitenoise"]):
    speed = 0.14
    for scale in np.arange(0.5,3.6,0.5):
        for model, wn_model in zip([kn1],[km1]):
            # Plot
            clip_n = 210
            fig = plt.figure(figsize=(10, 7))

            # Plot motion reversal response.
            xs = np.arange(-9, 3)
            t, mu, sig, deltas = run_motion_reversal(xs, (model,), speed, clip_n=clip_n, scaling=scale)
            #sig = np.zeros_like(sig)
            ax = fig.add_subplot(111)
            nat_color = 'lightcoral'
            ax.plot(t,mu,color=nat_color, linewidth=4)
            errorplot(t, mu, sig, color='lightcoral', ax=ax, linewidth=5)

            #t, mu, sig, deltas = run_motion_reversal(xs, (wn_model,), speed, clip_n=clip_n, scaling=3.9)
            ##ax = fig.add_subplot(122)
            #sig = np.zeros_like(sig)
            #whit_color = '#888888'
            #ax.plot(t,mu,color=whit_color, linewidth=4)
            ##errorplot(t, mu, sig, color='lightblue', ax=ax, linewidth=5)

            ax.set_xlim(-.7, 1.1)
            ax.set_ylim(0, 8)
            #ax.set_ylabel('Firing Rate (Hz)',fontsize=20)
            #ax.set_xlabel('Time (s)',fontsize=20)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            #ax.spines['left'].set_visible(False)
            ax.tick_params(axis='both', which='major', labelsize=35)
            plt.locator_params(nbins=3)
            plt.yticks(ticks=[0,5])
            plt.xticks(ticks=[-0.5, 0, 0.5, 1], labels=[-0.5, 0, 0.5, 1])
            l = plt.legend(['Natural Scenes'], fontsize=35, frameon=False,
                                                        handlelength=0,
                                                        markerscale=0)
            colors = [nat_color]
            for color,text in zip(colors, l.get_texts()):
                text.set_color(color)

            plt.tight_layout()

            plt.savefig("mot_rev/"+prepath+'motion_reversal'+str(scale)+'.png')
            plt.savefig("mot_rev/"+prepath+'motion_reversal'+str(scale)+'.pdf')


if __name__ == "__main__":
    main()
