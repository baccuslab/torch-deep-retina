"""Latency encoding figure."""


import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy.interpolate import interp1d

from torchdeepretina.io import load_model
import stimuli as s
import torchdeepretina.stimuli as tdrstim

from tqdm import tqdm


def upsample(r):
    t = np.linspace(0, 300, 30)
    ts = np.linspace(0, 200, 1000)
    return interp1d(t, r, kind='cubic')(ts)


def latenc(mdls, intensities):
    rates = []
    for i in tqdm(intensities):
        X = s.flash(2, 40, 70, intensity=i)
        X = np.broadcast_to(X,(X.shape[0],*mdls[0].img_shape[1:]))
        X = tdrstim.rolling_window(X,mdls[0].img_shape[0])
        X = torch.FloatTensor(X).cuda()
        preds = []
        for mdl in mdls:
            mdl.cuda()
            mdl.eval()
            pred = mdl(X).detach().cpu().numpy()
            preds.append(pred)
            mdl.cpu()
            mu,std = mdl.norm_stats['mean'],mdl.norm_stats['std']
        r = np.hstack(preds)
        rates.append(r)
    return np.stack(rates)

def main():
    prename = "convgc_"
    prepath = '/home/grantsrb/src/torch-deep-retina/models/'
    threshold = 3

    # Load natural scene models.
    kn1 = load_model(prepath+prename+'15-10-07_naturalscene.pt')
    kn2 = load_model(prepath+prename+'15-11-21a_naturalscene.pt')
    kn3 = load_model(prepath+prename+'15-11-21b_naturalscene.pt')
    nats = [kn1,kn2,kn3]
    for nat in nats:
        nat.eval()

    # Load white noise models.
    km1 = load_model(prepath+prename+'15-10-07_whitenoise.pt')
    km2 = load_model(prepath+prename+'15-11-21a_whitenoise.pt')
    km3 = load_model(prepath+prename+'15-11-21b_whitenoise.pt')
    whits = [km1,km2,km3]
    for whit in whits:
        whit.eval()

    # Time and Contrasts.
    ts = np.linspace(0, 200, 1000)
    intensities = np.linspace(0, -3, 21)

    wn = latenc((km1, km2, km3), intensities)
    ns = latenc((kn1, kn2, kn3), intensities)

    wn_lats = []
    ns_lats = []
    endx = 750
    cii = 1
    nat_color = 'lightcoral'
    whit_color = '#888888'
    for ci in range(wn.shape[-1]):
        #wn_resp = wn.mean(axis=-1)
        wn_resp = wn[...,ci]
        wn_resps = np.stack([upsample(ri) for ri in wn_resp])
        wn_resps[wn_resps<0] = 0
        ns_resp = ns[...,ci]
        ns_resps = np.stack([upsample(ri) for ri in ns_resp])
        ns_resps[ns_resps<0] = 0

        if ci == cii:
            # Make figure.
            plt.style.use('deepretina.mplstyle')
            fig = plt.figure(figsize=(21, 6))

            # Panel 1: White noise responses.
            if cii == 0:
                subp = 131
            else:
                subp = 121
            ax1 = fig.add_subplot(subp)
            colors = cm.gray_r(np.linspace(0, 1, len(wn_resps)))
            for r, c in zip(wn_resps, colors):
                ax1.plot(ts[:endx], r[:endx], '-', color=c)
            #ax1.set_xlabel('Time (s)',fontsize=40)
            #ax1.set_ylabel('Response (Hz)',fontsize=40)
            ax1.set_title('White Noise',fontsize=40)
            ax1.spines['top'].set_visible(False)
            ax1.spines['right'].set_visible(False)
            plt.locator_params(nbins=4)
            ax1.tick_params(axis='both', which='major', labelsize=35)

            # Panel 2: Natural scene responses.
            if cii == 0:
                subp = 132
            else:
                subp = 122
            ax2 = fig.add_subplot(subp)
            colors = cm.Reds(np.linspace(0, 1, len(ns)))
            for r, c in zip(ns_resps, colors):
                ax2.plot(ts[:endx], r[:endx], '-', color=c)
            #ax2.set_xlabel('Time (s)',fontsize=40)
            #ax2.set_ylabel('Response (Hz)',fontsize=40)
            ax2.set_title('Natural Scenes',fontsize=40)
            ax2.spines['top'].set_visible(False)
            ax2.spines['right'].set_visible(False)
            plt.locator_params(nbins=4)
            ax2.tick_params(axis='both', which='major', labelsize=35)
            if cii != 0:
                plt.locator_params(nbins=3)
                plt.tight_layout()
                plt.savefig("lat_encs/"+prename+'latency_encoding_latencies_ci{}.png'.format(ci))
                plt.savefig("lat_encs/"+prename+'latency_encoding_latencies_ci{}.pdf'.format(ci))
                return

        wn_resps[wn_resps<threshold] = 0
        ## First Local Max
        #temp = wn_resps.copy()
        #temp[temp<threshold] = 0
        #local_maxes = ((temp[:,:-1]-temp[:,1:])>0)[:,1:]&((temp[:,1:]-temp[:,:-1])>0)[:,:-1]
        #argmax = np.argmax(local_maxes,axis=1)+1

        ## Global Max
        #argmax = (wn_resps>threshold).argmax(axis=1)

        # Center of Mass
        temp = wn_resps[:,:endx]
        idxs = np.arange(endx)
        s = temp.sum(1)
        s[s<=0.00001] = -1
        argmax = ((idxs*temp).sum(1)/s).astype(np.int)
        argmax[argmax<=0] = 0

        wn_lat = ts[argmax]
        wn_lat[wn_lat==ts[0]] = np.nan
        wn_lats.append(wn_lat)

        # Natural scenes.
        #ns_resp = ns.mean(axis=-1)
        ns_resps[ns_resps<threshold] = 0

        ## FIrst Local Max
        #temp = ns_resps.copy()
        #temp[temp<threshold] = 0
        #local_maxes = ((temp[:,:-1]-temp[:,1:])>0)[:,1:]&((temp[:,1:]-temp[:,:-1])>0)[:,:-1]
        #argmax = np.argmax(local_maxes,axis=1)+1

        # Global Max
        #argmax = (ns_resps>threshold).argmax(axis=1)

        # Center of Mass
        temp = ns_resps[:,:endx]
        idxs = np.arange(endx)
        s = temp.sum(1)
        s[s<=0.00001] = -1
        argmax = ((idxs*temp).sum(1)/s).astype(np.int)
        argmax[argmax<=0] = 0

        ns_lat = ts[argmax]
        ns_lat[ns_lat==ts[0]] = np.nan
        ns_lats.append(ns_lat)
        #ns_lat = ts[ns_resps.argmax(axis=1)]
        #ns_lats.append(ns_lat)


        ## Save.
        #plt.tight_layout()
        #plt.savefig("lat_encs/"+prename+'latency_encoding'+str(ci)+'.png')
        #plt.savefig("lat_encs/"+prename+'latency_encoding'+str(ci)+'.pdf')

    # Panel 3: Latency vs. Intensity
    # Make figure.
    #plt.style.use('deepretina.mplstyle')
    #fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(133)
    wn_lats = np.asarray(wn_lats)
    wn_lats = [wn_lats[:,i][wn_lats[:,i]==wn_lats[:,i]].mean() for i in range(wn_lats.shape[1])]
    ns_lats = np.asarray(ns_lats)
    ns_lats = [ns_lats[:,i][ns_lats[:,i]==ns_lats[:,i]].mean() for i in range(ns_lats.shape[1])]
    ax.plot(-intensities, wn_lats, 'o', color=whit_color)
    ax.plot(-intensities, ns_lats, 'o', color=nat_color)
    #ax.set_xlabel('Intensity',fontsize=40)
    #ax.set_ylabel('Latency',fontsize=40)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.locator_params(nbins=4)
    ax.tick_params(axis='both', which='major', labelsize=35)

    # Add labels.
    #plt.xlabel('Intensity (a.u.)',fontsize=20)
    #plt.ylabel('Latency (ms)',fontsize=20)
    #plt.ylim(70, 100)
    #plt.xlim(0, 10.5)
    #plt.legend(['White Noise', 'Natural Scenes'],bbox_to_anchor=(.5, 1), loc=2,
    l = plt.legend(['White Noise', 'Natural Scenes'], loc='upper right',
                                                    borderaxespad=0.,fontsize=30,
                                                    handletextpad=-2.0, handlelength=0,
                                                    markerscale=0,
                                                    frameon=False)
    colors = [whit_color, nat_color]
    for color,text in zip(colors, l.get_texts()):
        text.set_color(color)
    plt.locator_params(nbins=3)
    plt.tight_layout()
    plt.savefig("lat_encs/"+prename+'latency_encoding_latencies_ci{}.png'.format(cii))
    plt.savefig("lat_encs/"+prename+'latency_encoding_latencies_ci{}.pdf'.format(cii))


if __name__ == "__main__":
    main()
