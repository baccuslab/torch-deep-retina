import torchdeepretina as tdr
import torch
import numpy as np
import h5py
import os
import scipy.stats
import re
import h5py
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from tqdm import tqdm

if __name__=="__main__":
    cor_maps = h5py.File("convgc_cor_maps.h5",'r')

    for cell_file in cor_maps.keys():
        for stim_key in cor_maps[cell_file].keys():
            path = "{}/{}/".format(cell_file,stim_key)
            for model_folder in cor_maps[path].keys():
                path = "{}/{}/{}".format(cell_file,stim_key,model_folder)
                for ci in cor_maps[path].keys():
                    path = "{}/{}/{}/{}".format(cell_file,stim_key,model_folder,ci)
                    for layer in cor_maps[path].keys():
                        path = "{}/{}/{}/{}/{}".format(cell_file,stim_key,model_folder,ci,layer)
                        for chan in tqdm(cor_maps[path].keys()):
                            path = "{}/{}/{}/{}/{}/{}/".format(cell_file,stim_key,model_folder,ci,layer,chan)
                            cor_map = np.asarray(cor_maps[path+"cor_map"])
                            max_resp = np.asarray(cor_maps[path+"max_resp"])
                            real_resp = np.asarray(cor_maps[path+"real_resp"])


                            fig=plt.figure(figsize=(9,10), dpi= 80, facecolor='w', edgecolor='k')
                            plt.clf()
                            gridspec.GridSpec(4,1)

                            plt.subplot2grid((4,1),(0,0),rowspan=2)
                            high_cor = round(np.max(abs(cor_map)),2)
                            low_cor = -high_cor

                            if high_cor < 0.55:
                                continue

                            plt.imshow(cor_map.squeeze(), cmap = 'seismic', clim=[low_cor, high_cor])
                            ax = plt.gca()
                            ax.set_xticks([])
                            ax.set_yticks([])
                            ax.set_xticklabels([])
                            ax.set_yticklabels([])
                            cbar = plt.colorbar()
                            cbar.ax.yaxis.set_major_locator(plt.LinearLocator(2))
                            cbar.ax.set_yticks([low_cor,high_cor])
                            cbar.ax.set_yticklabels([str(low_cor),str(high_cor)])
                            cbar.ax.tick_params(axis='both', which='major', labelsize=17)
                            cbar.set_label("Correlation", rotation=270)

                            plt.subplot2grid((4,1),(2,0))
                            ax = plt.gca()
                            real_z = (real_resp.squeeze()-real_resp.mean())/real_resp.std()
                            max_z = (max_resp.squeeze()-max_resp.mean())/max_resp.std()
                            ax.plot(real_z[:200],'k',linewidth=3)
                            ax.plot(max_z[:200],color='#0066cc',linewidth=7,alpha=.7)
                            ax.xaxis.set_major_locator(plt.LinearLocator(2))
                            for k in ax.spines.keys():
                                ax.spines[k].set_linestyle("--")
                                ax.spines[k].set_linewidth(4)
                                ax.spines[k].set_color("#0066cc")
                            ax.set_facecolor("#d4ebf2")
                            ax.set_xlabel("Time (s)",fontsize=20)
                            ax.tick_params(axis='both', which='major', labelsize=17)

                            plt.tight_layout()
                            ax.set_xticks([0,2],[0,2])
                            plt.savefig("figs/{}_{}_{}{}.pdf".format(model_folder,ci,layer,chan))
                            plt.savefig("figs/{}_{}_{}{}.png".format(model_folder,ci,layer,chan))

    cor_maps.close()









