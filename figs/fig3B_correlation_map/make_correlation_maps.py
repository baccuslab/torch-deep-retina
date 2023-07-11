import torchdeepretina as tdr
import torch
import numpy as np
import h5py
import os
import scipy.stats
import re
import h5py
import matplotlib.pyplot as plt
from tqdm import tqdm

if __name__=="__main__":
    prepath = os.path.expanduser("~/src/torch-deep-retina/training_scripts/") # path to model files
    whitenoise_model_files =[
        "convgc/convgc_64_dataset15-11-21b_stim_typewhitenoise",
        "convgc/convgc_53_dataset15-10-07_stim_typewhitenoise",
        "convgc/convgc_57_dataset15-11-21a_stim_typewhitenoise",
    ]
    naturalscene_model_files = [
        'convgc/convgc_26_chans[8, 8]',
        'convgc/convgc_49_dataset15-10-07_stim_typenaturalscene',
        'convgc/convgc_54_dataset15-11-21a_stim_typenaturalscene'
    ]
    model_folders = [os.path.join(prepath,f) for f in naturalscene_model_files]
    layers = ["sequential.1","sequential.5"]


    filter_length = 40
    interneuron_data = tdr.intracellular.load_interneuron_data(root_path="~/interneuron_data/",filter_length=filter_length)
    stims, mem_pots, intrnrn_files = interneuron_data

    cor_maps = h5py.File("convgc_cor_maps.h5",'w')

    for cell_file in mem_pots.keys():
        for stim_key in mem_pots[cell_file].keys():
            main_stim = stims[cell_file][stim_key]
            print("CellFile:", cell_file, " --  Stim:", stim_key)
            for model_folder in model_folders:
                model = tdr.io.load_model(model_folder)
                model.eval()
                model.cuda()
                stim = tdr.stimuli.spatial_pad(main_stim,model.img_shape[1],model.img_shape[2])
                stim = tdr.stimuli.concat(stim,nh=model.img_shape[0])
                model_output = tdr.utils.inspect(model, stim, insp_keys=layers, batch_size=500)
                for ci, mem_pot in enumerate(mem_pots[cell_file][stim_key]):
                    for l,layer in enumerate(layers):
                        for chan in tqdm(range(model.chans[1])):
                            response = model_output[layer].reshape((-1,model.chans[l],*model.shapes[l]))[:,chan]
                            cor_map = tdr.intracellular.correlation_map(mem_pot, response)
                            #plt.imshow(cor_map,cmap = 'seismic', clim=[-np.max(abs(cor_map)), np.max(abs(cor_map))])
                            #plt.show()
                            row,col = np.unravel_index(np.argmax(cor_map),cor_map.shape)
                            max_resp = response[:,row,col]
                            name = "_".join(model_folder.split("/")[-1].split("_")[:2])
                            f = cell_file.split("/")[-1].split(".")[0]
                            path = "{}/{}/{}/{}/{}/{}/".format(f,stim_key,name,ci,layer,chan)
                            cor_maps.create_dataset(path+"cor_map",data=cor_map)
                            cor_maps.create_dataset(path+"max_resp",data=max_resp)
                            cor_maps.create_dataset(path+"real_resp",data=mem_pot)

    cor_maps.close()








