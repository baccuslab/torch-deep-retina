import numpy as np
import matplotlib.pyplot as plt
import torch
import torchdeepretina as tdr
import os

datasets = ['15-11-21a'] #['15-10-07' , '15-11-21a', '15-11-21b']
stim_types = ['naturalscene', 'whitenoise']
mtype = "" # other options are "convgc_" and "drone_"
img_folder = "figs/"

prepath = os.path.expanduser("~/src/torch-deep-retina/models/")

c = 0.23 # Contrast
contrasts = np.asarray([c, 7*c])
for dataset in datasets:
    for stim_type in stim_types:
        model_name = mtype+dataset+"_"+stim_type+".pt"
        model_file = os.path.join(prepath, model_name)
        model = tdr.io.load_model(model_file)
        model.cuda()
        model.eval()
        print("Analyzing", model_name)
        low,high = contrasts
        print("Contrasts:", low, high)
        #for unit in range(model.n_units):
        #    fig = tdr.retinal_phenomena.contrast_fig(model, contrasts, unit_index=unit, 
        #                                         nonlinearity_type="bin", verbose=True)
        #    fig.savefig("convgc_{}_{}_unit{}_ctrlow{}_ctrhigh{}.png".format(dataset,stim_type,unit,
        #                                                                                low,high))
        #    fig.savefig("convgc_{}_{}_unit{}_ctrlow{}_ctrhigh{}.pdf".format(dataset,stim_type,unit,
        #                                                                                low,high))
        for i in [0.25, 0.5, 1, 1.25, 1.5, 2]:
            low,high = contrasts*i
            for unit in range(model.n_units):
                print("Contrasts:", low, high)
                fig = tdr.retinal_phenomena.contrast_fig(model, [low,high], unit_index=unit, 
                                                    nonlinearity_type="bin", verbose=True)
                fig.savefig(os.path.join(
                    img_folder,
                    "{}{}_{}_unit{}_ctrlow{:04e}_ctrhigh{:04e}.png".format(mtype,dataset,stim_type,unit, low,high)
                ))
