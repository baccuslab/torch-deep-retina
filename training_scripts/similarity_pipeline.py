import torch
import torchdeepretina as tdr
import numpy as np
import h5py as h5
import matplotlib.pyplot as plt
import time
import sys
import os

if __name__=="__main__":
    grand_folders = sys.argv[1:]
    torch.cuda.empty_cache()
    model_paths = None
    save_name = "similarities.csv"
    for grand_folder in grand_folders:
        print("Analyzing", grand_folder)
        paths = tdr.io.get_model_folders(grand_folder)
        paths = [os.path.join(grand_folder,path) for path in paths]
        if model_paths is None:
            model_paths = paths
        else:
            model_paths = model_paths + paths
        save_name = grand_folder.replace("/","") + "_" + save_name
    print("Models:")
    print("\n".join(model_paths))
    print("Saving to:", save_name)
    df = tdr.analysis.similarity_pipeline(model_paths,
                                          calc_cor=True,
                                          calc_cca=False,
                                          np_cca=False,
                                          n_samples=20000,
                                          save_file=save_name,
                                          verbose=True)
    df.to_csv(save_name, sep="!", header=True, index=False)
