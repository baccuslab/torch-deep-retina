"""
This script is made to automate the analysis of the model performance
for a batch of models.  You must give a command line argument of the
model search folder to be analyzed.

$ python3 analysis_pipeline.py bncnn

"""
import numpy as np
import torch
import sys
import torchdeepretina.analysis as analysis

if __name__ == "__main__":
    start_idx = None
    calc_intrs = False
    if len(sys.argv) >= 2:
        try:
            start_idx = int(sys.argv[1])
            grand_folders = sys.argv[2:]
        except:
            grand_folders = sys.argv[1:]
    torch.cuda.empty_cache()
    for grand_folder in grand_folders:
        print("Analyzing", grand_folder)
        dfs = analysis.analysis_pipeline(grand_folder,
                                      make_figs=False,
                                      make_model_rfs=False,
                                      slide_steps=0,
                                      intrnrn_stim='boxes',
                                      save_dfs=True,
                                      rec_intrs=True, #If false, cors are still calculated, details just aren't recorded
                                      calc_intrs=calc_intrs,
                                      verbose=True)

