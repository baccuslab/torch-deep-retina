"""
The purpose of this script is to demonstrate the easiest way to load a
saved checkpoint of a saved model, load a provided dataset, and
finally perform inference on the data.

Rough Outline:
    read_model_file() reads the relevant definitional parameters from
        the checkpoint and creates the appropriate model architecture
        and loads the state_dict (saved model weights)
    
    Once loaded the model can be made ready for inference by calling
        model.eval() or model.train(False)

    The test data can be found by arguing the appropriate parameters
        to the loadexpt() function
"""

import torchdeepretina as tdr
from scipy.stats import pearsonr
import torch
import numpy as np

if torch.cuda.is_available():
    DEVICE = torch.device("cuda:0")
else:
    DEVICE = torch.device("cpu")

if __name__=="__main__":
    path_to_data = "/home/TRAIN_DATA/"
    # Name of the saved checkpoint
    file_name = "../models/15-11-21a_naturalscene.pt"

    checkpt = tdr.io.load_checkpoint(file_name)

    # Load Model from Checkpoint File
    model = tdr.io.load_model(file_name) 
    # Convert to non-stacked model if using LinearStacked model type
    model = tdr.utils.stacked2conv(model)
    model.to(DEVICE)
    # If you want to do inference, do not forget this line!!!
    model.eval()
    print(model)
    tot_params = 0
    for n,p in model.named_parameters():
        print(n,np.prod(p.shape))
        tot_params += np.prod(p.shape)
    print("All Param Count:", tot_params)

    # Load the Appropriate Data (Make sure you have the data located
    # at `path_to_data`
    dataset = checkpt['dataset']
    cells = checkpt['cells']
    stim_type = checkpt["stim_type"]

    # Number of movie frames seen in 1 datapoint
    temporal_depth = checkpt['img_shape'][0]
    mean = checkpt['norm_stats']['mean']
    std = checkpt['norm_stats']['std']

    # Necessary to z-score the test data with statistics of the
    # training data
    norm_stats = [mean, std]
    print("NormStats:", norm_stats)

    # Although the example does not need the train data, this is how
    # you would load it.
    train_data = tdr.datas.loadexpt(dataset, cells, stim_type, 'train',
                                             temporal_depth, nskip=0,
                                             norm_stats=None,
                                             data_path=path_to_data)
    print("Train data shape:", train_data.X.shape)

    test_data = tdr.datas.loadexpt(dataset, cells, stim_type, 'test',
                                             temporal_depth, nskip=0,
                                             norm_stats=norm_stats,
                                             data_path=path_to_data)
    print("Test data shape:", test_data.X.shape)
    print("Test data Mean:", test_data.X.mean())
    print("Test data StD:", test_data.X.std())

    # Compute model responses and determine pearson correlation with
    # ganglion cell output
    bsize = 500
    model_response = tdr.utils.inspect(model, test_data.X,
                                         batch_size=bsize,
                                         to_numpy=True)
    preds = model_response['outputs'] # Shape (N,M)
    truth = test_data.y # Shape (N,M)
    pearsons = tdr.utils.pearsonr(preds,truth)
    avg_pearson = np.mean(pearsons)
    print("Avg Ganglion Correlation:", avg_pearson)
    for i, cell in enumerate(test_data.cells):
        print("Cell {}: {}".format(cell, pearsons[i]))




