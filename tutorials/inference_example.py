"""
The purpose of this script is to demonstrate the easiest way to load a saved checkpoint of a saved model,
load a provided dataset, and finally perform inference on the data.

Rough Outline:
    read_model_file() reads the relevant definitional parameters from the checkpoint and creates the 
    appropriate model architecture and loads the state_dict (saved model weights)
    
    Once loaded the model can be made ready for inference by calling model.eval() or model.train(False)

    The test data can be found by arguing the appropriate parameters to the loadexpt() function
"""

from torchdeepretina.analysis import read_model_file, batch_compute_model_response
from torchdeepretina.datas import loadexpt
import scipy.stats
import torch
import numpy as np

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

if __name__=="__main__":
    path_to_data = "~/experiments/data"
    file_name = "checkpoint.pt" # Name of the saved checkpoint

    checkpt = torch.load(file_name, map_location="cpu")

    # Load Model from Checkpoint File
    model = read_model_file(file_name) 
    model.to(device)
    model.eval() # If you want to do inference, do not forget this line!!!

    # Load the Appropriate Data (Make sure you have the data located at ~/experiments/data
    dataset = checkpt['dataset']
    cells = checkpt['cells']
    stim_type = checkpt["stim_type"]
    temporal_depth = checkpt['img_shape'][0] # Number of movie frames seen in 1 datapoint
    mean, std = checkpt['norm_stats']['mean'], checkpt['norm_stats']['std']
    norm_stats = [mean, std] # Used to z-score the test data using the same statistics as the training data
    test_data = loadexpt(dataset, cells, stim_type, 'test', temporal_depth, 0, norm_stats=norm_stats, 
                                                                                data_path=path_to_data)

    # Compute model responses and determine pearson correlation with ganglion cell output
    batch_size = 500
    model_response = batch_compute_model_response(test_data.X, model, batch_size, recurrent=model.recurrent)
    print(model_response['output'])
    pearsons = [scipy.stats.pearsonr(model_response['output'][:, i], test_data.y[:, i])[0] 
                                                    for i in range(test_data.y.shape[-1])]
    avg_pearson = np.mean(pearsons)
    print("Avg Ganlion Correlation:", avg_pearson)
    for i, cell in enumerate(test_data.cells):
        print("Cell {}: {}".format(cell, pearsons[i]))
    




