import torch
import numpy as np
from torchdeepretina.analysis import read_model_file, batch_compute_model_response
from torchdeepretina.datas import loadexpt
import torch.nn as nn
import sys
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, BatchNormalization, Dense, Dropout, Flatten, Reshape, Activation, Input
from tensorflow.keras import backend
import scipy.stats

def transfer_bnorm(tmodel, kmodel, t_idx, k_idx):
    bnorm = tmodel.sequential[t_idx]

    gamma = bnorm.scale.data.abs().cpu().numpy()
    kmodel.layers[k_idx].gamma.assign(gamma)

    beta = bnorm.shift.data.cpu().numpy()
    kmodel.layers[k_idx].beta.assign(beta)

    moving_mean = bnorm.running_mean.data.cpu().numpy()
    kmodel.layers[k_idx].moving_mean.assign(moving_mean)

    moving_var = bnorm.running_var.data.cpu().numpy()
    kmodel.layers[k_idx].moving_variance.assign(moving_var)

def transfer_weights(tmodel, kmodel):
    # First conv stack
    convs = tmodel.sequential[0].convs
    idx = 1
    for conv in convs:
        if "Conv2d" in str(conv):
            weight = conv.weight.data.permute((2,3,1,0)).cpu().numpy()
            kmodel.layers[idx].weights[0].assign(weight)
            if conv.bias is not None:
                kmodel.layers[idx].weights[1].assign(conv.bias.data.cpu().numpy())
            idx += 1
    
    # AbsBnorm0
    idx += 1
    transfer_bnorm(tmodel, kmodel, 2, idx)
    idx+=3
    
    # Second conv stack
    convs = tmodel.sequential[6].convs
    for i,conv in enumerate(convs):
        if "Conv2d" in str(conv):
            weight = conv.weight.data.permute((2,3,1,0)).cpu().numpy()
            kmodel.layers[idx].weights[0].assign(weight)
            if conv.bias is not None:
                kmodel.layers[idx].weights[1].assign(conv.bias.data.cpu().numpy())
            idx += 1
    
    # AbsBnorm1
    idx += 1
    transfer_bnorm(tmodel, kmodel, 8, idx)
    idx+=2
    
    # Fully Connected
    dense = tmodel.sequential[11]
    kmodel.layers[idx].weights[0].assign(dense.weight.data.cpu().numpy().transpose((1,0)))
    if dense.bias is not None:
        kmodel.layers[idx].weights[1].assign(dense.bias.data.cpu().numpy())
    
    # AbsBnorm2
    idx += 1
    transfer_bnorm(tmodel, kmodel, 12, idx)

def linstack_to_keras(inputs, model):
    fx = inputs

    # First conv stack
    mod = model.sequential[0]
    for i,conv in enumerate(mod.convs):
        if "Conv2d" in str(next(conv.modules())):
            use_bias = conv.bias is not None
            v = conv.weight
            fx = Conv2D(v.shape[0], kernel_size=(v.shape[2],v.shape[3]), data_format="channels_first", use_bias=use_bias)(fx)

    fx = Flatten()(fx)
    fx = BatchNormalization(axis=-1)(fx)
    fx = Activation('relu')(fx)
    fx = Reshape((model.chans[0], model.shapes[0][0], model.shapes[0][1]))(fx)

    # Second conv stack
    mod = model.sequential[6]
    for i, conv in enumerate(mod.convs):
        if "Conv2d" in str(next(conv.modules())):
            use_bias = conv.bias is not None
            v = conv.weight
            fx = Conv2D(v.shape[0], kernel_size=(v.shape[2],v.shape[3]), data_format="channels_first", use_bias=use_bias)(fx)

    fx = Flatten()(fx)
    fx = BatchNormalization(axis=-1)(fx)
    fx = Activation('relu')(fx)
    fx = Dense(model.n_units, use_bias=model.sequential[11].bias is not None)(fx)
    fx = BatchNormalization(axis=-1)(fx)
    outputs = Activation("softplus")(fx)
    kmodel = Model(inputs, outputs, name="linstack")
    return kmodel

if __name__ == "__main__":
    file_name = "checkpoint.pt"
    if len(sys.argv) > 1:
        file_name = str(sys.argv[1])
    print("Using model file:", file_name)
    tmodel = read_model_file(file_name) # Loads model architecture and saved weights
    tmodel.eval()
    tmodel.to(0)
    #print(tmodel)

    # Get Data
    print("Loading data...")
    path_to_data = "~/experiments/data"
    checkpt = torch.load(file_name, map_location="cpu")
    dataset = checkpt['dataset']
    cells = checkpt['cells']
    stim_type = checkpt["stim_type"]
    temporal_depth = checkpt['img_shape'][0] # Number of movie frames seen in 1 datapoint
    mean, std = checkpt['norm_stats']['mean'], checkpt['norm_stats']['std']
    norm_stats = [mean, std] # z-scores the test data using the same statistics as the training data
    test_data = loadexpt(dataset, cells, stim_type, 'test', temporal_depth, 0, norm_stats=norm_stats, 
                                                                                data_path=path_to_data)

    print()
    print("Creating Keras Version...")
    print()
    inputs = Input(shape=tmodel.img_shape)
    kmodel = linstack_to_keras(inputs, tmodel)

    transfer_weights(tmodel,kmodel)

    # PyTorch Inference
    model_response = batch_compute_model_response(test_data.X, tmodel, 500, recurrent=tmodel.recurrent)
    pearsons = [scipy.stats.pearsonr(model_response['output'][:, i], test_data.y[:, i])[0] 
                                                    for i in range(test_data.y.shape[-1])]
    avg_pearson = np.mean(pearsons)
    print("Torch Avg Ganlion Correlation:", avg_pearson)
    for i, cell in enumerate(test_data.cells):
        print("Cell {}: {}".format(cell, pearsons[i]))

    # Keras inference
    preds = kmodel.predict(test_data.X)
    pearsons = [scipy.stats.pearsonr(preds[:, i], test_data.y[:, i])[0] 
                                                    for i in range(test_data.y.shape[-1])]
    avg_pearson = np.mean(pearsons)
    print("Keras Avg Ganlion Correlation:", avg_pearson)
    for i, cell in enumerate(test_data.cells):
        print("Cell {}: {}".format(cell, pearsons[i]))
