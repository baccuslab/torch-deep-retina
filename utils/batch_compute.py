import numpy as np
from .physiology import Physio
import torch

DEVICE = torch.device("cuda:0")
# Functions for correlation maps and loading David's stimuli in deep retina models.
def pad_to_edge(stim):
    '''
    Pads the spatial dimensions of a stimulus to be 50x50 for deep retina models.
    
    Args:
        stim: (time, space, space) numpy array.
    Returns:
        padded_stim: (time, space, space) padded numpy array.
    '''
    padded_stim = np.zeros((stim.shape[0], 50, 50))
    height = stim.shape[1]
    width = stim.shape[2]
    
    assert height < 50, 'Height must be less than 50.'
    assert width < 50, 'Width must be less than 50.'
    
    y_start = int((50 - height)/2.0)
    x_start = int((50 - width)/2.0)
    
    padded_stim[:, y_start:y_start+height, x_start:x_start+width] = stim
    return padded_stim


def batch_compute_model_response(stimulus, model, batch_size):
    '''
    Computes a model response in batches in pytorch. Returns a dict of lists 
    where each list is the sequence of batch responses.
    Args:
        stimulus: 3-d checkerboard stimulus in (time, space, space)
        model: the model
        batch_size: the size of the batch
    '''
    if(stimulus.shape[1] < 50 and stimulus.shape[2] < 50):
        stimulus = pad_to_edge(stimulus)
    stim = stimulus[:batch_size, :, :]
    phys = Physio(model)
    stim = torch.FloatTensor(stim)
    model_response = phys.inspect(stim.to(DEVICE)).copy()
    model_response['output'] = model_response['output'].cpu().detach().numpy()
    start = batch_size
    stop = 2*batch_size
    # Cylce through stimulus data in batches appending responses into model_response dict
    while stop < stimulus.shape[0]:
        stim = stimulus[start:stop, :, :]
        stim = torch.FloatTensor(stim)
        temp = phys.inspect(stim.to(DEVICE)).copy()
        temp['output'] = temp['output'].cpu().detach().numpy()
        for key in model_response.keys():
            model_response[key] = np.append(model_response[key], temp[key], axis=0)
        start = stop
        stop = start + batch_size
    stim = torch.FloatTensor(stimulus[start:,:,:])
    temp = phys.inspect(stim.to(DEVICE)).copy()
    temp['output'] = temp['output'].cpu().detach().numpy()
    for key in model_response.keys():
         model_response[key] = np.append(model_response[key], temp[key], axis=0)
    return model_response




