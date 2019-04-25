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


def batch_compute_model_response(stimulus, model, batch_size, insp_keys={'all'}):
    '''
    Computes a model response in batches in pytorch. Returns a dict of lists 
    where each list is the sequence of batch responses.     
    Args:
        stimulus: 3-d checkerboard stimulus in (time, space, space)
        model: the model
        batch_size: the size of the batch
        insp_keys: set (or dict) with keys of layers to be inspected in Physio
    '''
    if(stimulus.shape[1] < 50 and stimulus.shape[2] < 50):
        stimulus = pad_to_edge(stimulus)
    phys = Physio(model)
    model_response = None
    for i in range(0, stimulus.shape[0], batch_size):
        stim = torch.FloatTensor(stimulus[i:i+batch_size])
        if model_response is None:
            model_response = phys.inspect(stim.to(DEVICE), insp_keys=insp_keys).copy()
            model_response['output'] = model_response['output'].cpu().detach().numpy()
        else:
            temp = phys.inspect(stim.to(DEVICE), insp_keys=insp_keys).copy()
            temp['output'] = temp['output'].cpu().detach().numpy()
            for key in model_response.keys():
                 model_response[key] = np.append(model_response[key], temp[key], axis=0)
            del temp

    # Get the last few samples
    phys.remove_hooks()
    phys.remove_refs()
    del phys
    return model_response




