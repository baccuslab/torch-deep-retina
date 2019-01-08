from physiology import Physio
from drone import stimuli

def batch_compute_model_response(stimulus, model, batch_size):
    '''
    Computes a model response in batches in pytorch.
    Args:
        stimulus: 3-d checkerboard stimulus in (time, space, space)
        model: the model
        batch_size: the size of the batch
    '''
    stimulus = pad_to_edge(stimulus)
    concat_stim = stimuli.concat(stimulus)
    stim = concat_stim[:batch_size, :, :]
    phys = Physio(model)
    model_response = phys.inspect(stim)
    start = batch_size
    stop = 2*batch_size
    while stop < stimulus.shape[0]:
        stim = concat_stim[start:stop, :, :]
        temp = phys.inspect(stim)
        for key in model_response.keys():
        	model_response[key] = np.append(model_response[key], temp[key], axis=0)
        start = stop
        stop = start + batch_size
    stim = concat_stim[start:,:,:]
    for key in model_response.keys():
     	model_response[key] = np.append(model_response[key], temp[key], axis=0)
    return model_response




