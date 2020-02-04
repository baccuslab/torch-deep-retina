import torch
import numpy as np
import torchdeepretina.utils as tdrutils
from torchdeepretina.custom_modules import LinearStackedConv2d

if torch.cuda.is_available():
    DEVICE = torch.device('cuda:0')
else:
    DEVICE = torch.device('cpu')

def zero_chans(model, chan_dict):
    """
    Zeros out the model channels specified in the chan_dict.

    model: nn.Module
        the deep learning model 
    chan_dict: dict (str, set)
        keys: layer name as string
        vals: set of ints corresponding to the channels to drop
    """
    for layer in chan_dict.keys():
        # Assumes layer name is of form 'sequential.<idx>'
        idx = int(layer.split(".")[-1])
        for chan in chan_dict[layer]:
            if isinstance(model.sequential[idx],LinearStackedConv2d):
                modu = model.sequential[idx].convs[-1]
                shape = modu.weight.data[chan].shape
                model.sequential[idx].convs[-1].weight.data[chan] =\
                                  torch.zeros(*shape[1:]).to(DEVICE)
            else:
                shape = model.sequential[idx].weight.data[chan].shape
                model.sequential[idx].weight.data[chan] =\
                                    torch.zeros(*shape[1:]).to(DEVICE)

def prune_channels(model, hyps, data_distr, zero_dict, intg_idx,
                                                prev_state_dict,
                                                prev_min_chan,
                                                val_acc, prev_acc,
                                                **kwargs):
    """
    Handles the channel pruning calculations. Should be called every
    n number of epochs just after validation during training.

    Inputs:
        model: torch Module
        hyps: dict
            keys: str
                prune_layers: set of str
                    the names of the model layers that should be
                    pruned. probably the convolutional layers.
                intg_bsize: int
                    size of batch for integrated gradient
                alpha_steps: int
                    the number of integration steps performed when
                    calculating the integrated gradient
                prune_tolerance: float
                    the maximum drop in accuracy willing to be
                    tolerated for a channel removal
        data_distr: DataDistributor
            the training data distributor
        zero_dict: dict of sets
            keys: str
            vals: set of ints
                the keys should each be the string name of a layer
                and the values should be the corresponding channel
                indices that should be zeroed out. i.e.:
                "layer_name": {chan_idx_0, chan_idx_1, ...}
        intg_idx: int
            the index of the layer that should be focused on for
            pruning
        prev_state_dict: torch Module state dict
            the state_dict of the last model state. The model is
            reverted back to this state dict if the most recent
            pruning choice performs worse
        prev_min_chan: int
            the last chan to be dropped
        val_acc: float
            the validation accuracy for the current model
        prev_acc: float
            the validation accuracy associated with the
            prev_state_dict weights
    Returns:
        dict:
            stop_pruning: bool
                if no more pruning can be completed, this is set to
                true
            zero_dict: same as input zero_dict
            intg_idx: int
                the updated layer index
            prev_acc: float
                the updated prev_acc. if new channel is pruned, this
                is set to the val_acc, otherwise stays as is

    """
    new_drop_layer = False
    stop_pruning = False
    prune_layers = hyps['prune_layers']
    tolerance = hyps['prune_tolerance']
    if intg_idx<len(prune_layers) and val_acc<prev_acc-tolerance:
        print("Validation decrease detected. "+\
                        "Returning to Previous Model")
        layer = prune_layers[intg_idx]
        zero_dict[layer].remove(prev_min_chan)
        # Return weights to previous values
        model.load_state_dict(prev_state_dict)
        intg_idx += 1
        new_drop_layer = True
    
    drop_layer = (val_acc>=prev_acc or new_drop_layer)
    if intg_idx<len(prune_layers) and drop_layer:
        print("Calculating Integrated Gradient | Layer:",
                            prune_layers[intg_idx])
        # Calc intg grad
        bsize = hyps['intg_bsize']
        steps = hyps['alpha_steps']
        gen = data_distr.train_sample(batch_size=bsize)
        (data_sample, _) = next(gen)
        del gen
        layer = prune_layers[intg_idx]
        tdr_ig = tdrutils.integrated_gradient
        intg_grad, gc_resp = tdr_ig(model, data_sample, layer=layer,
                                                    gc_idx=None,
                                                    alpha_steps=steps,
                                                    batch_size=500,
                                                    to_numpy=True,
                                                    verbose=True)
        shape = (*intg_grad.shape[:2],-1)
        intg_grad = intg_grad.reshape(shape)
        intg_grad = intg_grad.mean(-1).mean(0) #shape (C,)
        min_chans = np.argsort(np.abs(intg_grad))
    
        # Track changes
        min_chan = min_chans[len(zero_dict[layer])]
        layer_idx = int(layer.split(".")[-1])
        prev_state_dict = model.state_dict()
        zero_dict[layer].add(min_chan)
        prev_min_chan = min_chan
        s = "Dropping channel {} in layer {}"
        print(s.format(min_chan, layer))
    else:
        print("No more layers in prune_layers list. "+\
                                    "Stopping Training")
        stop_pruning = True
    # If ensures we do not use val_acc from discontinued model
    if not new_drop_layer: 
        prev_acc = val_acc

    return {"stop_pruning":stop_pruning, "zero_dict":zero_dict,
                                "prev_state_dict":prev_state_dict,
                                "prev_min_chan":prev_min_chan,
                                "intg_idx":intg_idx,
                                "prev_acc":prev_acc}
