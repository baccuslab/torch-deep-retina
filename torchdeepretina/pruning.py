import torch
import numpy as np
import torchdeepretina.utils as tdrutils
from torchdeepretina.custom_modules import LinearStackedConv2d

if torch.cuda.is_available():
    DEVICE = torch.device('cuda:0')
else:
    DEVICE = torch.device('cpu')

def zero_chans(model, chan_dict, zero_bias=True):
    """
    Zeros out the model channels specified in the chan_dict.

    model: nn.Module
        the deep learning model 
    chan_dict: dict (str, set)
        keys: layer name as string
        vals: set of ints corresponding to the channels to drop
    zero_bias: bool
        if true, bias is zeroed as well as channel
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
                if zero_bias and hasattr(modu, 'bias'):
                    model.sequential[idx].convs[-1].bias.data[chan] = 0
            else:
                modu = model.sequential[idx]
                shape = modu.weight.data[chan].shape
                model.sequential[idx].weight.data[chan] =\
                                   torch.zeros(*shape[1:]).to(DEVICE)
                if zero_bias and hasattr(modu, 'bias'):
                    model.sequential[idx].bias.data[chan] = 0

def prune_channels(model, hyps, data_distr, zero_dict, intg_idx,
                                                prev_state_dict,
                                                prev_min_chan,
                                                val_acc, prev_acc,
                                                lr, prev_lr,
                                                reset_sd=None,
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
        lr: float
            the current learning rate
        prev_lr: float
            the learning rate from before the last pruning
        reset_sd: None or torch State Dict
            If None, will have no effect. If state dict is argued,
            the model will be reset to this state dict after the
            next channel to be pruned is decided.

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
    new_drop_layer = False # Indicates we will move on to next layer
    stop_pruning = False # Indicates we want to stop the pruning
    prune_layers = hyps['prune_layers']
    tolerance = hyps['prune_tolerance']

    # If true, means we want to revert and move on to next layer
    if intg_idx<len(prune_layers) and val_acc<prev_acc-tolerance:
        print("Validation decrease detected. "+\
                        "Returning to Previous Model")
        layer = prune_layers[intg_idx]
        zero_dict[layer].remove(prev_min_chan)
        # Return weights to previous values
        model.load_state_dict(prev_state_dict)
        intg_idx += 1
        new_drop_layer = True
    
    # Only want to else if reached end of zeroable channels
    drop_layer = (val_acc>=prev_acc-tolerance or new_drop_layer)
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
                                                    chans=None,
                                                    spat_idx=None,
                                                    alpha_steps=steps,
                                                    batch_size=500,
                                                    to_numpy=True,
                                                    verbose=True)
        shape = (*intg_grad.shape[:2],-1)
        intg_grad = intg_grad.reshape(shape) #shape (T,C,N)
        if hyps['abssum']:
            print("Taking absolute value first,",
                        "then summing over channels")
            intg_grad = np.abs(intg_grad).sum(-1).mean(0) #shape (C,)
        else:
            print("Summing over channels first,",
                        "then taking absolute value")
            intg_grad = np.abs(intg_grad.sum(-1)).mean(0) #shape (C,)
        min_chans = np.argsort(intg_grad)
    
        # Track changes
        min_chan = min_chans[len(zero_dict[layer])]
        layer_idx = int(layer.split(".")[-1])
        prev_state_dict = model.state_dict()
        zero_dict[layer].add(min_chan)
        prev_min_chan = min_chan
        s = "Dropping channel {} in layer {}"
        print(s.format(min_chan, layer))
        if reset_sd is not None:
            model.load_state_dict(reset_sd)
    else:
        print("No more layers in prune_layers list. "+\
                                    "Stopping Training")
        stop_pruning = True

    # new_drop_layer means we have discontinued a pruning and wish
    # to move on to the next possible layer for pruning. Thus, we
    # want to revert our lr and acc to the values they were before
    # we attempted the pruning if new_drop_layer is true. Otherwise
    # we want to update them to the current values.
    if not new_drop_layer: 
        prev_acc = val_acc
        prev_lr = lr

    return {"stop_pruning":stop_pruning, "zero_dict":zero_dict,
                                "prev_state_dict":prev_state_dict,
                                "prev_min_chan":prev_min_chan,
                                "intg_idx":intg_idx,
                                "prev_acc":prev_acc,
                                "prev_lr":prev_lr}
