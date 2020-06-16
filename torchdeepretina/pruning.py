import torch
import torch.nn as nn
import numpy as np
import torchdeepretina.utils as tdrutils
from torchdeepretina.custom_modules import LinearStackedConv2d,AbsBatchNorm2d, AbsBatchNorm1d

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
        for chan in chan_dict[layer]:
            idx = int(layer.split(".")[-1])
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
    return model

def get_next_layer(keys,zero_dict):
    if len(keys)==1: return keys[0]
    elif len(keys)==0: return None
    min_count = len(zero_dict[keys[0]])
    min_idx = 0
    for i,key in enumerate(keys[1:]):
        if len(zero_dict[key]) < min_count:
            min_idx = i+1
            min_count = len(zero_dict[key])
    return keys[min_idx]

def prune_channels(model, hyps, data_sample, zero_dict,
                                             prev_state_dict,
                                             cur_chan, val_acc,
                                             prev_acc, lr,
                                             prev_lr, cur_layer,
                                             min_acc, **kwargs):
    """
    Handles the channel pruning calculations. Should be called every
    n number of epochs just after validation during training.

    Inputs:
        model: torch Module
        hyps: dict
            keys: str
                open_layers: set of str
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
        data_sample: DataDistributor
            the training data distributor
        zero_dict: dict of sets
            keys: str
            vals: set of ints
                the keys should each be the string name of a layer
                and the values should be the corresponding channel
                indices that should be zeroed out. i.e.:
                "layer_name": {chan_idx_0, chan_idx_1, ...}
        prev_state_dict: torch Module state dict
            the state_dict of the last model state. The model is
            reverted back to this state dict if the most recent
            pruning choice performs worse
        cur_chan: int
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
        min_acc: float
            the minimum acc we will allow

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
    altn = hyps['altn_layers']
    tol = hyps["prune_tolerance"]
    open_layers = hyps['open_layers']
    if val_acc<(prev_acc-tol): # revert and close layer
        zero_dict[cur_layer].remove(cur_chan)
        open_layers.remove(cur_layer)
        model.load_state_dict(prev_state_dict)
    else:
        prev_acc = val_acc
        prev_lr = lr
    keys = sorted(list(open_layers),key=lambda x: int(x.split(".")[-1]))
    if altn:
        for i,k in enumerate(keys):
            idx = tdrutils.get_layer_idx(model,k)
            if len(zero_dict[k]) == (model.chans[idx]-1):
                if k in open_layers:
                    open_layers.remove(k)
        cur_layer = get_next_layer(keys,zero_dict)
    elif cur_layer not in open_layers and len(keys)>0:
        cur_layer = keys[0]
    if len(open_layers) == 0:
        return {"zero_dict":zero_dict,
                "prev_state_dict":prev_state_dict,
                "cur_chan":cur_chan,
                "cur_layer":cur_layer,
                "prev_acc":prev_acc,
                "prev_lr":prev_lr,
                "min_acc":min_acc}
    print("Calculating Integrated Gradient | Layer:", cur_layer)
    # Calc intg grad
    steps = hyps['alpha_steps']
    tdr_ig = tdrutils.integrated_gradient
    intg_grad, gc_resp = tdr_ig(model, data_sample, layer=cur_layer,
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
    cur_chan = min_chans[len(zero_dict[cur_layer])]
    prev_state_dict = model.state_dict()
    zero_dict[cur_layer].add(cur_chan)
    s = "Dropping channel {} in layer {}"
    print(s.format(cur_chan, cur_layer))

    return {"zero_dict":zero_dict,
            "prev_state_dict":prev_state_dict,
            "cur_chan":cur_chan,
            "cur_layer":cur_layer,
            "prev_acc":prev_acc,
            "prev_lr":prev_lr,
            "min_acc":min_acc }

#def prune_channels(model, hyps, data_distr, zero_dict, intg_idx,
#                                                prev_state_dict,
#                                                prev_min_chan,
#                                                val_acc, prev_acc,
#                                                lr, prev_lr,
#                                                min_acc, **kwargs):
#    """
#    Handles the channel pruning calculations. Should be called every
#    n number of epochs just after validation during training.
#
#    Inputs:
#        model: torch Module
#        hyps: dict
#            keys: str
#                prune_layers: set of str
#                    the names of the model layers that should be
#                    pruned. probably the convolutional layers.
#                intg_bsize: int
#                    size of batch for integrated gradient
#                alpha_steps: int
#                    the number of integration steps performed when
#                    calculating the integrated gradient
#                prune_tolerance: float
#                    the maximum drop in accuracy willing to be
#                    tolerated for a channel removal
#        data_distr: DataDistributor
#            the training data distributor
#        zero_dict: dict of sets
#            keys: str
#            vals: set of ints
#                the keys should each be the string name of a layer
#                and the values should be the corresponding channel
#                indices that should be zeroed out. i.e.:
#                "layer_name": {chan_idx_0, chan_idx_1, ...}
#        intg_idx: int
#            the index of the layer that should be focused on for
#            pruning
#        prev_state_dict: torch Module state dict
#            the state_dict of the last model state. The model is
#            reverted back to this state dict if the most recent
#            pruning choice performs worse
#        prev_min_chan: int
#            the last chan to be dropped
#        val_acc: float
#            the validation accuracy for the current model
#        prev_acc: float
#            the validation accuracy associated with the
#            prev_state_dict weights
#        lr: float
#            the current learning rate
#        prev_lr: float
#            the learning rate from before the last pruning
#        min_acc: float
#            the minimum acc we will allow
#
#    Returns:
#        dict:
#            stop_pruning: bool
#                if no more pruning can be completed, this is set to
#                true
#            zero_dict: same as input zero_dict
#            intg_idx: int
#                the updated layer index
#            prev_acc: float
#                the updated prev_acc. if new channel is pruned, this
#                is set to the val_acc, otherwise stays as is
#
#    """
#
#    new_drop_layer = False # Indicates we will move on to next layer
#    stop_pruning = False # Indicates we want to stop the pruning
#    prune_layers = hyps['prune_layers']
#    tolerance = hyps['prune_tolerance']
#
#    # If true, means we want to revert and move on to next layer
#    low_acc = (val_acc<prev_acc-tolerance) or (val_acc<min_acc)
#    if intg_idx<len(prune_layers) and low_acc:
#        print("Validation decrease detected. "+\
#                        "Returning to Previous Model")
#        layer = prune_layers[intg_idx]
#        zero_dict[layer].remove(prev_min_chan)
#        # Return weights to previous values
#        model.load_state_dict(prev_state_dict)
#        intg_idx += 1 # Indicates we want to focus on the next layer now
#        new_drop_layer = True
#    
#    # Only want to else if reached end of zeroable channels
#    if intg_idx<len(prune_layers):
#        print("Calculating Integrated Gradient | Layer:",
#                                  prune_layers[intg_idx])
#        # Calc intg grad
#        bsize = hyps['intg_bsize']
#        steps = hyps['alpha_steps']
#        gen = data_distr.train_sample(batch_size=bsize)
#        (data_sample, _) = next(gen)
#        del gen
#        layer = prune_layers[intg_idx]
#        tdr_ig = tdrutils.integrated_gradient
#        intg_grad, gc_resp = tdr_ig(model, data_sample, layer=layer,
#                                                    chans=None,
#                                                    spat_idx=None,
#                                                    alpha_steps=steps,
#                                                    batch_size=500,
#                                                    to_numpy=True,
#                                                    verbose=True)
#        shape = (*intg_grad.shape[:2],-1)
#        intg_grad = intg_grad.reshape(shape) #shape (T,C,N)
#        if hyps['abssum']:
#            print("Taking absolute value first,",
#                        "then summing over channels")
#            intg_grad = np.abs(intg_grad).sum(-1).mean(0) #shape (C,)
#        else:
#            print("Summing over channels first,",
#                        "then taking absolute value")
#            intg_grad = np.abs(intg_grad.sum(-1)).mean(0) #shape (C,)
#        min_chans = np.argsort(intg_grad)
#    
#        # Track changes
#        min_chan = min_chans[len(zero_dict[layer])]
#        layer_idx = int(layer.split(".")[-1])
#        prev_state_dict = model.state_dict()
#        zero_dict[layer].add(min_chan)
#        prev_min_chan = min_chan
#        s = "Dropping channel {} in layer {}"
#        print(s.format(min_chan, layer))
#    else:
#        print("No more layers in prune_layers list. "+\
#                                    "Stopping Training")
#        stop_pruning = True
#
#    # new_drop_layer means we have discontinued a pruning and wish
#    # to move on to the next possible layer for pruning. Thus, we
#    # want to revert our lr and acc to the values they were before
#    # we attempted the pruning if new_drop_layer is true. Otherwise
#    # we want to update them to the current values.
#    if not new_drop_layer: 
#        prev_acc = val_acc
#        prev_lr = lr
#
#    return {"stop_pruning":stop_pruning, "zero_dict":zero_dict,
#                                "prev_state_dict":prev_state_dict,
#                                "prev_min_chan":prev_min_chan,
#                                "intg_idx":intg_idx,
#                                "prev_acc":prev_acc,
#                                "prev_lr":prev_lr,
#                                "min_acc":min_acc}

def reduce_model(model, zero_dict):
    """
    This function removes the unused channels and disperses the bias
    into the appropriate biases if zero_bias is false. Also changes
    the zero_dict to reflect the new model architecture.

    model: torch Module
    zero_dict: dict
        keys: str
            layer names
        vals: set of ints
            the channels that have been pruned
    zero_bias: bool
        if true, convolutional biases are zeroed
    """
    n_pruned = np.sum([len(zero_dict[k]) for k in zero_dict.keys()])
    if n_pruned == 0:
        return model
    was_cuda = False
    if next(model.parameters()).is_cuda:
        was_cuda = True
    model = model.cpu()
    model = tdrutils.stacked2conv(model)
    conv_layers = tdrutils.get_conv_layer_names(model)
    for layer in conv_layers:
        if layer not in zero_dict: zero_dict[layer] = set()
    bn_types = {AbsBatchNorm2d, AbsBatchNorm1d,
                    torch.nn.BatchNorm2d, torch.nn.BatchNorm1d}
    bn_layers = tdrutils.get_layer_names(model, bn_types)
    convs = [tdrutils.get_module_by_name(model,c) for c in conv_layers]
    for conv in convs:
        if conv.bias is None:
            zeros = torch.zeros(conv.weight.shape[0])
            conv.bias = nn.Parameter(zeros)
    bnorms = [tdrutils.get_module_by_name(model,c) for c in bn_layers]
    for i in range(len(conv_layers)):
        conv = convs[i]
        bnorm = bnorms[i]
        # Currently only support for 2d batchnorm
        bnorm2d = isinstance(bnorm,AbsBatchNorm2d)
        bnorm2d = bnorm2d or isinstance(bnorm, torch.nn.BatchNorm2d)
        assert (i == len(conv_layers)-1) or bnorm2d
        absbn = isinstance(bnorm, AbsBatchNorm2d)
        absbn = absbn or isinstance(bnorm, AbsBatchNorm1d)

        old_shape = conv.weight.shape
        old_weight = conv.weight.data
        old_bias = conv.bias.data
        if absbn:
            old_scale = bnorm.scale.data
            old_shift = bnorm.shift.data
            if bnorm.abs_bias: old_shift = old_shift.abs()
        else:
            old_scale = bnorm.weight.data
            old_shift = bnorm.bias.data
        old_mean = bnorm.running_mean.data
        old_var = bnorm.running_var.data

        in_chans = old_shape[1] # Only applies in zeroth layer
        keep_idxs = range(old_shape[1])
        if i > 0:
            in_chans = out_chans
            func = lambda x: x not in zero_dict[conv_layers[i-1]]
            keep_idxs = list(filter(func, range(old_shape[1])))

        out_chans = old_shape[0]-len(zero_dict[conv_layers[i]])

        new_shape = (out_chans,in_chans,*old_shape[2:])
        new_weight = torch.zeros(new_shape)
        new_bias = torch.zeros(new_shape[0])
        new_scale = torch.zeros(new_shape[0])
        new_shift = torch.zeros(new_shape[0])
        new_mean = torch.zeros(new_shape[0])
        new_var = torch.zeros(new_shape[0])
        new_j = 0
        for j in range(old_shape[0]):
            if j not in zero_dict[conv_layers[i]]:
                new_weight[new_j] = old_weight[j,keep_idxs]
                new_bias[new_j] = old_bias[j]
                new_scale[new_j] = old_scale[j]
                new_shift[new_j] = old_shift[j]
                new_mean[new_j] = old_mean[j]
                new_var[new_j] = old_var[j]
                new_j+=1
            elif i < len(conv_layers)-1:
                scale = old_scale.data[j]
                shift = old_shift.data[j]
                mean = bnorm.running_mean.data[j]
                var = bnorm.running_var.data[j]
                eps = bnorm.eps
                bias = (old_bias[j]-mean)/torch.sqrt(var+eps)
                bias = bias*scale + shift
                weight = convs[i+1].weight.data
                for o in range(weight.shape[0]):
                    temp = torch.sum(weight[o,j]*bias)
                    convs[i+1].bias.data[o] += temp
        assert new_j == new_shape[0]
        convs[i].weight.data = new_weight
        convs[i].bias.data = new_bias
        convs[i].in_channels = in_chans
        convs[i].out_channels = out_chans
        if absbn:
            bnorms[i].scale.data = new_scale
            bnorms[i].shift.data = new_shift
        else:
            bnorms[i].weight.data = new_scale
            bnorms[i].bias.data = new_shift
        bnorms[i].running_mean.data = new_mean
        bnorms[i].running_var.data = new_var
    model.chans = [c.weight.data.shape[0] for c in convs]
    if was_cuda: model.to(DEVICE)
    return model


