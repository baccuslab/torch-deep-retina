import os
import sys
import pickle
import torch

import numpy as np
import torch.nn as nn

def save_checkpoint_dict(save_dict, path, exp_id, del_prev=False):
    if del_prev:
        prev_path = os.path.join(path, exp_id + "_epoch_" + str(save_dict['epoch']-1) + '.pth')
        if os.path.exists(prev_path):
            data = torch.load(prev_path)
            keys = list(data.keys())
            for key in keys:
                if "state_dict" in key:
                    del data[key]
    path = os.path.join(path,exp_id + '_epoch_' + str(save_dict['epoch'])) + '.pth'
    path = os.path.abspath(os.path.expanduser(path))
    torch.save(save_dict, path)

def load_checkpoint(exp_id,epoch,folder,eval=True,return_ckpt=False):
    path = os.path.join(folder,exp_id + '_epoch_' + str(epoch))+'.pth'
    path = os.path.abspath(os.path.expanduser(path))

    checkpoint = torch.load(path)
    model = checkpoint['model']
    model.load_state_dict(checkpoint['model_state_dict'])
    if return_ckpt:
        return model, checkpoint
    else:
        return model

