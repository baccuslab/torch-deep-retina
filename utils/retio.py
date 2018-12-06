import os
import sys
import pickle
import torch

import numpy as np
import torch.nn as nn

def save_checkpoint(model,epoch, loss, optimizer, path, exp_id, state_dict=True):
    path = os.path.join(path,exp_id + '_epoch_' + str(epoch)) + '.pth'
    path = os.path.abspath(os.path.expanduser(path))
    torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss,
                'model': model
                }, path)
    print('Saved!')

def load_checkpoint(exp_id,epoch,folder,eval=True,return_ckpt=False):
    path = os.path.join(folder,exp_id + '_epoch_' + str(epoch))
    path = os.path.abspath(os.path.expanduser(path))

    checkpoint = torch.load(path)
    model = checkpoint['model']
    if return_ckpt:
        return model, checkpoint
    else:
        return model

