from scipy.stats import pearsonr
import os
import sys
from time import sleep
import numpy as np
import torch
import torch.nn as nn
import h5py as h5
import os.path as path
import sys
from torch.distributions import normal
import gc
import resource
sys.path.append('../models/')
sys.path.append('../utils/')

sys.path.append('/home/melander/first_mouse_deep_retina/torch-deep-retina/models/')
sys.path.append('/home/melander/first_mouse_deep_retina/deep-retina/deepretina/')

from experiments import loadexpt

from mouse_bn_cnn import BNCNN
import retio as io
import argparse
import time


# Helper function (used for memory leak debugging)
def cuda_if(tensor):
    if torch.cuda.is_available():
        return tensor.cuda()
    return tensor

# Constants
DEVICE = torch.device("cuda:0")

# Random Seeds (5 is arbitrary)
seed = 3
np.random.seed(seed)
torch.manual_seed(seed)

# Load data using Lane and Nirui's dataloader
train_data = loadexpt('19-02-26',[0,1],'naturalmovie','train',40,0)


def train(model_class,epochs=250,batch_size=5000,LR=1e-1,l2_scale=0.05,l1_scale=0.05, shuffle=True, save='./checkpoints', val_split=0.02,savename='savename'):
    if not os.path.exists(save):
        os.mkdir(save)
    LAMBDA1 = l1_scale
    LAMBDA2 = l2_scale
    EPOCHS = epochs
    BATCH_SIZE = batch_size

    model = BNCNN()
    model = model.to(DEVICE)

    loss_fn = torch.nn.PoissonNLLLoss()
    optimizer = torch.optim.Adam(model.parameters(),lr = LR, weight_decay = LAMBDA2)
   

    # train data
    epoch_tv_x = torch.FloatTensor(train_data.X)
    epoch_tv_y = torch.FloatTensor(train_data.y)

    # train/val split
    num_val = int(epoch_tv_x.shape[0]*val_split)
    epoch_train_x = epoch_tv_x[num_val:]
    epoch_val_x = epoch_tv_x[:num_val]
    epoch_train_y = epoch_tv_y[num_val:]
    epoch_val_y = epoch_tv_y[:num_val]
    epoch_length = epoch_train_x.shape[0]
    num_batches,leftover = divmod(epoch_length, BATCH_SIZE)
    batch_size = BATCH_SIZE
    print("Train size:", len(epoch_train_x))
    print("Val size:", len(epoch_val_x))


    # Train Loop
    for epoch in range(EPOCHS):
        if shuffle:
            indices = torch.randperm(epoch_train_x.shape[0]).long()
        else:
            indices = torch.arange(0, epoch_train_x.shape[0]).long()

        losses = []
        epoch_loss = 0
        print('Epoch ' + str(epoch))  
        
        model.eval()
        model.train(mode=True)

        
        starttime = time.time()
        for batch in range(num_batches):
            optimizer.zero_grad()
            idxs = indices[batch_size*batch:batch_size*(batch+1)]
            x = epoch_train_x[idxs]
            label = epoch_train_y[idxs]
            label = label.float()
            label = label.to(DEVICE)

            y = model(x.to(DEVICE))
            y = y.float() 

            activity_l1 = LAMBDA1 * torch.norm(y, 1).float()
            error = loss_fn(y,label)
            loss = error + activity_l1
            loss = loss_fn(y,label)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            print("Loss:", loss.item()," - error:", error.item(), " - l1:", activity_l1.item(), " | ", int(round(batch/num_batches, 2)*100), "% done", end='               \r')
        print('\nAvg Loss: ' + str(epoch_loss/num_batches), " - exec time:", time.time() - starttime)
    
        #validate model
        del x
        del y
        del label
        val_obs = model(epoch_val_x.to(DEVICE)).cpu().detach().numpy()
        val_acc = [pearsonr(val_obs[:, i], epoch_val_y[:, i]) for i in range(epoch_val_y.shape[-1])]
        print('Val Accuracy For One Cell: {}'.format(val_acc))

        io.save_checkpoint(model,epoch,epoch_loss/num_batches,optimizer,save,savename)
    return val_acc


if __name__ == '__main__':
    train(BNCNN,epochs=250,batch_size=5000,LR=1e-2,l2_scale=0.01,l1_scale=0.01, shuffle=True, save='./checkpoints', val_split=0.02,savename='onecell')
