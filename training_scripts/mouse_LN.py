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
sys.path.append('../')
sys.path.append('../utils/')

from models.LN import LN
import retio as io
import argparse
import time
sys.path.append('/home/melander/deep-retina/')
from deepretina.experiments import loadexpt

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

# Load data using Lane and Niru's dataloader
train_data = loadexpt('19-02-26',[1],'naturalmovie','train',40,0)
print(train_data.y.shape)
import sys
sys.exit()

def train(epochs=250,batch_size=5000,LR=1e-3,l1_scale=1e-4,l2_scale=1e-2, shuffle=True, save='./checkpoints'):
    if not os.path.exists(save):
        os.mkdir(save)
    
    # I <3 definitions that are redundant
    LAMBDA1 = l1_scale
    LAMBDA2 = l2_scale
    EPOCHS = epochs
    BATCH_SIZE = batch_size

    # Model
    output_units = 1
    model = LN(output_units)
    model = model.to(DEVICE)


    # init the actual optimization machinery
    loss_fn = torch.nn.PoissonNLLLoss()
    optimizer = torch.optim.Adam(model.parameters(),lr = LR, weight_decay = LAMBDA2)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor = 0.2)
    


    # train data
    epoch_tv_x = torch.FloatTensor(train_data.X)
    epoch_tv_y = torch.FloatTensor(train_data.y)

    #train/val split
    val_split = 0.007
    num_val = int(epoch_tv_x.shape[0]*val_split)
    num_test = num_val / 2
    print('Validating on Beginning and End with {} Samples'.format(num_val*2))
    print('Saving {} Samples from Middle for test'.format(num_test))

    epoch_train_x = epoch_tv_x[(2*num_val):-num_val]
    epoch_val_x_beginning = epoch_tv_x[:num_val]
    epoch_val_x_end = epoch_tv_x[-num_val:]

    # WONT VALIDATE SO WE CAN TEST ON IT
    TEST_X = epoch_tv_x[num_val:(2*num_val)]
    TEST_Y = epoch_tv_y[num_val:(2*num_val)]
    
    epoch_train_y = epoch_tv_y[(2*num_val):-num_val]
    epoch_val_y_beginning = epoch_tv_y[:num_val]
    epoch_val_y_end = epoch_tv_y[-num_val:]

    
    # print some useful meta statistics
    epoch_length = epoch_train_x.shape[0]
    num_batches,leftover = divmod(epoch_length, BATCH_SIZE)
    batch_size = BATCH_SIZE
    print("Train size:", len(epoch_train_x))
    print("N Batches:", num_batches, "  Leftover:", leftover)

    # train loop
    for epoch in range(EPOCHS):
        losses = []
        epoch_loss = 0
        print('Epoch ' + str(epoch))  
       
        model.train(mode=True)
        
        if shuffle:
            indices = np.random.permutation(epoch_train_x.shape[0]).astype('int')
        else:
            indices = np.arange(0,epoch_train_x.shape[0]).astype('int')
            
        starttime = time.time()

        activity_l1 = torch.zeros(1).to(DEVICE)
        for batch in range(num_batches):
            optimizer.zero_grad()
            idxs = indices[batch_size*batch:batch_size*(batch+1)]

            x = torch.FloatTensor(epoch_train_x[idxs])
            label = torch.FloatTensor(epoch_train_y[idxs])
            label = label.to(DEVICE)

            y = model(x.to(DEVICE))

            if LAMBDA1 > 0:
                activity_l1 = LAMBDA1 * torch.norm(y, 1)

            error = loss_fn(y,label)
            loss = error + activity_l1
            loss.backward()

            optimizer.step()

            epoch_loss += loss.item()
            print("Loss:", loss.item()," - error:", error.item(), " - l1:", activity_l1.item(), " | ", int(round(batch/num_batches, 2)*100), "% done", end='               \r')
        print('\nAvg Loss: ' + str(epoch_loss/num_batches), " - exec time:", time.time() - starttime)

        #validate model
        del x
        del y
        del label
        print("SaveFolder:", save)
        scheduler.step(error)
        io.save_checkpoint(model,epoch,epoch_loss/num_batches,optimizer,save,'one_cell')
        
        model.eval()

        val_obs_1 = model(epoch_val_x_beginning.to(DEVICE)).cpu().detach().numpy()
        val_obs_2 = model(epoch_val_x_end.to(DEVICE)).cpu().detach().numpy()

        val_1_acc = pearsonr(epoch_val_y_beginning,val_obs_1)[0]
        val_2_acc = pearsonr(epoch_val_y_end,val_obs_2)[0]
        
        print('Beginning Val: {} End Val: {} CALCULATE YOUR OWN DAMNED AVERAGE'.format(val_1_acc,val_2_acc))

    return



if __name__ == "__main__":
    hyps = {}
    
    hyps['exp_name'] = 'mouse_linear_nonlinear'
    hyps['n_epochs'] = 15
    hyps['shuffle'] = True
    hyps['batch_size'] = 5000
    lrs = [1e-4,1e-5,1e-6]
    l1s = [0, 1e-2]
    l2s = [0, 1e-2]
    
    exp_num = 0

  
    for lr in lrs:
        hyps['lr'] = lr
        for l1 in l1s:
            hyps['l1'] = l1
            for l2 in l2s:
                hyps['l2'] = l2
                hyps['save_folder'] = hyps['exp_name'] +"_"+ str(exp_num) + "_lr"+str(lr) + "_" + "l1" + str(l1) + "_" + "l2" + str(l2)
                           
                train(hyps['n_epochs'], hyps['batch_size'], lr, l1, l2, hyps['shuffle'], hyps['save_folder'])
                exp_num += 1



