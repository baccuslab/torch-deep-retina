# 1. Imports
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
from models.EltonCNN import EltonCNN
import retio as io
import argparse
import time
from load_eltons_data import get_eltons_data

# 2. Check if GPUs available
# Helper function (used for memory leak debugging)
def cuda_if(tensor):
    if torch.cuda.is_available():
        return tensor.cuda()
    return tensor

# 3. Define run device. 
# If you want to run on CPU, torch.device("cpu") 
# Constants
DEVICE = torch.device("cuda:0")

# Random Seeds (5 is arbitrary)
seed = 3
np.random.seed(seed)
torch.manual_seed(seed)

# 4. Load elton's data
(original_x,original_y) = get_eltons_data('/home/melander/electrical_deep_retina/electricalONCells.mat','/home/melander/electrical_deep_retina/WNMovie.csv',history=20)


def train(epochs=250,batch_size=512,LR=1e-3,l1_scale=1e-4,l2_scale=1e-2, shuffle=True, save='./checkpoints'):
    if not os.path.exists(save):
        os.mkdir(save)
    
    # I <3 definitions that are redundant
    LAMBDA1 = l1_scale
    LAMBDA2 = l2_scale
    EPOCHS = epochs
    BATCH_SIZE = batch_size

    # 5. Instantiate our model
    model = EltonCNN(n_output_units = 13, history = 20)
    model = model.to(DEVICE)


    # init the actual optimization machinery
    loss_fn = torch.nn.PoissonNLLLoss()
    optimizer = torch.optim.Adam(model.parameters(),lr = LR, weight_decay = LAMBDA2)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor = 0.2)
    
    # train data
    epoch_tv_x = torch.FloatTensor(original_x)
    epoch_tv_y = torch.FloatTensor(original_y)

    #train/val split
    val_split = 0.01
    num_val = int(epoch_tv_x.shape[0]*val_split)
    print('Validating on Beginning {} Samples'.format(num_val))
    
    # Training data set and val dataset
    epoch_train_x = epoch_tv_x[num_val:]
    epoch_val_x = epoch_tv_x[:num_val]
    
   
    epoch_train_y = epoch_tv_y[num_val:]
    epoch_val_y = epoch_tv_y[:num_val]
   

    
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

            # Getting the actual batch data from what we defineder earlier
            
            x = torch.FloatTensor(epoch_train_x[idxs])
            
            label = torch.FloatTensor(epoch_train_y[idxs])
            label = label.to(DEVICE)

            # Run your model
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
        io.save_checkpoint(model,epoch,epoch_loss/num_batches,optimizer,save,'train')
        
        model.eval()

        val_obs = model(epoch_val_x.to(DEVICE)).cpu().detach().numpy()
    
      
        epoch_val_y_cpu = epoch_val_y.cpu().detach().numpy()
        
        print(type(epoch_val_y_cpu))
        print(type(val_obs))
        
        print(val_obs.shape)
        print(epoch_val_y_cpu.shape)
        
        for c in range(13):
            print('Cell {} R: {}'.format(c,pearsonr(epoch_val_y_cpu[:,c],val_obs[:,c])))
        #val_acc = pearsonr(epoch_val_y,val_obs)
    
        
        #print('Validation Accuracy: {}'.format(val_acc))

    return



if __name__ == "__main__":
    hyps = {}
    
    hyps['exp_name'] = 'eltons_data'
    hyps['n_epochs'] = 15
    hyps['shuffle'] = True
    hyps['batch_size'] = 512
    lrs = [1e-3,1e-4,1e-5]
    l1s = [0,1e-2]
    l2s = [0,1e-1]
    
    exp_num = 1

  
    for lr in lrs:
        hyps['lr'] = lr
        for l1 in l1s:
            hyps['l1'] = l1
            for l2 in l2s:
                hyps['l2'] = l2
                hyps['save_folder'] = hyps['exp_name'] +"_"+ str(exp_num) + "_lr"+str(lr) + "_" + "l1" + str(l1) + "_" + "l2" + str(l2)
                           
                train(hyps['n_epochs'], hyps['batch_size'], lr, l1, l2, hyps['shuffle'], hyps['save_folder'])
                exp_num += 1



