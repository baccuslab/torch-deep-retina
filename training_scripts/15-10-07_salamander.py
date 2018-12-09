from tqdm import tqdm
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
sys.path.append('../models/')
sys.path.append('../utils/')
from lab import BN_CNN as BNCNN
import retio as io

from deepretina.experiments import loadexpt

# Constants
DEVICE = torch.device("cuda:0")


# Load data using Lane and Nirui's dataloader
train_data = loadexpt('15-10-07',[0,1,2,3,4],'naturalscene','train',40,0)
test_data = loadexpt('15-10-07',[0,1,2,3,4],'naturalscene','test',40,0)

def main(model_class,epochs=200,batch_size=1000,LR=1e-4,l2_scale=0.01,l1_scale=0.01,shuffle=False):
    LAMBDA1 = l1_scale
    LAMBDA2 = l2_scale
    EPOCHS = epochs
    BATCH_SIZE = batch_size

    model = BNCNN()
    model = model.to(DEVICE)

    loss_fn = torch.nn.PoissonNLLLoss()
    optimizer = torch.optim.Adam(model.parameters(),lr = LR, weight_decay = LAMBDA2)

    # Train Loop
    for epoch in range(EPOCHS):
        epoch_train_x = torch.from_numpy(train_data.X)
        epoch_train_y = torch.from_numpy(train_data.y)

        if shuffle:
            print('shuffling data...')
            np.random.shuffle(epoch_train_x)
            np.random.shuffle(epoch_train_y)
            print('data shuffled!')

        epoch_length = epoch_train_x.shape[0]
        num_batches,leftover = divmod(epoch_length, BATCH_SIZE)
        batch_size = BATCH_SIZE

        losses = []
        epoch_loss = 0
        print('Epoch ' + str(epoch))  
        
        test_x = torch.from_numpy(test_data.X)
        test_x = test_x.to(DEVICE)

        test_obs = model(test_x)
        test_obs = test_obs.cpu()
        test_obs = test_obs.detach().numpy()

        for cell in range(test_obs.shape[-1]):
            obs = test_obs[:,cell]
            lab = test_data.y[:,cell]
            r,p = pearsonr(obs,lab)
            print('Cell ' + str(cell) + ': ')
            print('-----> pearsonr: ' + str(r))

        for batch in tqdm(range(num_batches),ascii=True,desc='batches'):
            x = epoch_train_x[batch_size*batch:batch_size*(batch+1),:,:,:]
            label = epoch_train_y[batch_size*batch:batch_size*(batch+1),:]
            label = label.double()
            label = label.to(DEVICE)

            x = x.to(DEVICE)
            y = model(x)
            y = y.double() 

            all_linear1_params = torch.cat([x.view(-1) for x in model.linear.parameters()])
            loss = loss_fn(y,label) + LAMBDA1 * torch.norm(all_linear1_params, 1).double()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss
        print('Loss: ' + str(epoch_loss/num_batches))
        io.save_checkpoint(model,epoch,loss,optimizer,'~/','test')

if __name__ == "__main__":
    main(BNCNN)
