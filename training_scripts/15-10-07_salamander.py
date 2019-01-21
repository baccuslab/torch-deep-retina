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
from BN_CNN import BNCNN
import retio as io

from deepretina.experiments import loadexpt

# Constants
DEVICE = torch.device("cuda:0")


# Load data using Lane and Nirui's dataloader
train_data = loadexpt('15-10-07',[0,1,2,3,4],'naturalscene','train',40,0)
test_data = loadexpt('15-10-07',[0,1,2,3,4],'naturalscene','test',40,0)
val_split = 500

def main(model_class,epochs=200,batch_size=1000,LR=1e-4,l2_scale=0.01,l1_scale=0.01,shuffle=False):
    LAMBDA1 = l1_scale
    LAMBDA2 = l2_scale
    EPOCHS = epochs
    BATCH_SIZE = batch_size

    model = BNCNN()
    model = model.to(DEVICE)

    loss_fn = torch.nn.PoissonNLLLoss()
    optimizer = torch.optim.Adam(model.parameters(),lr = LR, weight_decay = LAMBDA2)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor = 0.2)


    # Train Loop
    for epoch in range(EPOCHS):
        epoch_tv_x = torch.from_numpy(train_data.X)
        epoch_tv_y = torch.from_numpy(train_data.y)

        if shuffle:
            print('shuffling data...')
            np.random.shuffle(epoch_tv_x)
            np.random.shuffle(epoch_train_y)
            print('data shuffled!')

        # train/val split
        val_ind = 500
        epoch_train_x = epoch_tv_x[val_ind:]
        epoch_val_x = epoch_tv_x[:val_ind]
        epoch_train_y = epoch_tv_y[val_ind:]
        epoch_val_y = epoch_tv_y[:val_ind]
        epoch_length = epoch_train_x.shape[0]
        num_batches,leftover = divmod(epoch_length, BATCH_SIZE)
        batch_size = BATCH_SIZE

        losses = []
        epoch_loss = 0
        print('Epoch ' + str(epoch))  
        
        test_x = torch.from_numpy(test_data.X)
        test_x = test_x.to(DEVICE)[:500]

        test_obs = model(test_x).cpu().detach().numpy()

        for cell in range(test_obs.shape[-1]):
            obs = test_obs[:500,cell]
            lab = test_data.y[:500,cell]
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

        #validate model
        val_obs = model(epoch_val_x.to(DEVICE)).cpu().detach().numpy()
        val_loss = np.sum([pearsonr(val_obs[:, i], epoch_val_y[:, i]) for i in range(epoch_val_y.shape[-1])])
        scheduler.step(val_loss)
        io.save_checkpoint(model,epoch,epoch_loss,optimizer,'~/','test')

if __name__ == "__main__":
    main(BNCNN)
