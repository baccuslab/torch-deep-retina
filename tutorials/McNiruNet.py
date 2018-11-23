from tqdm import tqdm
from scipy.stats import pearsonr
import sys
from time import sleep
sys.path.append('../')
import metrics
import pickle
import numpy as np
import torch
import torch.nn as nn
import h5py as h5
import os.path as path
import sys

from deepretina.experiments import loadexpt

# Constants
DEVICE = torch.device("cuda:0")

# Hyperparams
EPOCHS = 200
BATCH_SIZE = 1000
LR = 1e-4


# Load data using Lane and Nirui's dataloader
train_data = loadexpt('15-10-07',[0,1,2,3,4],'naturalscene','train',40,0)
test_data = loadexpt('15-10-07',[0,1,2,3,4],'naturalscene','test',40,0)

# Model definitions
class McNiruNet(nn.Module):
    def __init__(self):
        super(McNiruNet,self).__init__()
        self.name = 'McNiruNet'
        self.conv1 = nn.Conv2d(40,8,kernel_size=15)
        self.conv2 = nn.Conv2d(8,8,kernel_size=9)
        self.linear = nn.Linear(8*28*28,5)

        

    def forward(self,x):
        x = nn.functional.relu(self.conv1(x))
        x = nn.functional.relu(self.conv2(x))
        x = x.view(-1,8*28*28)
        x = nn.functional.softplus(self.linear(x.view(-1,8*28*28)))
        return x
    
model = McNiruNet()
model = model.to(DEVICE)
loss_fn = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(),lr = LR)


# Train Loop
for epoch in range(EPOCHS):
    epoch_train_x = torch.from_numpy(train_data.X)
    epoch_train_y = torch.from_numpy(train_data.y)

    #print('Shuffling data...')
    #np.random.shuffle(epoch_train_x)
    #np.random.shuffle(epoch_train_y)

    print('Shuffled')
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

        loss = loss_fn(y,label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_loss += loss

    print('Loss: ' + str(epoch_loss/num_batches))
