import pickle
import numpy as np
import torch
import torch.nn as nn
import h5py as h5
import os.path as path
import sys

from deepretina.experiments import loadexpt

# Contsants
DEBUG = False
DEVICE = torch.device("cuda:0")
EPOCHS = 200
BATCH_SIZE = 64

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

 # Train definitions
model = McNiruNet()
model = model.to(DEVICE)
loss_fn = torch.nn.PoissonNLLLoss()
optimizer = torch.optim.Adam(model.parameters(),lr = 0.01)


# Train Loop
for epoch in range(EPOCHS):
    epoch_train_x = torch.from_numpy(train_data.X)
    epoch_train_y = torch.from_numpy(train_data.y)

    print('Shuffling data...')
    np.random.shuffle(epoch_train_x)
    np.random.shuffle(epoch_train_y)

    print('Shuffled')
    epoch_length = epoch_train_x.shape[0]
    num_batches,leftover = divmod(epoch_length, BATCH_SIZE)
    batch_size = BATCH_SIZE

    losses = []

    for batch in range(num_batches):
        sys.stdout.write('\n')
        sys.stdout.write('Batch: %s / %s' % (batch, num_batches))
        sys.stdout.flush()
         
        x = epoch_train_x[batch_size*batch:(batch_size*batch+1),:,:,:]
        label = epoch_train_y[batch_size*batch:(batch_size*batch+1),:]
        label = label.double()
        label = label.to(DEVICE)

        x = x.to(DEVICE)
        y = model(x)
        y = y.double() 
        loss = loss_fn(y,label)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print("Loss: ")
        print(loss)
        losses.append(loss)
        
    savestr = str(epoch) + '_' + '1e2' + model.name + '.pickle'
    o = open(savestr,'wb')
    pickle.dump(losses,o)
    o.close()
    
model = []

model = McNiruNet()
model = model.to(DEVICE)
loss_fn = torch.nn.PoissonNLLLoss()
optimizer = torch.optim.Adam(model.parameters(),lr = 0.0001)


# Train Loop
for epoch in range(EPOCHS):
    epoch_train_x = torch.from_numpy(train_data.X)
    epoch_train_y = torch.from_numpy(train_data.y)

    print('Shuffling data...')
    np.random.shuffle(epoch_train_x)
    np.random.shuffle(epoch_train_y)

    print('Shuffled')
    epoch_length = epoch_train_x.shape[0]
    num_batches,leftover = divmod(epoch_length, BATCH_SIZE)
    batch_size = BATCH_SIZE

    losses = []

    for batch in range(num_batches):
        sys.stdout.write('\n')
        sys.stdout.write('Batch: %s / %s' % (batch, num_batches))
        sys.stdout.flush()
         
        x = epoch_train_x[batch_size*batch:(batch_size*batch+1),:,:,:]
        label = epoch_train_y[batch_size*batch:(batch_size*batch+1),:]
        label = label.double()
        label = label.to(DEVICE)

        x = x.to(DEVICE)
        y = model(x)
        y = y.double() 
        loss = loss_fn(y,label)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print("Loss: ")
        print(loss)
        losses.append(loss)
        
    savestr = str(epoch) + '_' + '1e2' + model.name + '.pickle'
    o = open(savestr,'wb')
    pickle.dump(losses,o)
    o.close()
