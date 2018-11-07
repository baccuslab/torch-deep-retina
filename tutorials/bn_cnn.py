import pickle
import numpy as np
import torch
import torch.nn as nn
import h5py as h5
import os.path as path
import sys
from torch.distributions import normal
from deepretina.experiments import loadexpt

# Contsants
DEBUG = False
DEVICE = torch.device("cuda:0")
EPOCHS = 200
BATCH_SIZE = 65

# Load data using Lane and Nirui's dataloader
train_data = loadexpt('15-10-07',[0,1,2,3,4],'naturalscene','train',40,0)
test_data = loadexpt('15-10-07',[0,1,2,3,4],'naturalscene','test',40,0)

# Model definitions
class BN_CNN_Net(nn.Module):
    def __init__(self):
        super(BN_CNN_Net,self).__init__()
        self.name = 'BN_CNN_Net'
        self.conv1 = nn.Conv2d(40,8,kernel_size=15)
        self.batch1 = nn.BatchNorm2d(8)
        self.conv2 = nn.Conv2d(8,8,kernel_size=9)
        self.batch2 = nn.BatchNorm2d(8)
        self.linear = nn.Linear(8*28*28,5)
        self.batch3 = nn.BatchNorm1d(5)
        self.losses = [];
        
    def gaussian(self, x, is_training):
        if is_training:
            noise = torch.autograd.Variable(x.data.new(x.size()).normal_(0, 0.05))
            return x + noise
        return ins
        

    def forward(self,x):
        x = nn.functional.relu(self.gaussian(self.conv1(x), True))
        x = self.batch1(x)
        x = nn.functional.relu(self.gaussian(self.conv2(x), True))
        x = self.batch2(x)
        x = x.view(-1,8*28*28)

        x = nn.functional.softplus(self.batch3(self.linear(x)))
        return x

    def inspect(self, x):
        model_dict = {}
        model_dict['stimulus'] = x;
        model_dict['conv2d_1'] = self.conv1(x)
        model_dict['gaussian_noise_1'] = self.gaussian(x, False)
        model_dict['batch_normalization_1'] = self.batch1(x)
        model_dict['activation_1'] = nn.functional.relu(x)
        model_dict['conv2d_2'] = self.conv2(x)
        model_dict['gaussian_noise_2'] = self.gaussian(x, False)
        model_dict['batch_normalization_2'] = self.batch2(x)
        model_dict['activation_2'] = nn.functional.relu(x)
        model_dict['dense_1'] = self.linear(x.view(-1, 8*28*28))
        model_dict['batch_normalization_3'] = self.batch3(x)
        model_dict['activation_3'] = nn.functional.softplus(x)
        return model_dict
    
model = BN_CNN_Net()
model = model.to(DEVICE)
loss_fn = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(),lr = 0.001, weight_decay = 0.01)


# Train Loop
for epoch in range(EPOCHS):
    epoch_train_x = torch.from_numpy(train_data.X)
    epoch_train_y = torch.from_numpy(train_data.y)

    print('Shuffling data...')
    np.random.shuffle(epoch_train_x)
    np.random.shuffle(epoch_train_y)

    print('Shuffled')
    epoch_length = epoch_train_x.shape[0]
    print("Epoch length = {0}".format(epoch_length))
    num_batches,leftover = divmod(epoch_length, BATCH_SIZE)
    batch_size = BATCH_SIZE

    epoch_loss = 0
    print('Starting new batch')

    for batch in range(num_batches):
         
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
    print('Avg Epoch Loss')
    print(epoch_loss / num_batches)
    model.losses.append(epoch_loss/num_batches)
    if epoch_loss < 0.1: break
print(model.losses)
with open("bn_cnn_15_10_07", "wb") as fd:
    pickle.dump(model, fd)