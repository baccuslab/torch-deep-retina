import sys
sys.path.append('../')
import pickle
import numpy as np
import torch
import torch.nn as nn
import h5py as h5
import os
import sys
from torch.distributions import normal
from deepretina.experiments import loadexpt
import metrics

# Contsants
DEBUG = False
DEVICE = torch.device("cuda:0")
EPOCHS = 200
BATCH_SIZE = 65

# Load data using Lane and Niru's dataloader
train_data = loadexpt('15-10-07',[0,1,2,3,4],'naturalscene','train',40,0)
test_data = loadexpt('15-10-07',[0,1,2,3,4],'naturalscene','test',40,0)

# Model definitions
class BN_CNN_Net(nn.Module):
    def __init__(self):
        super(BN_CNN_Net,self).__init__()
        self.name = 'BN_CNN_Net'
        self.conv1 = nn.Conv2d(40,8,kernel_size=15)
        self.batch1 = nn.BatchNorm1d(8*36*36)
        self.conv2 = nn.Conv2d(8,8,kernel_size=11)
        self.batch2 = nn.BatchNorm1d(8*26*26)
        self.linear = nn.Linear(8*26*26,5, bias=False)
        self.batch3 = nn.BatchNorm1d(5)
        self.losses = [];
        self.metrics = {'mse': metrics.mean_squared_error, 'cc':metrics.correlation_coefficient, 'var':metrics.fraction_of_explained_variance}
        
    def gaussian(self, x, sigma):

        noise = normal.Normal(torch.zeros(x.size()), sigma*torch.ones(x.size()))
        return x + noise.sample().cuda()
        

    def forward(self,x):
        x = self.conv1(x)
        x = self.batch1(x.view(x.size(0), -1))
        x = nn.functional.relu(self.gaussian(x.view(-1, 8, 36, 36), 0.05))
        x = self.conv2(x)
        x = self.batch2(x.view(x.size(0), -1))
        x = nn.functional.relu(self.gaussian(x.view(-1, 8, 26, 26), 0.05))
        x = self.linear(x.view(-1, 8*26*26))
        x = self.batch3(x)
        x = nn.functional.softplus(x)
        return x

    def inspect(self, x):
        model_dict = {}
        model_dict['stimulus'] = x
        x = self.conv1(x);
        model_dict['conv2d_1'] = x
        x = x.view(x.size(0), -1)
        model_dict['flatten_1'] = x
        x = self.batch1(x)
        model_dict['batch_normalization_1'] = x
        x = self.gaussian(x.view(-1, 8, 36, 36), 0.05)
        model_dict['gaussian_1'] = x
        x = nn.functional.relu(x)
        model_dict['activation_1'] = x
        x = self.conv2(x);
        model_dict['conv2d_2'] = x
        x = x.view(x.size(0), -1)
        model_dict['flatten_2'] = x
        x = self.batch2(x)
        model_dict['batch_normalization_2'] = x
        x = self.gaussian(x.view(-1, 8, 26, 26), 0.05)
        model_dict['gaussian_2'] = x
        x = nn.functional.relu(x)
        model_dict['activation_2'] = x
        x = self.linear(x.view(-1, 8*26*26))
        model_dict['dense'] = x
        x = self.batch3(x)
        model_dict['batch_normalization_3'] = x
        x = nn.functional.softplus(x)
        model_dict['activation_3'] = x
        return model_dict

    
def main():
    train_folder = 'Trained_11_13_18'
    if not os.path.exists(train_folder):
        mkdir(train_folder)
    model = BN_CNN_Net()
    model = model.to(DEVICE)
    model.train()
    loss_fn = torch.nn.PoissonNLLLoss()
    optimizer = torch.optim.Adam(model.parameters(),lr = 0.001, weight_decay = 0.01)


    # Train Loop
    train_metrics = {};
    for metric in model.metrics:
        train_metrics[metric] = [];
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
        print('Starting new batch in epoch {0}'.format(epoch))
        summ = [];
        for batch in range(num_batches):
             
            x = torch.autograd.Variable(epoch_train_x[batch_size*batch:batch_size*(batch+1),:,:,:], requires_grad = True)
            label = torch.autograd.Variable(epoch_train_y[batch_size*batch:batch_size*(batch+1),:])
            label = label.to(DEVICE)
            label = label.double()

            x = x.to(DEVICE)
            x.retain_grad()
            y = model(x)
            y = y.double()
            loss = loss_fn(y,label)
            optimizer.zero_grad()
            loss.backward()
            print(x.grad)
            optimizer.step()
            epoch_loss += loss
            summary_batch = {metric:model.metrics[metric](y, label)
                                 for metric in model.metrics}
            summ.append(summary_batch)
        print('Avg Epoch Loss')
        print(epoch_loss / num_batches)
        for metric in summ[0]:
            train_metrics[metric].append(np.mean([x[metric].cpu().detach().numpy() for x in summ]))
        model.losses.append(epoch_loss/num_batches)
    with open("{0}/bn_cnn_15_10_07".format(train_folder), "wb") as fd:
        pickle.dump(model, fd)
    with open('{0}/losses'.format(train_folder), "wb") as fd:
        pickle.dump(model.losses, fd)
    with open("{0}/metrics".format(train_folder), "wb") as fd:
        pickle.dump(train_metrics, fd)

if __name__ == "__main__":
    main();
