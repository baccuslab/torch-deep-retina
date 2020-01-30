import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import numpy as np
import scipy.stats
import h5py

import deepretina.experiments
import pyret.spiketools as spk
from model import CNN, LN

# device 
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# hyperparameters
learning_rate = 1e-3
batch_size = 2000
num_cells = 120
num_epochs = 5

# load data
num_frame_window = 60
with h5py.File("/home/dlee/data/17-11-08_17+/17-10-18_17+_rf_30m.h5", 'r') as f:
    stim = np.asarray(f['data/rf_30m/stim'])
    data = np.asarray(f['data/rf_30m/ganglion'][:num_cells, :])
    tbins = np.asarray(f['data/rf_30m/tbins'])

stim_roll = deepretina.experiments.rolling_window(stim, num_frame_window)
#del stim#, tbins
#resp_roll = resp[num_frame_window:]
data = np.stack([spk.estfr(data[g, :], tbins, sigma=0.01) for g in range(data.shape[0])])
#data = data / np.std(data, 1)[:, None]
resp_roll = data[:, num_frame_window:].T
del data

# torch.utils.data.TensorDataset
# train_loader = torch.utils.data.DataLoader

#train_x = torch.Tensor(stim_roll[10000:130000])
#train_y = torch.Tensor(resp_roll[10000:130000])
#test_x = torch.Tensor(stim_roll[130000:135000])
#test_y = torch.Tensor(resp_roll[130000:135000])
train_x = torch.Tensor(stim_roll[2000:122000])
train_y = torch.Tensor(resp_roll[2000:122000])
test_x = torch.Tensor(stim_roll[1000:2000])
test_y = torch.Tensor(resp_roll[1000:2000])
del stim_roll, resp_roll

train_dataset = torch.utils.data.TensorDataset(train_x, train_y)
test_dataset = torch.utils.data.TensorDataset(test_x, test_y)

train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
        batch_size=batch_size, shuffle=False)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
        batch_size=100, shuffle=False)

# model
model = CNN(out=num_cells).to(DEVICE)

# loss, optimizer and scheduler
loss_fn = nn.PoissonNLLLoss(log_input=True)#
#loss_fn = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-2)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min')

# train
def train(model):
    num_batches = len(train_loader)
    for epoch in range(num_epochs):
        for batch, (movies, response) in enumerate(train_loader):
            movies = movies.to(DEVICE)
            response = response.to(DEVICE)

            # forward pass
            pred = model(movies)
#            activity_l1 = 1e-3 * torch.norm(pred, 1)
            loss = loss_fn(pred[:, :, 0, 0], response)# + activity_l1

            # backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # print
            if (batch+1) % 10 == 0:
                print('Epoch [{}/{}], Iteration [{}/{}], Loss: {:.9f}'
                        .format(epoch+1, num_epochs, batch+1, num_batches,
                            loss.item()))

    model.eval()
    with torch.no_grad():
        train_corr = []
        test_corr = []

        movies1 = train_x[2000:2500].to(DEVICE)
        pred1 = model(movies1).cpu().detach().numpy()
        response1 = train_y[2000:2500].numpy()

        movies2 = test_x.to(DEVICE)
        pred2 = model(movies2).cpu().detach().numpy()
        response2 = test_y.numpy()

        for cell in range(pred1.shape[1]):
            r1, p1 = scipy.stats.pearsonr(pred1[:, cell, 0, 0].reshape(-1),
                    response1[:, cell].reshape(-1))
            r2, p2 = scipy.stats.pearsonr(pred2[:, cell, 0, 0].reshape(-1),
                    response2[:, cell].reshape(-1))
            #            print(pred2, response2)
            train_corr.append(r1)
            test_corr.append(r2)
        print("Train correlation coefficient is: {}".format(train_corr))
        print("Test correlation coefficient is: {}".format(test_corr))

    torch.save(model.state_dict(), 'model.ckpt')

def main():
    train(model)

if __name__ == "__main__":
    main()
