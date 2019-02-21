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
import gc
import resource
sys.path.append('../models/')
sys.path.append('../utils/')
from BN_CNN import BNCNN
import retio as io
import argparse

from deepretina.experiments import loadexpt

# Helper function (used instead of .to(DEVICE) for memory leak debugging)
def cuda_if(tensor):
	if torch.cuda.is_available():
		return tensor.cuda()
	return tensor

# Constants
DEVICE = torch.device("cuda:0")

# Load data using Lane and Nirui's dataloader
train_data = loadexpt('15-10-07',[0,1,2,3,4],'naturalscene','train',40,0)
test_data = loadexpt('15-10-07',[0,1,2,3,4],'naturalscene','test',40,0)
val_split = 0.005

def train(model_class,epochs=200,batch_size=1000,LR=1e-3,l2_scale=0.01,l1_scale=0.01,shuffle=False, save='./save_folder'):
    if not os.path.exists(save):
        os.mkdir(save)
    LAMBDA1 = l1_scale
    LAMBDA2 = l2_scale
    EPOCHS = epochs
    BATCH_SIZE = batch_size

    model = BNCNN()
    model = cuda_if(model)

    loss_fn = torch.nn.PoissonNLLLoss()
    optimizer = torch.optim.Adam(model.parameters(),lr = LR, weight_decay = LAMBDA2)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor = 0.2)

    # train data
    epoch_tv_x = torch.FloatTensor(train_data.X)
    epoch_tv_y = torch.FloatTensor(train_data.y)
    if shuffle:
        print('shuffling data...')
        np.random.shuffle(epoch_tv_x)
        np.random.shuffle(epoch_tv_y)
        print('data shuffled!')
    # train/val split
    num_val = 2000
    epoch_train_x = epoch_tv_x[num_val:]
    epoch_val_x = epoch_tv_x[:num_val]
    epoch_train_y = epoch_tv_y[num_val:]
    epoch_val_y = epoch_tv_y[:num_val]
    epoch_length = epoch_train_x.shape[0]
    num_batches,leftover = divmod(epoch_length, BATCH_SIZE)
    batch_size = BATCH_SIZE

    # test data
    test_x = torch.from_numpy(test_data.X)
    test_x = test_x[:500]

    # Train Loop
    for epoch in range(EPOCHS):
        if shuffle:
            print('shuffling data...')
            np.random.shuffle(epoch_train_x)
            np.random.shuffle(epoch_train_y)
            print('data shuffled!')

        losses = []
        epoch_loss = 0
        print('Epoch ' + str(epoch))  
        
        model.eval()
        test_obs = model(cuda_if(test_x)).cpu().detach().numpy()
        model.train(mode=True)

        for cell in range(test_obs.shape[-1]):
            obs = test_obs[:500,cell]
            lab = test_data.y[:500,cell]
            r,p = pearsonr(obs,lab)
            print('Cell ' + str(cell) + ': ')
            print('-----> pearsonr: ' + str(r))

        for batch in range(num_batches):
            optimizer.zero_grad()
            x = epoch_train_x[batch_size*batch:batch_size*(batch+1)]
            label = epoch_train_y[batch_size*batch:batch_size*(batch+1)]
            label = label.float()
            label = cuda_if(label)

            x = cuda_if(x)
            y = model(x)
            y = y.float() 

            activity_l1 = LAMBDA1 * torch.norm(y, 1).float()
            loss = loss_fn(y,label) + activity_l1
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            print("Loss:", loss.item(), " | ", round(batch/num_batches, 2)*100, "% complete", end='               \r')
        print('Loss: ' + str(epoch_loss/num_batches))
        gc.collect()
        max_mem_used = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        print("Memory Used: {:.2f} memory\n".format(max_mem_used / 1024))

        #validate model
        val_obs = model(epoch_val_x.to(DEVICE)).cpu().detach().numpy()
        val_acc = np.sum([pearsonr(val_obs[:, i], epoch_val_y[:, i]) for i in range(epoch_val_y.shape[-1])])
        scheduler.step(val_acc)
        io.save_checkpoint(model,epoch,epoch_loss/num_batches,optimizer,save,'test')
    return val_acc

def hyperparameter_search(param, values):
    best_val_acc = 0
    best_val = None
    for val in values:
        save = '~/julia/torch-deepretina/Trained_1/29/18_{0}_{1}'.format(param, val)
        if param == 'batch_size':
            val_acc = train(BNCNN, batch_size=val, save=save)
        elif param == 'lr':
            val_acc = train(BNCNN, LR=val, save=save)
        elif param == 'l2':
            val_acc = train(BNCNN, l2_scale=val, save=save)
        elif param == 'l1':
            val_acc = train(BNCNN, l1_scale=val, save=save)
        if val_loss > best_val_loss:
            best_val_acc = val_acc
            best_val = val
    print("The best valuation loss achieved was {0} with a {1} value of {2}".format(best_val_loss, param, best_val))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', default = 200)
    parser.add_argument('--batch', default = 1000)
    parser.add_argument('--lr', default = 1e-4)
    parser.add_argument('--l2', default = 0.01)
    parser.add_argument('--l1', default = 0.01)
    parser.add_argument('--shuffle', action='store_true')
    parser.add_argument('--save', default='./checkpoints')
    args = parser.parse_args(sys.argv[1:])
    train(BNCNN, int(args.epochs), int(args.batch), float(args.lr), float(args.l2), float(args.l1), args.shuffle, args.save)


if __name__ == "__main__":
    main()
