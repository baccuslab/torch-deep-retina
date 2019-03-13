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
from utils.hyperparams import HyperParams
from models.BN_CNN import BNCNN
from models.CNN import CNN
from models.SS_CNN import SSCNN
from models.Dales_BN_CNN import DalesBNCNN
from models.Dales_SS_CNN import DalesSSCNN
from models.Dales_CNN import DalesCNN
import retio as io
import argparse
import time

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


# Load data using Lane and Nirui's dataloader
#test_data = loadexpt('15-10-07',[0,1,2,3,4],'naturalscene','test',40,0)
#val_split = 0.005

def train(epochs=250,batch_size=5000,LR=1e-3,l1_scale=1e-4,l2_scale=1e-2, shuffle=True, save='./checkpoints'):
    if not os.path.exists(save):
        os.mkdir(save)
    train_data = loadexpt('19-02-26',[0,1],'naturalmovie','train',40,0)
    LAMBDA1 = l1_scale
    LAMBDA2 = l2_scale
    EPOCHS = epochs
    BATCH_SIZE = batch_size

    # Model
    #model = model_class()
    output_units = 2
    model = BNCNN(2)
    #model = CNN(bias=False)
    #model = SSCNN(scale=True, shift=False, bias=True)
    #model = DalesBNCNN(bias=True, neg_p=.5)
    print(model)
    model = model.to(DEVICE)

    loss_fn = torch.nn.PoissonNLLLoss()
    optimizer = torch.optim.Adam(model.parameters(),lr = LR, weight_decay = LAMBDA2)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor = 0.2)

    # train/val split
    val_p = .015
    n_samples = train_data.X.shape[0]
    n_val = int(n_samples*val_p)
    val_chunk = n_val//3
    val_idxs = np.concatenate([np.arange(0,val_chunk), 
                               np.arange(n_samples//2, n_samples//2+val_chunk), 
                               np.arange(n_samples-val_chunk, n_samples)], 0).astype(np.int)
    print("Validxs:", val_idxs[:10]) 
    test_x = train_data.X[val_idxs]
    test_y = train_data.y[val_idxs]
    del val_idxs
    train_idxs = np.concatenate([np.arange(val_chunk, n_samples//2), 
                                np.arange(n_samples//2+val_chunk, n_samples-val_chunk)], 0).astype(np.int)
    epoch_train_x = train_data.X[train_idxs]
    epoch_train_y = train_data.y[train_idxs]
    del train_data
    del train_idxs
    epoch_train_x = torch.FloatTensor(epoch_train_x)
    epoch_train_y = torch.FloatTensor(epoch_train_y)
    epoch_length = epoch_train_x.shape[0]
    num_batches,leftover = divmod(epoch_length, BATCH_SIZE)
    batch_size = BATCH_SIZE
    print("Train size:", len(epoch_train_x))
    print("Val size:", len(epoch_val_x))
    print("N Batches:", num_batches, "  Leftover:", leftover)

    # test data
    #test_x = torch.from_numpy(test_data.X)
    #test_x = test_x[:500]

    # Train Loop
    for epoch in range(EPOCHS):
        if shuffle:
            indices = torch.randperm(epoch_train_x.shape[0]).long()
        else:
            indices = torch.arange(0, epoch_train_x.shape[0]).long()

        losses = []
        epoch_loss = 0
        print('Epoch ' + str(epoch))  
        
        model.eval()
        test_obs = model(test_x.to(DEVICE)).cpu().detach().numpy()
        model.train(mode=True)

        for cell in range(test_obs.shape[-1]):
            obs = test_obs[:500,cell]
            lab = test_y[:500,cell]
            r,p = pearsonr(obs,lab)
            print('Cell ' + str(cell) + ': ')
            print('-----> pearsonr: ' + str(r))
        
        starttime = time.time()
        activity_l1 = torch.zeros(1).to(DEVICE)
        for batch in range(num_batches):
            optimizer.zero_grad()
            idxs = indices[batch_size*batch:batch_size*(batch+1)]
            x = epoch_train_x[idxs]
            label = epoch_train_y[idxs]
            label = label.float()
            label = label.to(DEVICE)

            y = model(x.to(DEVICE))
            y = y.float() 

            if LAMBDA1 > 0:
                activity_l1 = LAMBDA1 * torch.norm(y, 1).float()
            error = loss_fn(y,label)
            loss = error + activity_l1
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            print("Loss:", loss.item()," - error:", error.item(), " - l1:", activity_l1.item(), " | ", int(round(batch/num_batches, 2)*100), "% done", end='               \r')
        print('\nAvg Loss: ' + str(epoch_loss/num_batches), " - exec time:", time.time() - starttime)
        #gc.collect()
        #max_mem_used = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        #print("Memory Used: {:.2f} memory".format(max_mem_used / 1024))

        #validate model
        del x
        del y
        del label
        #val_obs = model(epoch_val_x.to(DEVICE)).cpu().detach().numpy()
        #val_acc = np.mean([pearsonr(val_obs[:, i], epoch_val_y[:, i]) for i in range(epoch_val_y.shape[-1])])
        print("SaveFolder:", save)
        scheduler.step(error)
        io.save_checkpoint(model,epoch,epoch_loss/num_batches,optimizer,save,'test')
        print()
    return

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


def parseargs():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', default = 200)
    parser.add_argument('--batch', default = 1028)
    parser.add_argument('--lr', default = 1e-4)
    parser.add_argument('--l2', default = 0.01)
    parser.add_argument('--l1', default = 1e-7)
    parser.add_argument('--shuffle', default=True)
    parser.add_argument('--save', default='./checkpoints')
    return parser.parse_args(sys.argv[1:])


if __name__ == "__main__":
    #args = parseargs()
    #train(int(args.epochs), int(args.batch), float(args.lr), float(args.l1), float(args.l2), args.shuffle, args.save)
    #train(50, 512, 1e-4, 0, .01, True, "delete_me")
    hp = HyperParams()
    hyps = hp.hyps
    hyps['exp_name'] = 'mouseBN'
    hyps['n_epochs'] = 35
    hyps['batch_size'] = 512
    hyps['shuffle'] = True
    lrs = [1e-3, 1e-4, 1e-5, 1e-1, 1e-2, 1e-6]
    l1s = [0]
    l2s = [1e-2]
    exp_num = 0
    for lr in lrs:
        hyps['lr'] = lr
        for l1 in l1s:
            hyps['l1'] = l1
            for l2 in l2s:
                hyps['l2'] = l2
                hyps['save_folder'] = hyps['exp_name'] +"_"+ str(exp_num) + "_lr"+str(lr) + "_" + "l1" + str(l1) + "_" + "l2" + str(l2)
                hp.print()            
                train(hyps['n_epochs'], hyps['batch_size'], lr, l1, l2, hyps['shuffle'], hyps['save_folder'])
                exp_num += 1


