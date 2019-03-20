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
from utils.miscellaneous import parallel_shuffle
from utils.hyperparams import HyperParams
from models import BNCNN, CNN, SSCNN, DalesBNCNN, DalesSSCNN, DalesHybrid, PracticalBNCNN, StackedBNCNN
import retio as io
import argparse
import time
from tqdm import tqdm

from deepretina.experiments import loadexpt

DEVICE = torch.device("cuda:0")

# Random Seeds (5 is arbitrary)
seed = 3
np.random.seed(seed)
torch.manual_seed(seed)

# Load data using Lane and Nirui's dataloader
cells = [0,1,2,3,4]
dataset = '15-10-07'
train_data = loadexpt(dataset,cells,'naturalscene','train',40,0)
print("Shuffling...")
parallel_shuffle([train_data.X, train_data.y], set_seed=seed)
print("train_data shape",train_data.X.shape)
test_data = loadexpt(dataset,cells,'naturalscene','test',40,0)

def train(hyps, model):
    SAVE = hyps['save_folder']
    if not os.path.exists(SAVE):
        os.mkdir(SAVE)
    LR = hyps['lr']
    LAMBDA1 = hyps['l1']
    LAMBDA2 = hyps['l2']
    EPOCHS = hyps['n_epochs']
    BATCH_SIZE = hyps['batch_size']

    with open(SAVE + "/hyperparams.txt",'w') as f:
        f.write(str(model)+'\n')
        for k in sorted(hyps.keys()):
            f.write(str(k) + ": " + str(hyps[k]) + "\n")

    model = model.to(DEVICE)
    loss_fn = torch.nn.PoissonNLLLoss()
    optimizer = torch.optim.Adam(model.parameters(),lr = LR, weight_decay = LAMBDA2)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor = 0.2)

    # train data
    epoch_tv_x = torch.FloatTensor(train_data.X)
    epoch_tv_y = torch.FloatTensor(train_data.y)

    # train/val split
    num_val = 30000
    epoch_train_x = epoch_tv_x[:-num_val]
    epoch_train_y = epoch_tv_y[:-num_val]
    epoch_val_x = epoch_tv_x[-num_val:]
    epoch_val_y = epoch_tv_y[-num_val:]
    epoch_length = epoch_train_x.shape[0]
    num_batches,leftover = divmod(epoch_length, BATCH_SIZE)
    batch_size = BATCH_SIZE
    print("Train size:", len(epoch_train_x))
    print("Val size:", len(epoch_val_x))
    print("N Batches:", num_batches, "  Leftover:", leftover)

    # test data
    test_x = torch.from_numpy(test_data.X)
    test_x = test_x[:500]

    # Train Loop
    for epoch in range(EPOCHS):
        indices = torch.randperm(epoch_train_x.shape[0]).long()
        losses = []
        epoch_loss = 0
        print('Epoch ' + str(epoch))  
        
        model.eval()
        test_obs = model(test_x.to(DEVICE)).cpu().detach().numpy()
        model.train(mode=True)

        for cell in range(test_obs.shape[-1]):
            obs = test_obs[:500,cell]
            lab = test_data.y[:500,cell]
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
        avg_loss = epoch_loss/num_batches
        print('\nAvg Loss: ' + str(avg_loss), " - exec time:", time.time() - starttime)
        #gc.collect()
        #max_mem_used = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        #print("Memory Used: {:.2f} memory".format(max_mem_used / 1024))

        #validate model
        del label
        model.eval()
        val_obs = []
        val_loss = 0
        step_size = 5000
        n_loops = epoch_val_x.shape[0]//step_size
        for v in tqdm(range(0, n_loops*step_size, step_size)):
            x = epoch_val_x[v:v+step_size]
            y = epoch_val_y[v:v+step_size]
            temp = model(x.to(DEVICE)).detach()
            val_loss += loss_fn(temp, y.to(DEVICE)).item()
            val_obs.append(temp.cpu().numpy())
        val_loss = val_loss/n_loops
        val_obs = np.concatenate(val_obs, axis=0)
        val_acc = np.mean([pearsonr(val_obs[:, i], epoch_val_y[:val_obs.shape[0], i].numpy()) for i in range(epoch_val_y.shape[-1])])
        print("Val Acc:", val_acc, " -- Val Loss:", val_loss, " | SaveFolder:", SAVE)
        scheduler.step(val_loss)
        save_dict = {
            "model": model,
            "model_state_dict":model.state_dict(),
            "optim_state_dict":optimizer.state_dict(),
            "loss": avg_loss,
            "epoch":epoch,
            "val_loss":val_loss,
            "val_acc":val_acc,
        }
        io.save_checkpoint_dict(save_dict,SAVE,'test')
        del val_obs
        del temp
        print()
    
    results = {"Loss":avg_loss, "ValAcc":val_acc, "ValLoss":val_loss}
    with open(SAVE + "/hyperparams.txt",'a') as f:
        f.write("\n" + " ".join([k+":"+str(results[k]) for k in sorted(results.keys())]) + '\n')
    return results

def hyper_search(hyps, hyp_ranges, keys, train, idx=0):
    """
    Recursive function to loop through each of the hyperparameter combinations

    hyps - dict of hyperparameters created by a HyperParameters object
        type: dict
        keys: name of hyperparameter
        values: value of hyperparameter
    hyp_ranges - dict of ranges for hyperparameters to take over the search
        type: dict
        keys: name of hyperparameters to be searched over
        values: list of values to search over for that hyperparameter
    keys - keys of the hyperparameters to be searched over. Used to
            specify order of keys to search
    train - method that handles training of model. Should return a dict of results.
    idx - the index of the current key to be searched over
    """
    # Base call, runs the training and saves the result
    if idx >= len(keys):
        if 'exp_num' not in hyps:
            hyps['exp_num'] = 0
            if not os.path.exists(hyps['exp_name']):
                os.mkdir(hyps['exp_name'])
            hyps['results_file'] = hyps['exp_name']+"/results.txt"
        hyps['save_folder'] = hyps['exp_name'] + "/" + hyps['exp_name'] +"_"+ str(hyps['exp_num']) 
        for k in keys:
            hyps['save_folder'] += "_" + str(k)+str(hyps[k])
        model = hyps['model_type'](hyps['n_output_units'], noise=hyps['noise'])
        results = train(hyps, model)
        with open(hyps['results_file'],'a') as f:
            if hyps['exp_num'] == 0:
                f.write(str(model)+'\n\n')
            results = " ".join([k+":"+str(results[k]) for k in sorted(results.keys())])
            f.write(hyps['save_folder'].split("/")[-1] + ":\n\t" + results +"\n\n")
        hyps['exp_num'] += 1

    # Non-base call. Sets a hyperparameter to a new search value and passes down the dict.
    else:
        key = keys[idx]
        for param in hyp_ranges[key]:
            hyps[key] = param
            hyper_search(hyps, hyp_ranges, keys, train, idx+1)
    return

if __name__ == "__main__":
    hyps = {}
    hyps['model_type'] = DalesSSCNN
    hyps['exp_name'] = 'absdalesSS'
    hyps['n_epochs'] = 60
    hyps['batch_size'] = 512
    hyps['shuffle'] = True
    hyps['n_output_units'] = len(cells)
    hyps = HyperParams(hyps).hyps
    hyp_ranges = {
        'lr':[1e-3, 1e-4],
        'l1':[1e-6],
        'l2':[1e-4],
        'noise':[.05],
    }
    keys = ['noise', 'lr', 'l1', 'l2']
    hyper_search(hyps, hyp_ranges, keys, train)

