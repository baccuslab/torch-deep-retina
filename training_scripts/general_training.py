"""
Used as a general training script. Can give command line arguments to specify which json files to use
for hyperparameters and the ranges that those hyperparameters can take on.

$ python3 general_training.py params=hyperparams.json ranges=hyperranges.json

Defaults to hyperparams.json and hyperranges.json if no arguments are provided
"""
import matplotlib
matplotlib.use('Agg')
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
from torchdeepretina.miscellaneous import ShuffledDataSplit
from torchdeepretina.deepretina_loader import loadexpt
from torchdeepretina.models import *
import torchdeepretina.retio as io
import argparse
import time
from tqdm import tqdm
import json
import math

DEVICE = torch.device("cuda:0")

# Random Seeds
seed = 3
np.random.seed(seed)
torch.manual_seed(seed)

def train(hyps, model, data, model_hyps):
    train_data = data[0]
    test_data = data[1]
    SAVE = hyps['save_folder']
    if not os.path.exists(SAVE):
        os.mkdir(SAVE)
    LR = hyps['lr']
    LAMBDA1 = hyps['l1']
    LAMBDA2 = hyps['l2']
    EPOCHS = hyps['n_epochs']
    batch_size = hyps['batch_size']

    with open(SAVE + "/hyperparams.txt",'w') as f:
        f.write(str(model)+'\n')
        for k in sorted(hyps.keys()):
            f.write(str(k) + ": " + str(hyps[k]) + "\n")

    print(model)
    model = model.to(DEVICE)

    loss_fn = torch.nn.PoissonNLLLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr = LR, weight_decay = LAMBDA2)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor = 0.2*LR)

    # train/val split
    num_val = 20000
    data = ShuffledDataSplit(train_data, num_val)
    data.torch()
    epoch_length = data.train_shape[0]
    num_batches,leftover = divmod(epoch_length, batch_size)
    print("Train size:", epoch_length)
    print("Val size:", data.val_shape[0])
    print("N Batches:", num_batches, "  Leftover:", leftover)

    # test data
    test_x = torch.from_numpy(test_data.X)

    # Train Loop
    for epoch in range(EPOCHS):
        model.train(mode=True)
        indices = torch.randperm(data.train_shape[0]).long()

        losses = []
        epoch_loss = 0
        print('Epoch ' + str(epoch))  
        
        starttime = time.time()
        activity_l1 = torch.zeros(1).to(DEVICE)
        for batch in range(num_batches):
            optimizer.zero_grad()
            idxs = indices[batch_size*batch:batch_size*(batch+1)]
            x = data.train_X[idxs]
            label = data.train_y[idxs]
            label = label.float()
            label = label.to(DEVICE)

            y = model(x.to(DEVICE))
            y = y.float() 

            if LAMBDA1 > 0:
                activity_l1 = LAMBDA1 * torch.norm(y, 1).float()/y.shape[0]
            error = loss_fn(y,label)
            loss = error + activity_l1
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            print("Loss:", loss.item()," - error:", error.item(), " - l1:", activity_l1.item(), " | ", int(round(batch/num_batches, 2)*100), "% done", end='               \r')
            if math.isnan(epoch_loss) or math.isinf(epoch_loss):
                break
        avg_loss = epoch_loss/num_batches
        print('\nAvg Loss: ' + str(avg_loss), " - exec time:", time.time() - starttime)

        #validate model
        del x
        del y
        del label
        model.eval()
        val_preds = []
        val_loss = 0
        step_size = 500
        n_loops = data.val_shape[0]//step_size
        for v in tqdm(range(0, n_loops*step_size, step_size)):
            temp = model(data.val_X[v:v+step_size].to(DEVICE)).detach()
            val_loss += loss_fn(temp, data.val_y[v:v+step_size].to(DEVICE)).item()
            if LAMBDA1 > 0:
                val_loss += (LAMBDA1 * torch.norm(temp, 1).float()/temp.shape[0]).item()
            val_preds.append(temp.cpu().numpy())
        val_loss = val_loss/n_loops
        val_preds = np.concatenate(val_preds, axis=0)
        pearsons = []
        for cell in range(val_preds.shape[-1]):
            pearsons.append(pearsonr(val_preds[:, cell], data.val_y[:val_preds.shape[0]][:,cell].numpy())[0])
        print("Val Cell Pearsons:", " - ".join([str(p) for p in pearsons]))
        val_acc = np.mean(pearsons)
        print("Avg Val Pearson:", val_acc, " -- Val Loss:", val_loss, " | SaveFolder:", SAVE)
        scheduler.step(val_loss)
        del val_preds
        del temp

        test_obs = model(test_x.to(DEVICE)).cpu().detach().numpy()

        avg_pearson = 0
        for cell in range(test_obs.shape[-1]):
            obs = test_obs[:,cell]
            lab = test_data.y[:,cell]
            r,p = pearsonr(obs,lab)
            avg_pearson += r
            print('Cell ' + str(cell) + ': ')
            print('-----> pearsonr: ' + str(r))
        avg_pearson = avg_pearson / float(test_obs.shape[-1])
        print("Avg Test Pearson:", avg_pearson)
        del test_obs

        save_dict = {
            "model_hyps": model_hyps,
            "model_state_dict":model.state_dict(),
            "optim_state_dict":optimizer.state_dict(),
            "loss": avg_loss,
            "epoch":epoch,
            "val_loss":val_loss,
            "val_acc":val_acc,
            "test_pearson":avg_pearson,
            "norm_stats":train_data.stats,
        }
        io.save_checkpoint_dict(save_dict,SAVE,'test')
        print()
        # If loss is nan, training is futile
        if math.isnan(avg_loss) or math.isinf(avg_loss):
            break
    
    results = {"Loss":avg_loss, "ValAcc":val_acc, "ValLoss":val_loss, "TestPearson":avg_pearson}
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
            if 'starting_exp_num' not in hyps: hyps['starting_exp_num'] = 0
            hyps['exp_num'] = hyps['starting_exp_num']
            if not os.path.exists(hyps['exp_name']):
                os.mkdir(hyps['exp_name'])
            hyps['results_file'] = hyps['exp_name']+"/results.txt"
        hyps['save_folder'] = hyps['exp_name'] + "/" + hyps['exp_name'] +"_"+ str(hyps['exp_num']) 
        for k in keys:
            hyps['save_folder'] += "_" + str(k)+str(hyps[k])
        print("Loading", hyps['stim_type'],"using Cells:", hyps['cells'], "from dataset:", hyps['dataset'])
        train_data = DataContainer(loadexpt(hyps['dataset'],hyps['cells'],
                                            hyps['stim_type'],'train',40,0))
        norm_stats = [train_data.stats['mean'], train_data.stats['std']] 
        test_data = DataContainer(loadexpt(hyps['dataset'],hyps['cells'],hyps['stim_type'],
                                                        'test',40,0, norm_stats=norm_stats))
        test_data.X = test_data.X[:500]
        test_data.y = test_data.y[:500]
        data = [train_data, test_data]
        model_hyps = {"n_units":test_data.y.shape[-1]}
        for key in hyps.keys():
            model_hyps[key] = hyps[key]
        fn_args = set(hyps['model_type'].__init__.__code__.co_varnames)
        keys = list(model_hyps.keys())
        for k in keys:
            if k not in fn_args:
                del model_hyps[k]
        model = hyps['model_type'](**model_hyps)
        results = train(hyps, model, data, model_hyps)
        with open(hyps['results_file'],'a') as f:
            if hyps['exp_num'] == hyps['starting_exp_num']:
                f.write(str(model)+'\n\n')
                f.write("Hyperparameters:\n")
                for k in hyps.keys():
                    if k not in hyp_ranges:
                        f.write(str(k) + ": " + str(hyps[k]) + '\n')
                f.write("\nHyperranges:\n")
                for k in hyp_ranges.keys():
                    f.write(str(k) + ": [" + ",".join([str(v) for v in hyp_ranges[k]])+']\n')
                f.write('\n')
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

def load_data(dataset, cells):
    return train_data, test_data

def load_json(file_name):
    with open(file_name) as f:
        s = f.read()
        j = json.loads(s)
    return j

class DataContainer():
    def __init__(self, data):
        self.X = data.X
        self.y = data.y
        self.stats = data.stats

if __name__ == "__main__":
    hyperparams_file = "hyperparams.json"
    hyperranges_file = 'hyperranges.json'
    if len(sys.argv) > 1:
        for i,arg in enumerate(sys.argv[1:]):
            temp = sys.argv[1].split("=")
            if len(temp) > 1:
                if "params" in temp[0]:
                    hyperparams_file = temp[1]
                elif "ranges" in temp[0]:
                    hyperranges_file = temp[1]
            else:
                if i == 0:
                    hyperparams_file = arg
                elif i == 1:
                    hyperranges_file = arg
                else:
                    print("Too many command line args")
                    assert False
    print("Using hyperparams file:", hyperparams_file)
    print("Using hyperranges file:", hyperranges_file)

    hyps = load_json(hyperparams_file)
    hyp_ranges = load_json(hyperranges_file)
    sleep_time = 8
    print("You have "+str(sleep_time)+" seconds to cancel experiment name "+
                hyps['exp_name']+" (num "+ str(hyps['starting_exp_num'])+"): ")
    time.sleep(sleep_time)
    print("Model type:", hyps['model_type'])
    hyps['model_type'] = globals()[hyps['model_type']]
    keys = list(hyp_ranges.keys())
    print("Searching over:", keys)

    hyper_search(hyps, hyp_ranges, keys, train, 0)

