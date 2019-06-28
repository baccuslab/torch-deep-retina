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
from torchdeepretina.utils import ShuffledDataSplit
from torchdeepretina.models import *
import torchdeepretina.retio as io
from torchdeepretina.deepretina_loader import loadexpt
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

def train(hyps, model, train_datas, model_hyps):
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
    datas = []
    for train_data in train_datas:
        datas.append(ShuffledDataSplit(train_data, num_val))
        datas[-1].set_batch_size(batch_size)
        datas[-1].torch()
    epoch_length = np.max([data.train_shape[0] for data in datas])
    num_batches, leftover = divmod(epoch_length, batch_size)
    print("Train sizes:", ",".join([str(data.train_shape[0]) for data in datas]))
    print("Val size:", num_val)
    print("N Batches:", num_batches)

    # Train Loop
    for epoch in range(EPOCHS):
        #indices = torch.randperm(data.train_shape[0]).long()
        epoch_loss = 0
        print('Epoch ' + str(epoch))  
        
        starttime = time.time()
        activity_l1 = torch.zeros(1).to(DEVICE)
        for batch in range(num_batches):
            optimizer.zero_grad()
            for i,data in enumerate(datas):
                x,label = next(data.train_sample())
                label = label.float()
                label = label.to(DEVICE)

                y = model(x.to(DEVICE), i) # i specifies output layer to use
                y = y.float() 

                if LAMBDA1 > 0:
                    activity_l1 = LAMBDA1 * torch.norm(y, 1).float()/y.shape[0]
                error = loss_fn(y,label)
                loss = (error + activity_l1)/len(datas) # Avg loss accross datasets
                loss.backward()
                epoch_loss += loss.item()
                print("Loss:", loss.item()," - error:", error.item(), " - l1:", activity_l1.item(), " | ", int(round(batch/num_batches, 2)*100), "% done", end='               \r')
            optimizer.step()
            if math.isnan(epoch_loss) or math.isinf(epoch_loss):
                return {"Loss":epoch_loss, "ValAcc":None, "ValLoss":None}
        avg_loss = epoch_loss/num_batches
        print('\nAvg Loss: ' + str(avg_loss), " - exec time:", time.time() - starttime)

        #validate model
        del x
        del y
        del label
        model.eval()
        model.calc_grad(False)
        step_size = 2500
        val_losses = []
        val_accs = []
        for i,data in enumerate(datas):
            val_preds = []
            n_loops = data.val_shape[0]//step_size
            val_loss = 0
            for v in tqdm(range(0, n_loops*step_size, step_size)):
                temp = model(data.val_X[v:v+step_size].to(DEVICE), i).detach()
                val_loss += loss_fn(temp, data.val_y[v:v+step_size].to(DEVICE)).item()
                if LAMBDA1 > 0:
                    val_loss += (LAMBDA1 * torch.norm(temp, 1).float()/temp.shape[0]).item()
                val_preds.append(temp.cpu().numpy())
            val_losses.append(val_loss/n_loops)
            val_preds = np.concatenate(val_preds, axis=0)
            val_accs.append(np.mean([pearsonr(val_preds[:, i], data.val_y[:val_preds.shape[0]][:,i].numpy()) for i in range(val_preds.shape[-1])]))
        model.train(mode=True)
        model.calc_grad(True)
        print("Val Acc:", np.mean(val_accs), " -- Val Loss:", np.mean(val_losses), " | SaveFolder:", SAVE)
        scheduler.step(val_loss)

        save_dict = {
            "model_hyps": model_hyps,
            "model_state_dict":model.state_dict(),
            "optim_state_dict":optimizer.state_dict(),
            "loss": avg_loss,
            "epoch":epoch,
            "val_losses":val_losses,
            "val_accs":val_accs,
            "norm_stats":train_data.stats,
        }
        io.save_checkpoint_dict(save_dict,SAVE,'test',del_prev=True)
        del val_preds
        del temp
        print()
    
    results = {"Loss":avg_loss, "ValAcc":np.mean(val_accs), "ValLoss":np.mean(val_losses)}
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
        train_datas = []
        norm_stats = None
        for dataset,cells in zip(hyps['datasets'], hyps['cells']):
            for stim_type in hyps['stim_types']:
                print("Loading", stim_type,"using Cells:", cells, "from dataset:", dataset)
                train_datas.append(DataContainer(loadexpt(dataset,cells, stim_type,'train',40,0, norm_stats=norm_stats)))
                if len(train_datas) == 1: # Simply use normalization stats from first dataset
                    norm_stats = [train_datas[-1].stats['mean'], train_datas[-1].stats['std']]                
        n_units = [data.y.shape[-1] for data in train_datas]
        model_hyps = {"n_units":n_units,"noise":hyps['noise'],"bias":hyps['bias']}
        if "chans" in hyps:
            model_hyps['chans'] = hyps['chans']
        if "adapt_gauss" in hyps:
            model_hyps['adapt_gauss'] = hyps['adapt_gauss']
        fn_args = set(hyps['model_type'].__init__.__code__.co_varnames)
        for k in model_hyps.keys():
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
    hyperparams_file = "paralleldata_hyperparams.json"
    hyperranges_file = 'hyperranges.json'
    hyps = load_json(hyperparams_file)
    inp = input("Last chance to change the experiment name "+hyps['exp_name']+": ")
    inp = inp.strip()
    if inp is not None and inp != "":
        hyps['exp_name'] = inp
    hyp_ranges = load_json(hyperranges_file)
    print("Model type:", hyps['model_type'])
    hyps['model_type'] = globals()[hyps['model_type']]
    keys = list(hyp_ranges.keys())
    print("Searching over:", keys)

    hyper_search(hyps, hyp_ranges, keys, train, 0)

