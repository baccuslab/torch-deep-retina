from scipy.stats import pearsonr
import os
import sys
from time import sleep
import numpy as np
import torch
import torch.nn as nn
from torch.nn import PoissonNLLLoss, MSELoss
import os.path as path
from torchdeepretina.utils import get_cuda_info, save_checkpoint
from torchdeepretina.datas import loadexpt, DataContainer, DataDistributor
from torchdeepretina.models import *
import time
from tqdm import tqdm
import math
import torch.multiprocessing as mp
from queue import Queue
import psutil
import gc
import resource
import json

class Trainer:
    def __init__(self, run_q=None, return_q=None, early_stopping=10, stop_tolerance=0.01):
        self.run_q = run_q
        self.ret_q = return_q
        self.early_stopping = early_stopping
        self.tolerance = stop_tolerance
        self.prev_acc = None

    def stop_early(self, acc):
        if self.early_stopping <= 0:
            return False # use 0 as way to cancel early stopping
        if self.prev_acc is None:
            self.prev_acc = acc
            self.stop_count = 0
            return False
        if acc-self.prev_acc < self.tolerance:
            self.stop_count += 1
            if self.stop_count >= self.early_stopping:
                return True
        self.stop_count = 0
        self.prev_acc = acc
        return False

    def loop_training(self):
        train_args = self.run_q.get()
        while True:
            try:
                results = self.train(*train_args)
                self.ret_q.put([results])
                train_args = self.run_q.get()
            except Exception as e:
                print("Caught error",e,"on", train_args[0]['exp_num'], "will retry in 100 seconds...")
                sleep(100)

    def train(self, hyps, model_hyps, device, verbose=False):
        SAVE = hyps['save_folder']
        if 'skip_nums' in hyps and len(hyps['skip_nums']) > 0 and hyps['exp_num'] in hyps['skip_nums']:
            print("Skipping", SAVE)
            results = {"save_folder":SAVE, "Loss":None, "ValAcc":None, "ValLoss":None, "TestPearson":None}
            return results
            
        # Get Data
        img_depth, img_height, img_width = hyps['img_shape']
        train_data = DataContainer(loadexpt(hyps['dataset'],hyps['cells'], hyps['stim_type'],'train',img_depth,0))
        norm_stats = [train_data.stats['mean'], train_data.stats['std']] 
        test_data = DataContainer(loadexpt(hyps['dataset'],hyps['cells'],hyps['stim_type'],'test',img_depth,0, 
                                                                            norm_stats=norm_stats))
        test_data.X = test_data.X[:500]
        test_x = torch.from_numpy(test_data.X)
        test_data.y = test_data.y[:500]

        # Make model
        model_hyps["n_units"] = test_data.y.shape[-1]
        model = hyps['model_class'](**model_hyps)
        model = model.to(device)

        # Initialize miscellaneous parameters 
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

        with open(SAVE + "/hyperparams.json",'w') as f:
            temp_hyps = {k:v for k,v in hyps.items()}
            del temp_hyps['model_class']
            json.dump(temp_hyps, f)

        # train/val split
        num_val = 10000
        shuffle = hyps['shuffle']
        recurrent = model.recurrent
        seq_len = 1
        if recurrent:
            seq_len = hyps['recur_seq_len']
        data_distr = DataDistributor(train_data, num_val, batch_size=batch_size, shuffle=shuffle, 
                                                            recurrent=recurrent, seq_len=seq_len)
        data_distr.torch()
    
        # Make optimization objects (lossfxn, optimizer, scheduler)
        if 'lossfxn' not in hyps:
            hyps['lossfxn'] = "PoissonNLLLoss"
        if hyps['lossfxn'] == "PoissonNLLLoss" and 'log_poisson' in hyps:
            loss_fn = globals()[hyps['lossfxn']](log_input=hyps['log_poisson'])
        else:
            loss_fn = globals()[hyps['lossfxn']]()
        optimizer = torch.optim.Adam(model.parameters(), lr = LR, weight_decay = LAMBDA2)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor = 0.1)

        # Train Loop
        hs = None
        if "shapes" in model.__dict__:
            h1_shape = model.h_shapes[0] # (H,W)
            h2_shape = model.h_shapes[1] # (H, W)
            
        for epoch in range(EPOCHS):
            print("Beginning Epoch", epoch, " -- ", SAVE)
            print()
            n_loops = data_distr.n_loops
            if model.recurrent:
                hs = [torch.zeros(batch_size, *h1_shape).to(device),
                        torch.zeros(batch_size, *h2_shape).to(device)]
            model.train(mode=True)
            indices = torch.randperm(data_distr.train_shape[0]).long()

            losses = []
            epoch_loss = 0
            stats_string = 'Epoch ' + str(epoch) + " -- " + SAVE + "\n"
            
            starttime = time.time()
            activity_l1 = torch.zeros(1).to(device)
            for i,(x,label) in enumerate(data_distr.train_sample()):
                optimizer.zero_grad()
                label = label.float()
                label = label.to(device)

                # Recurrent loop
                if hs is not None:
                    seq_len = hyps['recur_seq_len']
                    hs_out = hs
                    ys = []
                    for ri in range(seq_len):
                        ins = x[:,ri]
                        y, hs_out = model(ins.to(device), hs_out)
                        ys.append(y.view(batch_size, 1, label.shape[-1]))
                        if ri == 0:
                            hs = [h.clone() for h in hs]
                    y = torch.cat(ys, dim=1)
                else:
                    y = model(x.to(device))
                error = loss_fn(y,label)

                if LAMBDA1 > 0:
                    activity_l1 = LAMBDA1 * torch.norm(y, 1).float().mean()
                loss = error + activity_l1
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                if verbose:
                    print("Loss:", round(loss.item(), 6), " -- ", i, "/", n_loops, end="       \r")
                if math.isnan(epoch_loss) or math.isinf(epoch_loss):
                    break
            avg_loss = epoch_loss/n_loops
            stats_string += 'Avg Loss: ' + str(avg_loss) + " -- exec time:"+ str(time.time() - starttime) +"\n"

            #validate model
            del x
            del y
            del label
            model.eval()
            with torch.no_grad():
                val_preds = []
                val_loss = 0
                if model.recurrent:
                    step_size = 1
                    hs = [torch.zeros(1, *h1_shape).to(device),
                            torch.zeros(1, *h2_shape).to(device)]
                    n_loops = data_distr.val_shape[0]
                else:
                    step_size = 500
                    n_loops = data_distr.val_shape[0]//step_size
                if verbose:
                    print()
                    print("Validating")
                for i, (val_x, val_y) in enumerate(data_distr.val_sample(step_size)):
                    if hs is not None:
                        outs, hs = model(val_x[:,0].to(device), hs)
                        val_y = val_y[:,0]
                    else:
                        outs = model(val_x.to(device)).detach()
                    val_loss += loss_fn(outs, val_y.to(device)).item()
                    if LAMBDA1 > 0:
                        val_loss += (LAMBDA1 * torch.norm(outs, 1).float()/outs.shape[0]).item()
                    val_preds.append(outs.cpu().numpy())
                    if verbose:
                        print("{}/{}".format(i,data_distr.val_y.shape[0]), end="     \r")
                val_loss = val_loss/n_loops
                val_preds = np.concatenate(val_preds, axis=0)
                val_targs = data_distr.val_y[:val_preds.shape[0]].numpy()
                if recurrent:
                    val_targs = val_targs[:,0]
                pearsons = []
                for cell in range(val_preds.shape[-1]):
                    pearsons.append(pearsonr(val_preds[:, cell], val_targs[:,cell])[0])
                stats_string += "Val Cell Pearsons:" + " - ".join([str(p) for p in pearsons])+'\n'
                val_acc = np.mean(pearsons)
                stop = self.stop_early(val_acc)
                exp_val_acc = None
                if hyps["log_poisson"] and hyps['softplus'] and not model.infr_exp:
                    val_preds = np.exp(val_preds)
                    for cell in range(val_preds.shape[-1]):
                        pearsons.append(pearsonr(val_preds[:, cell], data_distr.val_y[:val_preds.shape[0]][:,cell].numpy())[0])
                    exp_val_acc = np.mean(pearsons)
                stats_string += "Avg Val Pearson: {} -- Val Loss: {} -- Exp Val: {}\n".format(val_acc, val_loss, exp_val_acc)
                scheduler.step(val_loss)
                del val_preds
                del outs

                if hs is not None:
                    avg_pearson = 0
                else:
                    test_obs = model(test_x.to(device)).cpu().detach().numpy()

                    avg_pearson = 0
                    rng = range(test_obs.shape[-1])
                    if verbose:
                        rng = tqdm(rng)
                    for cell in rng:
                        obs = test_obs[:,cell]
                        lab = test_data.y[:,cell]
                        r,p = pearsonr(obs,lab)
                        avg_pearson += r
                        stats_string += 'Cell ' + str(cell) + ': ' + str(r)+"\n"
                    avg_pearson = avg_pearson / float(test_obs.shape[-1])
                    stats_string += "Avg Test Pearson: "+ str(avg_pearson) + "\n"
                    del test_obs
    
            optimizer.zero_grad()
            save_dict = {
                "model_type": hyps['model_type'],
                "model_hyps": model_hyps,
                "model_state_dict":model.state_dict(),
                "optim_state_dict":optimizer.state_dict(),
                "loss": avg_loss,
                "epoch":epoch,
                "val_loss":val_loss,
                "val_acc":val_acc,
                "exp_val_acc":exp_val_acc,
                "test_pearson":avg_pearson,
                "norm_stats":train_data.stats,
            }
            for k in hyps.keys():
                if k not in save_dict:
                    save_dict[k] = hyps[k]
            save_checkpoint(save_dict, SAVE, 'test', del_prev=True)
            gc.collect()
            max_mem_used = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            stats_string += "Memory Used: {:.2f} mb".format(max_mem_used / 1024)+"\n"
            print(stats_string)
            # If loss is nan, training is futile
            if math.isnan(avg_loss) or math.isinf(avg_loss) or stop:
                break

        results = {"save_folder":SAVE, "Loss":avg_loss, "ValAcc":val_acc, "ValLoss":val_loss, "TestPearson":avg_pearson}
        with open(SAVE + "/hyperparams.txt",'a') as f:
            f.write("\n" + " ".join([str(k)+":"+str(results[k]) for k in sorted(results.keys())]) + '\n')
        return results

def fill_hyper_q(hyps, hyp_ranges, keys, hyper_q, idx=0):
    """
    Recursive function to load each of the hyperparameter combinations 
    onto a queue.

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
    hyper_q - Queue to hold all parameter sets
    idx - the index of the current key to be searched over
    """
    # Base call, runs the training and saves the result
    if idx >= len(keys):
        if 'exp_num' not in hyps:
            if 'starting_exp_num' not in hyps: hyps['starting_exp_num'] = 0
            hyps['exp_num'] = hyps['starting_exp_num']
        hyps['save_folder'] = hyps['exp_name'] + "/" + hyps['exp_name'] +"_"+ str(hyps['exp_num']) 
        for k in keys:
            hyps['save_folder'] += "_" + str(k)+str(hyps[k])

        # Make model hyps
        hyps['model_class'] = globals()[hyps['model_type']]
        model_hyps = {k:v for k,v in hyps.items()}
        fn_args = set(hyps['model_class'].__init__.__code__.co_varnames) 
        if "kwargs" in fn_args:
            fn_args = fn_args | set(TDRModel.__init__.__code__.co_varnames)
        keys = list(model_hyps.keys())
        for k in keys:
            if k not in fn_args:
                del model_hyps[k]
        
        # Load q
        hyper_q.put([{k:v for k,v in hyps.items()}, {k:v for k,v in model_hyps.items()}])
        hyps['exp_num'] += 1

    # Non-base call. Sets a hyperparameter to a new search value and passes down the dict.
    else:
        key = keys[idx]
        for param in hyp_ranges[key]:
            hyps[key] = param
            hyper_q = fill_hyper_q(hyps, hyp_ranges, keys, hyper_q, idx+1)
    return hyper_q

def get_device(visible_devices, cuda_buffer=3000):
    info = get_cuda_info()
    for i,mem_dict in enumerate(info):
        if i in visible_devices and mem_dict['remaining_mem'] >= cuda_buffer:
            return i
    return -1

def mp_hyper_search(hyps, hyp_ranges, keys, n_workers=4, visible_devices={0,1,2,3,4,5}, cuda_buffer=3000,
                                                    ram_buffer=6000, early_stopping=10, stop_tolerance=.01):
    starttime = time.time()
    # Make results file
    if not os.path.exists(hyps['exp_name']):
        os.mkdir(hyps['exp_name'])
    results_file = hyps['exp_name']+"/results.txt"
    with open(results_file,'a') as f:
        f.write("Hyperparameters:\n")
        for k in hyps.keys():
            if k not in hyp_ranges:
                f.write(str(k) + ": " + str(hyps[k]) + '\n')
        f.write("\nHyperranges:\n")
        for k in hyp_ranges.keys():
            f.write(str(k) + ": [" + ",".join([str(v) for v in hyp_ranges[k]])+']\n')
        f.write('\n')
    
    hyper_q = mp.Queue()
    
    hyper_q = fill_hyper_q(hyps, hyp_ranges, keys, hyper_q, idx=0)
    total_searches = hyper_q.qsize()
    print("n_searches:", total_searches)

    n_workers = min(total_searches, n_workers) # No need to waste resources
    run_q = mp.Queue(n_workers)
    return_q = mp.Queue(n_workers+1)
    workers = []
    procs = []
    print("Initializing workers")
    for i in range(n_workers):
        worker = Trainer(run_q, return_q, early_stopping=early_stopping, stop_tolerance=stop_tolerance)
        workers.append(worker)
        proc = mp.Process(target=worker.loop_training)
        proc.start()
        procs.append(proc)
        
    result_count = 0
    print("Starting Hyperloop")
    while result_count < total_searches:
        print("Running Time:", time.time()-starttime)
        device = get_device(visible_devices, cuda_buffer)
        enough_ram = psutil.virtual_memory().free//1028**2 > ram_buffer
        # must be careful not to threadlock here
        if (not enough_ram or device <= -1) and hyper_q.qsize() >= total_searches:
            print("RAM shortage or no devices available, sleeping for 20s")
            time.sleep(20)
        elif not hyper_q.empty() and not run_q.full():
            hyperset = hyper_q.get()
            hyperset.append(device)
            print("Loading hyperset...")
            run_q.put(hyperset)
            time.sleep(5) # Timer to ensure ram measurements are completed appropriately
            print("Loaded", hyperset[0]["exp_num"])
        else:
            print("Waiting...")
            results = return_q.get()[0]
            print("Collected", results['save_folder'])
            with open(results_file,'a') as f:
                results = " -- ".join([str(k)+":"+str(results[k]) for k in sorted(results.keys())])
                f.write("\n"+results+"\n")
            result_count += 1

    for proc in procs:
        proc.terminate()
        proc.join(timeout=1.0)

def hyper_search(hyps, hyp_ranges, keys, device, early_stopping=10, stop_tolerance=.01):
    starttime = time.time()
    # Make results file
    if not os.path.exists(hyps['exp_name']):
        os.mkdir(hyps['exp_name'])
    results_file = hyps['exp_name']+"/results.txt"
    with open(results_file,'a') as f:
        f.write("Hyperparameters:\n")
        for k in hyps.keys():
            if k not in hyp_ranges:
                f.write(str(k) + ": " + str(hyps[k]) + '\n')
        f.write("\nHyperranges:\n")
        for k in hyp_ranges.keys():
            f.write(str(k) + ": [" + ",".join([str(v) for v in hyp_ranges[k]])+']\n')
        f.write('\n')
    
    hyper_q = Queue()
    
    hyper_q = fill_hyper_q(hyps, hyp_ranges, keys, hyper_q, idx=0)
    total_searches = hyper_q.qsize()
    print("n_searches:", total_searches)

    trainer = Trainer(early_stopping=early_stopping, stop_tolerance=stop_tolerance)

    result_count = 0
    print("Starting Hyperloop")
    while not hyper_q.empty():
        print("Searches left:", hyper_q.qsize(),"-- Running Time:", time.time()-starttime)
        hyperset = hyper_q.get()
        hyperset.append(device)
        results = trainer.train(*hyperset, verbose=True)
        with open(results_file,'a') as f:
            results = " -- ".join([str(k)+":"+str(results[k]) for k in sorted(results.keys())])
            f.write("\n"+results+"\n")






