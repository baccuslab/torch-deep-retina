from scipy.stats import pearsonr
import os
import sys
from time import sleep
import numpy as np
import torch
import torch.nn as nn
from torch.nn import PoissonNLLLoss, MSELoss
import torch.nn.functional as F
import os.path as path
from torchdeepretina.utils import get_cuda_info, save_checkpoint
from torchdeepretina.datas import loadexpt, DataContainer, DataDistributor
from torchdeepretina.models import *
import torchdeepretina.analysis as analysis
import time
from tqdm import tqdm
import math
import torch.multiprocessing as mp
from queue import Queue
from collections import deque
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

    def get_data(self, hyps):
        """
        hyps: dict
            dict of relevant hyperparameters
        """
        img_depth, img_height, img_width = hyps['img_shape']
        train_data = DataContainer(loadexpt(hyps['dataset'],hyps['cells'], hyps['stim_type'],'train',img_depth,0))
        norm_stats = [train_data.stats['mean'], train_data.stats['std']] 

        test_data = DataContainer(loadexpt(hyps['dataset'],hyps['cells'],hyps['stim_type'],'test',img_depth,0, 
                                                                            norm_stats=norm_stats))
        test_data.X = test_data.X[:500]
        test_data.y = test_data.y[:500]
        return train_data, test_data

    def get_model_and_distr(self, hyps, model_hyps, train_data):
        """
        hyps: dict
            dict of relevant hyperparameters
        model_hyps: dict
            dict of relevant hyperparameters
        train_data: DataContainer
            a DataContainer of the training data as returned by self.get_data
        """
        model = hyps['model_class'](**model_hyps)
        model = model.to(hyps['device'])
        num_val = 10000
        seq_len = 1 if not model.recurrent else hyps['recur_seq_len']
        data_distr = DataDistributor(train_data, num_val, batch_size=hyps['batch_size'], shuffle=hyps['shuffle'], 
                                                            recurrent=model.recurrent, seq_len=seq_len)
        data_distr.torch()
        return model, data_distr

    def record_session(self, hyps, model):
        """
        hyps: dict
            dict of relevant hyperparameters
        model: torch nn.Module
            the model to be trained
        """
        if not os.path.exists(hyps['save_folder']):
            os.mkdir(hyps['save_folder'])
        with open(os.path.join(hyps['save_folder'],"hyperparams.txt"),'w') as f:
            f.write(str(model)+'\n')
            for k in sorted(hyps.keys()):
                f.write(str(k) + ": " + str(hyps[k]) + "\n")
        with open(os.path.join(hyps['save_folder'],"hyperparams.json"),'w') as f:
            temp_hyps = {k:v for k,v in hyps.items()}
            del temp_hyps['model_class']
            json.dump(temp_hyps, f)

    def get_optim_objs(self, hyps, model):
        """
        hyps: dict
            dict of relevant hyperparameters
        model: torch nn.Module
            the model to be trained
        """
        if 'lossfxn' not in hyps:
            hyps['lossfxn'] = "PoissonNLLLoss"
        if hyps['lossfxn'] == "PoissonNLLLoss" and 'log_poisson' in hyps:
            loss_fn = globals()[hyps['lossfxn']](log_input=hyps['log_poisson'])
        else:
            loss_fn = globals()[hyps['lossfxn']]()
        optimizer = torch.optim.Adam(model.parameters(), lr = hyps['lr'], weight_decay = hyps['l2'])
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor = 0.1)
        return optimizer, scheduler, loss_fn

    def get_hs(self, hyps, model, batch_size=None):
        """
        hyps: dict
            dict of relevant hyperparameters
        model: torch nn.Module
            the model to be trained
        batch_size: int
        """
        hs = None
        device = hyps['device']
        if model.recurrent:
            if batch_size is None:
                batch_size = hyps['batch_size']
            hs = [torch.zeros(batch_size, *h).to(device) for h in model.h_shapes]
            if model.kinetic:
                hs[0][:,0] = hyps['intl_pop']
                hs[1] = deque([],maxlen=hyps['recur_seq_len'])
                for i in range(hyps['recur_seq_len']):
                    hs[1].append(torch.zeros(batch_size, *model.h_shapes[1]).to(device))
        return hs

    def recurrent_eval(self, hyps, x, label, model, hs, loss_fn, teacher=None):
        """
        hyps: dict
            dict of relevant hyperparameters
        x: torch FloatTensor
            a batch of the training data
        label: torch FloatTensor
            a batch of the training labels
        model: torch nn.Module
            the model to be trained
        hs: list
            the hidden states of the recurrent model as obtained through self.get_hs
        """
        hs_out = hs
        batch_size = hyps['batch_size']
        ys = []
        answers = []
        device = hyps['device']
        teacher_device = device
        if teacher is None:
            teacher = lambda x: None
            teacher_device = torch.device("cpu")
        for ri in range(hyps['recur_seq_len']):
            ins = x[:,ri]
            y, hs_out = model(ins.to(device), hs_out)
            ys.append(y.view(batch_size, 1, label.shape[-1]))
            with torch.no_grad():
                ans = teacher(x.to(teacher_device))
            answers.append(ans)
            if ri == 0 and not hyps['reset_hs']:
                if model.kinetic:
                    hs[0] = hs_out[0].data.clone()
                    hs[1] = deque([h.data.clone() for h in hs[1]], maxlen=hyps['recur_seq_len'])
                else:
                    hs = [h.data.clone() for h in hs_out]
        y = torch.cat(ys, dim=1)
        error = loss_fn(y,label)/hyps['recur_seq_len']

        # Teacher
        if answers[0] is not None:
            answers = torch.cat(answers, dim=1)
            grade = hyps['teacher_coef']*F.mse_loss(y,answers.data)/hyps['recur_seq_len']
            hyps['teacher_coef'] *= hyps['teacher_decay']
        else:
            grade = torch.zeros(1).to(device)

        # Kinetics
        if hyps['model_type'] == "KineticsModel" and model.kinetics.kfr + model.kinetics.ksi > 1:
            error += ((1-(model.kinetics.kfr+model.kinetics.ksi))**2).mean()

        return y, error, grade, hs

    def static_eval(self, hyps, x, label, model, loss_fn, teacher=None):
        """
        hyps: dict
            dict of relevant hyperparameters
        x: torch FloatTensor
            a batch of the training data
        label: torch FloatTensor
            a batch of the training labels
        model: torch nn.Module
            the model to be trained
        teacher: torch nn.Module
            optional teacher network
        """
        device = hyps['device']
        y = model(x.to(device))
        error = loss_fn(y,label)
        if teacher is not None:
            with torch.no_grad():
                ans = teacher(x.to(device))
            grade = F.mse_loss(y,ans.data)
        else:
            grade = torch.zeros(1).to(device)
        return y,error,grade

    def print_train_update(self, error, grade, l1, model, n_loops, i):
        loss = error + grade + l1
        s = "Loss: {:.5e} – ".format(loss.item())
        if grade.item() > 0:
            s += "Error: {:.5e} – Grade: {:.5e} – ".format(error.item(), grade.item())
        if model.kinetic:
            ps = model.kinetics.named_parameters()
            s += " – ".join([str(name)+":"+str(round(p.data.item(),4)) for name,p in list(ps)]) + " – "
        print(s, i, "/", n_loops, end="       \r")

    def validate_recurrent(self, hyps, model, data_distr, hs, loss_fn, verbose=False):
        """
        hyps: dict
            dict of relevant hyperparameters
        model: torch nn.Module
            the model to be trained
        data_distr: DataDistributor
            the data distribution object as obtained through self.get_model_and_distr
        hs: list
            the hidden states of the recurrent model as obtained through self.get_hs
        loss_fn: loss function
            the loss function
        """
        val_preds = []
        val_targs = []
        val_loss = 0
        batch_size = hyps['batch_size']
        n_loops = data_distr.val_shape[0]*batch_size
        step_size = 1
        device = hyps['device']
        for b in range(batch_size):
            for i, (val_x, val_y) in enumerate(data_distr.val_sample(step_size)):
                # val_x (batch_size, seq_len, depth, height, width)
                # val_y (batch_size, seq_len, n_ganlion)
                outs, hs = model(val_x[b:b+1,0].to(device), hs)
                val_y = val_y[:,0]
                val_preds.append(outs[:,None,:].cpu().detach().numpy())
                val_targs.append(val_y[b:b+1,None,:].cpu().detach().numpy())
                val_loss += loss_fn(outs, val_y.to(device)).item()
                if hyps['l1'] > 0:
                    val_loss += (hyps['l1'] * torch.norm(outs, 1).float()/outs.shape[0]).item()
                if verbose and i%(n_loops//10) == 0:
                    print("{}/{}".format(b*data_distr.val_shape[0] + i,data_distr.val_shape[0]*batch_size), end="     \r")
        return val_loss, val_preds, val_targs

    def validate_static(self, hyps, model, data_distr, loss_fn, step_size=500, verbose=False):
        """
        hyps: dict
            dict of relevant hyperparameters
        model: torch nn.Module
            the model to be trained
        data_distr: DataDistributor
            the data distribution object as obtained through self.get_model_and_distr
        loss_fn: loss function
            the loss function
        step_size: int
            optional size of batches when evaluating validation set
        """
        val_preds = []
        val_targs = []
        val_loss = 0
        n_loops = data_distr.val_shape[0]//step_size
        device = hyps['device']
        for i, (val_x, val_y) in enumerate(data_distr.val_sample(step_size)):
            outs = model(val_x.to(device)).detach()
            val_preds.append(outs.cpu().detach().numpy())
            val_targs.append(val_y.cpu().detach().numpy())
            val_loss += loss_fn(outs, val_y.to(device)).item()
            if hyps['l1'] > 0:
                val_loss += (hyps['l1'] * torch.norm(outs, 1).float()/outs.shape[0]).item()
            if verbose and i%(n_loops//10) == 0:
                print("{}/{}".format(i*step_size,data_distr.val_y.shape[0]), end="     \r")
        return val_loss, val_preds, val_targs

    def train(self, hyps, model_hyps, device, verbose=False):
        """
        hyps: dict
            dict of relevant hyperparameters
        model_hyps: dict
            dict of relevant hyperparameters
        model: torch nn.Module
            the model to be trained
        train_data: DataContainer
            a DataContainer of the training data as returned by self.get_data
        """
        # Initialize miscellaneous parameters 
        torch.cuda.empty_cache()
        hyps['device'] = device
        batch_size = hyps['batch_size']
        if 'skip_nums' in hyps and len(hyps['skip_nums']) > 0 and hyps['exp_num'] in hyps['skip_nums']:
            print("Skipping", hyps['save_folder'])
            results = {"save_folder":hyps['save_folder'], "Loss":None, "ValAcc":None, "ValLoss":None, "TestPearson":None}
            return results

        # Get Data, Make Model, Record Initial Hyps and Model
        train_data, test_data = self.get_data(hyps)
        model_hyps["n_units"] = train_data.y.shape[-1]
        model, data_distr = self.get_model_and_distr(hyps, model_hyps, train_data)
        self.record_session(hyps, model)
        teacher = None
        if 'teacher' in hyps and hyps['teacher'] is not None:
            teacher = analysis.read_model_file(hyps['teacher'])
            teacher.to(device)
            teacher.eval()
            #if hyps['teacher_layers'] is not None:

        # Make optimization objects (lossfxn, optimizer, scheduler)
        optimizer, scheduler, loss_fn = self.get_optim_objs(hyps, model)

        # Training
        for epoch in range(hyps['n_epochs']):
            print("Beginning Epoch", epoch, " -- ", hyps['save_folder'])
            print()
            n_loops = data_distr.n_loops
            hs = self.get_hs(hyps, model)
            model.train(mode=True)
            epoch_loss = 0
            stats_string = 'Epoch ' + str(epoch) + " -- " + hyps['save_folder'] + "\n"
            starttime = time.time()

            # Train Loop
            for i,(x,label) in enumerate(data_distr.train_sample()):
                optimizer.zero_grad()
                label = label.float().to(device)

                # Error Evaluation
                if model.recurrent:
                    y,error,grade,hs = self.recurrent_eval(hyps, x, label, model, hs, loss_fn, teacher=teacher)
                else:
                    y,error,grade = self.static_eval(hyps, x, label, model, loss_fn, teacher=teacher)
                activity_l1 = torch.zeros(1).to(device) if hyps['l1']<=0 else hyps['l1'] * torch.norm(y, 1).float().mean()

                # Backwards Pass
                loss = error + activity_l1 + grade
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                if verbose:
                    self.print_train_update(error, grade, activity_l1, model, n_loops, i)
                if math.isnan(epoch_loss) or math.isinf(epoch_loss) or hyps['exp_name']=="test":
                    break

            # Clean Up Train Loop
            avg_loss = epoch_loss/n_loops
            stats_string += 'Avg Loss: ' + str(avg_loss) + " -- exec time:"+ str(time.time() - starttime) +"\n"
            del x
            del y
            del label

            # Validation
            model.eval()
            with torch.no_grad():

                # Miscellaneous Initializations
                if model.recurrent:
                    step_size = 1
                    n_loops = data_distr.val_shape[0]*batch_size
                    hs = self.get_hs(hyps, model, batch_size=step_size)
                else:
                    step_size = 500
                    n_loops = data_distr.val_shape[0]//step_size
                if verbose:
                    print()
                    print("Validating")

                # Validation Block
                if model.recurrent:
                    val_loss, val_preds, val_targs = self.validate_recurrent(hyps, model, data_distr, hs,
                                                                                loss_fn, verbose=verbose)
                else:
                    val_loss, val_preds, val_targs = self.validate_static(hyps, model, data_distr, loss_fn,
                                                                       step_size=step_size, verbose=verbose)

                # Validation Evaluation
                val_loss = val_loss/n_loops
                n_units = data_distr.val_y.shape[-1]
                if model.recurrent:
                    val_preds = np.concatenate(val_preds, axis=1).reshape((-1,n_units))
                    val_targs = np.concatenate(val_targs, axis=1).reshape((-1,n_units))
                else:
                    val_preds = np.concatenate(val_preds, axis=0)
                    val_targs = np.concatenate(val_targs, axis=0)
                pearsons = []
                for cell in range(val_preds.shape[-1]):
                    pearsons.append(pearsonr(val_preds[:, cell], val_targs[:,cell])[0])
                stats_string += "Val Cell Pearsons:" + " - ".join([str(p) for p in pearsons])+'\n'
                val_acc = np.mean(pearsons)
                stop = self.stop_early(val_acc)

                # Exp Validation
                exp_val_acc = None
                if hyps["log_poisson"] and hyps['softplus'] and not model.infr_exp:
                    val_preds = np.exp(val_preds)
                    for cell in range(val_preds.shape[-1]):
                        pearsons.append(pearsonr(val_preds[:, cell], val_targs[:,cell])[0])
                    exp_val_acc = np.mean(pearsons)

                # Clean Up
                stats_string += "Avg Val Pearson: {} -- Val Loss: {} -- Exp Val: {}\n".format(val_acc, val_loss, exp_val_acc)
                scheduler.step(val_loss)
                del val_preds

                # Validation on Test Subset (Static Only)
                avg_pearson = 0
                if not model.recurrent:
                    test_x = torch.from_numpy(test_data.X)
                    test_obs = model(test_x.to(device)).cpu().detach().numpy()
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

            # Save Model Snapshot
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
            save_checkpoint(save_dict, hyps['save_folder'], 'test', del_prev=True)

            # Print Epoch Stats
            gc.collect()
            max_mem_used = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            stats_string += "Memory Used: {:.2f} mb".format(max_mem_used / 1024)+"\n"
            print(stats_string)
            # If loss is nan, training is futile
            if math.isnan(avg_loss) or math.isinf(avg_loss) or stop:
                break

        # Final save
        results = {"save_folder":hyps['save_folder'], "Loss":avg_loss, "ValAcc":val_acc, "ValLoss":val_loss, "TestPearson":avg_pearson}
        with open(hyps['save_folder'] + "/hyperparams.txt",'a') as f:
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

class ModulePackage:
    def __init__(self, model, layer_keys):
        self.model = model
        self.layer_keys = set(layer_keys)
        self.layers = dict()
        self.handles = []
        for name, mod in model.named_modules():
            if name in self.layer_keys:
                self.handles.append(mod.register_forward_hook(self.get_hook(name)))

    def get_hook(self, layer_key):
        def hook(module, inp, out):
            self.layers[layer_key] = out
        return hook

    def remove_hooks(self):
        for h in self.handles:
            h.remove()
        self.handles = []

    def detach_all(self):
        self.remove_hooks()
        for k in self.layers.keys():
            try:
                self.layers[k].detach().cpu()
                del self.layers[k]
            except:
                pass

    def __call__(self, x):
        return self.model(x)

