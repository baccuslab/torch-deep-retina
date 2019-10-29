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
import torchdeepretina.utils as tdrutils
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

DEVICE = torch.device("cuda:0")

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
                print("Caught error",e,"on", train_args[0]['exp_num'], 
                                            "will retry in 100 seconds...")
                sleep(100)

    def get_model_and_distr(self, hyps, model_hyps, train_data):
        """
        hyps: dict
            dict of relevant hyperparameters
        model_hyps: dict
            dict of relevant hyperparameters
        train_data: DataContainer
            a DataContainer of the training data as returned by get_data
        """
        model = hyps['model_class'](**model_hyps)
        model = model.to(hyps['device'])
        num_val = 10000
        seq_len = 1 if not model.recurrent else hyps['recur_seq_len']
        shift_labels = False if 'shift_labels' not in hyps else hyps['shift_labels']
        zscorey = False if 'zscorey' not in hyps else hyps['zscorey']
        data_distr = DataDistributor(train_data, num_val, batch_size=hyps['batch_size'], 
                                        shuffle=hyps['shuffle'], recurrent=model.recurrent, 
                                                seq_len=seq_len, shift_labels=shift_labels, 
                                                                            zscorey=zscorey)
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

    def print_train_update(self, error, grade, l1, model, n_loops, i):
        loss = error + grade + l1
        s = "Loss: {:.5e}".format(loss.item())
        if grade.item() > 0:
            s = "{} | Error: {:.5e} | Grade: {:.5e} | ".format(s, error.item(), grade.item())
        if model.kinetic:
            ps = model.kinetics.named_parameters()
            s += " | " + " | ".join([str(name)+":"+str(round(p.data.item(),4)) for name,p in list(ps)])
        s = "{} | {}/{}".format(s,i,n_loops)
        print(s, end="       \r")

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
                    numer = b*data_distr.val_shape[0] + i
                    denom = data_distr.val_shape[0]*batch_size
                    print("{}/{}".format(numer,denom), end="     \r")
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
            a DataContainer of the training data as returned by get_data
        """
        # Initialize miscellaneous parameters 
        torch.cuda.empty_cache()
        hyps['device'] = device
        batch_size = hyps['batch_size']

        if 'skip_nums' in hyps and hyps['skip_nums'] is not None and\
                                        len(hyps['skip_nums']) > 0 and\
                                        hyps['exp_num'] in hyps['skip_nums']:

            print("Skipping", hyps['save_folder'])
            results = {"save_folder":hyps['save_folder'], 
                                "Loss":None, "ValAcc":None, 
                                "ValLoss":None, "TestPearson":None}
            return results

        # Get Data, Make Model, Record Initial Hyps and Model
        train_data, test_data = get_data(hyps)
        model_hyps["n_units"] = train_data.y.shape[-1]
        model_hyps['centers'] = train_data.centers
        model, data_distr = self.get_model_and_distr(hyps, model_hyps, train_data)
        print("train shape:", data_distr.train_shape)
        print("val shape:", data_distr.val_shape)
        self.record_session(hyps, model)
        teacher = None
        if 'teacher' in hyps and hyps['teacher'] is not None:
            teacher = analysis.read_model_file(hyps['teacher'])
            teacher.to(device)
            teacher.eval()
            #if hyps['teacher_layers'] is not None:

        # Make optimization objects (lossfxn, optimizer, scheduler)
        optimizer, scheduler, loss_fn = get_optim_objs(hyps, model, train_data.centers)
        if 'gauss_reg' in hyps and hyps['gauss_reg'] > 0:
            gauss_reg = tdrutils.GaussRegularizer(model, [0,6], std=hyps['gauss_reg'])

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
                    y,error,grade,hs = recurrent_eval(hyps, x, label, model, hs, loss_fn,
                                                                         teacher=teacher)
                else:
                    y,error,grade = static_eval(hyps, x, label, model, loss_fn, teacher=teacher)
                if hyps['l1']<=0:
                    activity_l1 = torch.zeros(1).to(device)
                else: 
                    activity_l1 = hyps['l1'] * torch.norm(y, 1).float().mean()

                if 'gauss_reg' in hyps and hyps['gauss_reg'] > 0:
                    activity_l1 += hyps['gauss_loss_coef']*gauss_reg.get_loss()

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
            stats_string += 'Avg Loss: {} -- Time: {}\n'.format(avg_loss, time.time()-starttime)
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
                    val_loss, val_preds, val_targs = self.validate_recurrent(hyps, model, 
                                                                    data_distr, hs, loss_fn, 
                                                                    verbose=verbose)
                else:
                    val_loss, val_preds, val_targs = self.validate_static(hyps,model,data_distr, 
                                                                    loss_fn, step_size=step_size,
                                                                    verbose=verbose)

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
                stats_string += "Val Cell Cors:" + " | ".join([str(p) for p in pearsons])+'\n'
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
                stats_string += "Val Cor: {} | Val Loss: {} | Exp Val: {}\n".format(val_acc,
                                                                      val_loss, exp_val_acc)
                scheduler.step(val_loss)
                del val_preds

                # Validation on Test Subset (Static Only)
                avg_pearson = 0
                if not model.recurrent and test_data is not None:
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
                "y_stats":{'mean':data_distr.y_mean, 'std':data_distr.y_std}
            }
            for k in hyps.keys():
                if k not in save_dict:
                    save_dict[k] = hyps[k]
            tdrutils.save_checkpoint(save_dict, hyps['save_folder'], 'test', del_prev=True)

            # Print Epoch Stats
            gc.collect()
            max_mem_used = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            stats_string += "Memory Used: {:.2f} mb".format(max_mem_used / 1024)+"\n"
            print(stats_string)
            # If loss is nan, training is futile
            if math.isnan(avg_loss) or math.isinf(avg_loss) or stop:
                break

        # Final save
        results = {"save_folder":hyps['save_folder'], 
                    "Loss":avg_loss, 
                    "ValAcc":val_acc, 
                    "ValLoss":val_loss, 
                    "TestPearson":avg_pearson}
        with open(hyps['save_folder'] + "/hyperparams.txt",'a') as f:
            s = " ".join([str(k)+":"+str(results[k]) for k in sorted(results.keys())])
            s = "\n" + s + '\n'
            f.write(s)
        return results

def recurrent_eval(hyps, x, label, model, hs, loss_fn, teacher=None):
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

def static_eval(hyps, x, label, model, loss_fn, teacher=None):
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

def get_data(hyps):
    """
    hyps: dict
        dict of relevant hyperparameters
    """
    img_depth, img_height, img_width = hyps['img_shape']
    train_data = DataContainer(loadexpt(hyps['dataset'],hyps['cells'], 
                                hyps['stim_type'],'train',img_depth,0))
    norm_stats = [train_data.stats['mean'], train_data.stats['std']] 

    try:
        test_data = DataContainer(loadexpt(hyps['dataset'],hyps['cells'],hyps['stim_type'],
                                                'test',img_depth,0, norm_stats=norm_stats))
        test_data.X = test_data.X[:500]
        test_data.y = test_data.y[:500]
    except:
        test_data = None
    return train_data, test_data

def get_optim_objs(hyps, model, centers=None):
    """
    hyps: dict
        dict of relevant hyperparameters
    model: torch nn.Module
        the model to be trained
    centers: list of tuples or lists, shape: (n_cells, 2)
        the centers of each ganglion cell in terms of image coordinates
        if None centers is ignored
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

    Returns:
        hyper_q: Queue of lists of dicts [hyps, model_hyps]
    """
    # Base call, runs the training and saves the result
    if idx >= len(keys):
        if 'exp_num' not in hyps:
            if 'starting_exp_num' not in hyps or hyps['starting_exp_num'] is None or\
                                                      hyps['starting_exp_num'] == []:
                hyps['starting_exp_num'] = 0
            hyps['exp_num'] = hyps['starting_exp_num']
        hyps['save_folder'] = hyps['exp_name'] + "/" + hyps['exp_name'] +"_"+ str(hyps['exp_num']) 
        for k in keys:
            hyps['save_folder'] += "_" + str(k)+str(hyps[k])

        hyps['model_class'] = globals()[hyps['model_type']]
        model_hyps = get_model_hyps(hyps)

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
    info = tdrutils.get_cuda_info()
    for i,mem_dict in enumerate(info):
        if i in visible_devices and mem_dict['remaining_mem'] >= cuda_buffer:
            return i
    return -1

def mp_hyper_search(hyps, hyp_ranges, n_workers=4, visible_devices={0,1,2,3,4,5}, 
                                                      cuda_buffer=3000, ram_buffer=6000, 
                                                      early_stopping=10, stop_tolerance=.01):
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
    hyper_q = fill_hyper_q(hyps, hyp_ranges, list(hyp_ranges.keys()), hyper_q, idx=0)
    total_searches = hyper_q.qsize()
    print("n_searches:", total_searches)

    n_workers = min(total_searches, n_workers) # No need to waste resources
    run_q = mp.Queue(n_workers)
    return_q = mp.Queue(n_workers+1)
    workers = []
    procs = []
    print("Initializing workers")
    for i in range(n_workers):
        worker = Trainer(run_q, return_q, early_stopping=early_stopping, 
                                            stop_tolerance=stop_tolerance)
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
                s = " | ".join([str(k)+":"+str(results[k]) for k in sorted(results.keys())])
                f.write("\n"+s+"\n")
            result_count += 1

    for proc in procs:
        proc.terminate()
        proc.join(timeout=1.0)

def hyper_search(hyps, hyp_ranges, device, early_stopping=10, stop_tolerance=.01):
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
    hyper_q = fill_hyper_q(hyps, hyp_ranges, list(hyp_ranges.keys()), hyper_q, idx=0)
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

def get_model_hyps(hyps):
    hyps['model_class'] = globals()[hyps['model_type']]
    model_hyps = {k:v for k,v in hyps.items()}

    fn_args = set(hyps['model_class'].__init__.__code__.co_varnames) 
    if "kwargs" in fn_args:
        fn_args = fn_args | set(TDRModel.__init__.__code__.co_varnames)
    keys = list(model_hyps.keys())
    for k in keys:
        if k not in fn_args:
            del model_hyps[k]
    return model_hyps

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

class LossFxnWrapper:
    def __init__(self, loss_fn, hyps, centers):
        self.centers = centers
        self.loss_fn = loss_fn
        self.coords = self.convert_coords(hyps, centers)

    def convert_coords(self, hyps, centers):
        """
        Assumes a stride of 1 with 0 padding in each layer.
        """
        # Each quantity is even, thus the final half_effective_ksize is odd
        half_effective_ksize = (hyps['ksizes'][0]-1) + (hyps['ksizes'][1]-1) +\
                                                    (hyps['ksizes'][2]//2-1) + 1
        coords = []
        for center in centers:
            row = min(max(0,center[0]-half_effective_ksize), 
                            hyps['img_shape'][1]-2*(half_effective_ksize-1))
            col = min(max(0,center[1]-half_effective_ksize), 
                            hyps['img_shape'][2]-2*(half_effective_ksize-1))
            coords.append([row,col])
        return torch.LongTensor(coords)

    def __call__(self, x, y):
        idxs = torch.arange(x.shape[-4]).long()
        chans = torch.arange(x.shape[-3]).long()
        units = x[...,:,chans,self.coords[:,0],self.coords[:,1]]
        return self.loss_fn(units, y)


def fit_sta(test_chunk, chunked_data, normalize=True):
    """
    Calculates the STA from the chunked data and returns a cpu torch tensor of the STA.
    This function is mainly used for creating figures in deepretina paper.

    test_chunk: int
        the held out test data index
    chunked_data: ChunkedData object (see datas.py)
        An object to handle chunking the data for cross validation
    """
    if normalize:
        norm_stats = chunked_data.get_norm_stats(test_chunk)
        mean, std = norm_stats
    n_samples = 0
    cumu_sum = 0
    batch_size = 500
    for i in range(chunked_data.n_chunks):
        if i != test_chunk:
            x = chunked_data.X[chunked_data.chunks[i]]
            y = chunked_data.y[chunked_data.chunks[i]]
            for j in range(0,len(x),batch_size):
                temp_x = x[j:j+batch_size]
                if normalize:
                    temp_x = (temp_x-mean)/(std+1e-5)
                matmul = torch.einsum("ij,i->j",temp_x.to(DEVICE),y[j:j+batch_size].to(DEVICE))
                cumu_sum = cumu_sum + matmul.cpu()
                n_samples += len(temp_x)
    if normalize:
        return cumu_sum/n_samples, norm_stats
    return cumu_sum/n_samples

def fit_nonlin(chunked_data, test_chunk, model, degree=5,n_repeats=10, ret_all=False):
    """
    Fits a polynomial nonlinearity to the model.

    chunked_data - ChunkedData object (see datas.py)
    test_chunk - int
        the chunk of data that will be held for testing
    model - RevCorLN object (see models.py)
    degree - int
        degree of polynomial to fit
    n_repeats - int
        number of repeated trials to fit the nonlinearity
    ret_all - bool
        will return multiple fields if true
    """
    if type(degree)==type(int()):
        degree=[degree]
    best_r = -1
    try:
        X,y = chunked_data.get_train_data(test_chunk)
        X = model.normalize(X)
        lin_outs = model.convolve(X)
    except:
        batch_size = 500
        n_samples = np.sum([len(chunk) if i != test_chunk else 0 for i,chunk in\
                                                enumerate(chunked_data.chunks)])
        lin_outs = torch.empty(n_samples).float()
        truth = torch.empty(n_samples).float()
        outdx = 0
        for i in range(chunked_data.n_chunks):
            if i != test_chunk:
                x = chunked_data.X[chunked_data.chunks[i]]
                x = model.normalize(x)
                y = chunked_data.y[chunked_data.chunks[i]]
                for j in range(0,len(x),batch_size):
                    temp_x = x[j:j+batch_size]
                    outs = model.convolve(temp_x).cpu()
                    lin_outs[outdx:outdx+len(outs)] = outs
                    truth[outdx:outdx+len(outs)] = y[j:j+batch_size]
                    outdx += len(outs)
        y = truth

    for d in degree:
        lin_outs = lin_outs.numpy().astype(np.float32).squeeze()
        y = y.numpy().astype(np.float32).squeeze()
        fit = np.polyfit(lin_outs, y, d)
        poly = tdrutils.poly1d(fit)
        preds = poly(lin_outs)
        r = tdrutils.pearsonr(preds, y)
        if r > best_r:
            best_poly = poly
            best_degree=d
            best_r = r
            best_preds = preds

    if ret_all:
        return best_poly, best_degree, best_r, best_preds
    return best_poly

def fit_ln_nonlin(X, y, model, degree=5, ret_all=False):
    """
    Fits a polynomial nonlinearity to the model.

    X: torch tensor (T, C) or (T, C, H, W)
    y: torch tensor
    model: RevCorLN object (see models.py)
    degree: int
        degree of polynomial to fit
    ret_all: bool
        will return multiple fields if true
    """
    if type(degree)==type(int()):
        degree=[degree]
    best_r = -1
    try:
        X = model.normalize(X)
        lin_outs = model.convolve(X)
    except:
        batch_size = 500
        lin_outs = torch.empty(len(X)).float()
        for i in range(0, len(X), batch_size):
            x = X[i:i+batch_size]
            x = model.normalize(x)
            outs = model.convolve(x).cpu().detach()
            lin_outs[i:i+batch_size] = outs

    for d in degree:
        lin_outs = lin_outs.numpy().astype(np.float32).squeeze()
        y = y.numpy().astype(np.float32).squeeze()
        fit = np.polyfit(lin_outs, y, d)
        poly = tdrutils.poly1d(fit)
        preds = poly(lin_outs)
        r = tdrutils.pearsonr(preds, y)
        if r > best_r:
            best_poly = poly
            best_degree=d
            best_r = r
            best_preds = preds
    if ret_all:
        return best_poly, best_degree, best_r, best_preds
    return best_poly

def train_ln(X, y, rf_center, cutout_size):
    """
    Fits an LN model to the data

    X: torch FloatTensor (T,C) or (T,C,H,W)
        training stimulus
    y: torch FloatTensor (T)
    rf_center: list or tuple (row, col)
        the receptive field center of the cell
    cutout_size: int
        stimulus is cutout centered at argued center
    """
    if type(X) == type(np.array([])):
        X = torch.FloatTensor(X)
    if type(y) == type(np.array([])):
        y = torch.FloatTensor(y)

    # STA is cpu torch tensor
    sta, norm_stats, _ = tdrutils.revcor(X, y, ret_norm_stats=True)
    model = RevCorLN(sta.reshape(-1),ln_cutout_size=cutout_size,center=rf_center,
                                                            norm_stats=norm_stats)
    bests = fit_ln_nonlin(X, y, model,degree=[5], ret_all=True)
    best_poly,best_degree,best_r,best_preds = bests
    model.poly = best_poly # Torch compatible polys
    model.poly_degree = best_degree

    return model
    

def cross_validate_ln(chunked_data, ln_cutout_size, center, ret_models=True, skip_chunks={}, 
                                                                                verbose=True):
    """
    Performs cross validation for LN model trained using reverse correlation

    chunked_data: ChunkedData object (see datas.py)
        This is an object that segregates the data into N distinct chunks
    ln_cutout_size: int
        the the size of the window to train on the stimulus
    center: list or tuple of ints (row,col)
        the center coordinate of the receptive field for the cell
    ret_models: bool
        if true, the models are each collected and returned at the end of the 
        cross validation
    skip_chunks: set or list
        chunk indices to skip during training
    """
    accs = []
    models = []
    if verbose:
        print("Fitting LNs...")
    for i in range(chunked_data.n_chunks):
        basetime=time.time()

        # STA is cpu torch tensor
        sta,norm_stats = fit_sta(i,chunked_data,normalize=True)

        model = RevCorLN(sta,ln_cutout_size=ln_cutout_size,center=center,norm_stats=norm_stats)
        bests = fit_nonlin(chunked_data,i,model,degree=[5],ret_all=True)
        best_poly,best_degree,best_r,best_preds = bests
        model.poly = best_poly # Torch compatible polys
        model.poly_degree = best_degree

        val_X = chunked_data.X[chunked_data.chunks[i]]
        val_y = chunked_data.y[chunked_data.chunks[i]]
        val_X = model.normalize(val_X)
        preds = model(val_X).squeeze()
        r = tdrutils.pearsonr(preds.squeeze(), val_y.squeeze()).item()
        accs.append(r)
        if ret_models:
            models.append(model)
        exec_time = time.time()-basetime
        if verbose:
            print("Fit Trial:",i, ", Best Degree:",best_degree, ", Acc:", r, 
                                                        ", Time:", exec_time)
        del val_X
        del val_y
    return models, accs

