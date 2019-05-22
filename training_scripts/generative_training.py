"""
Use this script to create a distribution that maximally diverges the outputs of two models. 
Can give command line arguments to specify which json files to use for parameter specification.

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
from torch.nn import PoissonNLLLoss, MSELoss
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
from torchdeepretina.miscellaneous import freeze_weights
from torchdeepretina.generative_models import StimGenerator
from torchdeepretina.loss_funcs import MinMax
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

def train(model1, model2, gen_model, hyps, model_hyps):
    exp_folder = os.path.join(".", hyps['exp_name'])
    if not os.path.exists(exp_folder):
        os.mkdir(exp_folder)
    _, sub_ds, _ = next(os.walk(exp_folder))
    largest_num = -1
    for d in sub_ds:
        try:
            num = int(d.split("_")[-1])
            if num > largest_num: 
                largest_num = num
        except:
            pass
    hyps['exp_num'] = largest_num + 1
    hyps['save_folder'] = hyps['exp_name'] + "/" + hyps['exp_name'] +"_"+ str(hyps['exp_num']) 
    SAVE = hyps['save_folder']
    if not os.path.exists(SAVE):
        os.mkdir(SAVE)
    print("Saving to:", SAVE)
    LR = hyps['lr']
    LAMBDA2 = hyps['l2']
    EPOCHS = hyps['n_epochs']
    batch_size = hyps['batch_size']
    model1.to(DEVICE)
    model2.to(DEVICE)
    gen_model.to(DEVICE)

    with open(SAVE + "/hyperparams.txt",'w') as f:
        f.write(str(gen_model)+'\n')
        for k in sorted(hyps.keys()):
            f.write(str(k) + ": " + str(hyps[k]) + "\n")

    print(gen_model)

    lossfxn = globals()[hyps['lossfxn']]()
    freeze_weights(model1)
    freeze_weights(model2)
    optimizer = torch.optim.Adam(gen_model.parameters(), lr = LR, weight_decay = LAMBDA2)

    # Train Loop
    gen_model.train(mode=True)
    for epoch in range(EPOCHS):
        starttime = time.time()
        print('Epoch ' + str(epoch))  
        
        optimizer.zero_grad()

        gen_img = gen_model.generate_img(batch_size)
        y1 = model1(gen_img)
        y2 = model2(gen_img)
        loss = -lossfxn(y1, y2) # Negative to maximize
        loss.backward()
        optimizer.step()

        avg_loss = loss.item()
        print('Avg Loss: ' + str(avg_loss), " - exec time:", time.time() - starttime)
        # If loss is nan, training is futile
        if math.isnan(avg_loss) or math.isinf(avg_loss):
            break
    
    save_dict = {
        "model_hyps": model_hyps,
        "model_state_dict":gen_model.state_dict(),
        "optim_state_dict":optimizer.state_dict(),
        "loss": avg_loss,
        "epoch":epoch
    }
    io.save_checkpoint_dict(save_dict,SAVE,'test')
    results = {"Loss":avg_loss}
    with open(SAVE + "/hyperparams.txt",'a') as f:
        f.write("\n" + " ".join([k+":"+str(results[k]) for k in sorted(results.keys())]) + '\n')
    examples = gen_model.generate_img(50).cpu().detach().numpy()
    np.save(os.path.join(SAVE, "examples.npy"), examples)
    return results

def load_json(file_name):
    with open(file_name) as f:
        s = f.read()
        j = json.loads(s)
    return j

def get_generative_model(hyps):
    """
    Creates the generative model using the hyperparameters dict.
    """
    model_class = globals()[hyps['model_type']]
    model_hyps = {}
    for key in hyps.keys():
        model_hyps[key] = hyps[key]
    fn_args = set(model_class.__init__.__code__.co_varnames)
    keys = list(model_hyps.keys())
    for k in keys:
        if k not in fn_args:
            del model_hyps[k]
    return model_class(**model_hyps), model_hyps

def get_hyps(folder):
    hyps = dict()
    with open(os.path.join(folder, "hyperparams.txt")) as f:
        for line in f:
            if "(" not in line and ")" not in line:
                splt = line.strip().split(":")
                if len(splt) > 1:
                    hyps[splt[0]] = splt[1].strip()
    return hyps

def load_gen_model(folder, data):
    """
    Loads the model saved in the argued save_file.

    save_file - string
        the should be the direct path to a model checkpoint.
    """
    try:
        hyps = get_hyps(folder)
        if "<" in hyps['model_type']:
            hyps['model_type'] = hyps['model_type'].split(".")[-1].split("\'")[0].strip()
        hyps['model_type'] = globals()[hyps['model_type']]
        model = hyps['model_type'](**data['model_hyps'])
    except Exception as e:
        model_hyps = {"z_size":int(hyps["z_size"]),
                        "bnorm":bool(hyps['bnorm']),
                        "out_depths":None}
        model = hyps['model_type'](**model_hyps)
    return model

def load_model(folder, pth):
    try:
        hyps=get_hyps(folder)
        hyps['model_type'] = hyps['model_type'].split(".")[-1].split("\'")[0].strip()
        hyps['model_type'] = globals()[hyps['model_type']]
        model = hyps['model_type'](**pth['model_hyps'])
    except Exception as e:
        model_hyps = {"n_units":5,"noise":float(hyps['noise'])}
        if "bias" in hyps:
            model_hyps['bias'] = hyps['bias'] == "True"
        if "chans" in hyps:
            model_hyps['chans'] = [int(x) for x in 
                                   hyps['chans'].replace("[", "").replace("]", "").strip().split(",")]
        if "adapt_gauss" in hyps:
            model_hyps['adapt_gauss'] = hyps['adapt_gauss'] == "True"
        if "linear_bias" in hyps:
            model_hyps['linear_bias'] = hyps['linear_bias'] == "True"
        fn_args = set(hyps['model_type'].__init__.__code__.co_varnames)
        for k in model_hyps.keys():
            if k not in fn_args:
                del model_hyps[k]
        model = hyps['model_type'](**model_hyps)
    return model

def read_model(folder):
    for i in range(1000):
        file = os.path.join(folder.strip(),"test_epoch_{0}.pth".format(i))
        try:
            with open(file, "rb") as fd:
                data = torch.load(fd)
        except Exception as e:
            break
    try:
        model = data['model']
    except Exception as e:
        model = load_model(folder, data)

    try:
        model.load_state_dict(data['model_state_dict'])
    except RuntimeError as e:
        keys = list(data['model_state_dict'].keys())
        for key in keys:
            if "cuda_param" in key:
                new_key = key.replace("cuda_param", "sigma")
                data['model_state_dict'][new_key] = data['model_state_dict'][key]
                del data['model_state_dict'][key]
        model.load_state_dict(data['model_state_dict'])
    return model

if __name__ == "__main__":
    hyperparams_file = "hyps/generative_hyperparams.json"
    hyperranges_file = 'hyps/hyperranges.json'
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
    hyps = load_json(hyperparams_file)
    sleep_time = 8
    print("You have "+str(sleep_time)+" seconds to cancel experiment name "+ hyps['exp_name'])
    time.sleep(sleep_time)
    
    model1 = read_model(hyps['model1'])
    #model1 = nn.Sequential(*model1.sequential[:-2])
    model2 = read_model(hyps['model2'])
    #model2 = nn.Sequential(*model2.sequential[:-2])
    noises = [0.05, 0.1, 0.2]
    for noise in noises:
        hyps['noise'] = noise
        gen_model, model_hyps = get_generative_model(hyps)
        train(model1, model2, gen_model, hyps, model_hyps)

