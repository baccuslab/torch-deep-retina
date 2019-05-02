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
from models.Dales_Hybrid import DalesHybrid
from models.practical_BN_CNN import PracticalBNCNN
from models.AbsBN_BN_CNN import AbsBNBNCNN
import retio as io
import argparse
import time
from tqdm import tqdm

from deepretina.experiments import loadexpt

# Constants
DEVICE = torch.device("cuda:0")

# Random Seeds (5 is arbitrary)
seed = 3
np.random.seed(seed)
torch.manual_seed(seed)


# Load data using Lane and Nirui's dataloader
train_data = loadexpt('15-11-21b',[0, 1, 3, 5, 8, 9, 13, 14, 16, 17, 18, 20, 21, 22, 23, 24, 25],'naturalscene','train',40,0)
test_data = loadexpt('15-11-21b',[0, 1, 3, 5, 8, 9, 13, 14, 16, 17, 18, 20, 21, 22, 23, 24, 25],'naturalscene','test',40,0)
val_split = 0.005

def train(hyps, epochs=250, batch_size=5000, LR=1e-3, l1_scale=1e-4, l2_scale=1e-2, b1_scale=5, shuffle=True, save='./checkpoints', noise=.1):
    if not os.path.exists(save):
        os.mkdir(save)
    LAMBDA1 = l1_scale
    LAMBDA2 = l2_scale
    EPOCHS = epochs
    BATCH_SIZE = batch_size

    # Model
    #model = model_class()
    #model = CNN(bias=False)
    #model = SSCNN(scale=True, shift=False, bias=True)
    #model = DalesHybrid(bias=True, neg_p=hyps['neg_p'], noise=noise)
    model = AbsBNBNCNN(17, noise=noise) # Uses dropout
    print(model)
    model = model.to(DEVICE)

    with open(save + "/hyperparams.txt",'w') as f:
        f.write(str(model)+'\n')
        for k in sorted(hyps.keys()):
            f.write(str(k) + ": " + str(hyps[k]) + "\n")

    loss_fn = torch.nn.PoissonNLLLoss()
    optimizer = torch.optim.Adam(model.parameters(),lr = LR, weight_decay = LAMBDA2)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor = 0.2)

    # train data
    epoch_tv_x = torch.FloatTensor(train_data.X)
    epoch_tv_y = torch.FloatTensor(train_data.y)

    # train/val split
    num_val = 30000
    epoch_train_x = epoch_tv_x[num_val//2:-num_val//2]
    epoch_train_y = epoch_tv_y[num_val//2:-num_val//2]
    epoch_val_x = torch.cat([epoch_tv_x[:num_val//2], epoch_tv_x[-num_val//2:]], dim=0)
    epoch_val_y = torch.cat([epoch_tv_y[:num_val//2], epoch_tv_y[-num_val//2:]], dim=0)
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
            loss_b1 = b1* (torch.sum(torch.max(model.sequential[2].scale) - torch.min(model.sequential[2].scale))
                + torch.sum(torch.max(model.sequential[8].scale) - torch.min(model.sequential[8].scale)))
            error = loss_fn(y,label)
            loss = error + activity_l1 + loss_b1
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
        del x
        del y
        del label
        model.eval()
        val_obs = []
        val_loss = 0
        step_size = 10000
        n_loops = epoch_val_x.shape[0]//step_size
        for v in tqdm(range(0, n_loops*step_size, step_size)):
            temp = model(epoch_val_x[v:v+step_size].to(DEVICE)).detach()
            val_loss += loss_fn(temp, epoch_val_y[v:v+step_size].to(DEVICE)).item()
            val_obs.append(temp.cpu().numpy())
        val_loss = val_loss/n_loops
        val_obs = np.concatenate(val_obs, axis=0)
        val_acc = np.mean([pearsonr(val_obs[:, i], epoch_val_y[:val_obs.shape[0], i].numpy()) for i in range(epoch_val_y.shape[-1])])
        print("Val Acc:", val_acc, " -- Val Loss:", val_loss, " | SaveFolder:", save)
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
        io.save_checkpoint_dict(save_dict,save,'test')
        del val_obs
        del temp
        print()
    with open(save + "/hyperparams.txt",'a') as f:
        f.write("\nFinal Loss:"+str(avg_loss)+" ValLoss:"+str(val_loss)+" ValAcc:"+str(val_acc)+"\n")
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
    hyps['exp_name'] = 'batchConstrainRange'
    hyps['n_epochs'] = 60
    hyps['batch_size'] = 512
    hyps['shuffle'] = True
    lrs = [5e-4]
    l1s = [1e-5]
    l2s = [1e-2]
    b1s = [0, 1, 5, 10]
    noises = [.05]
    exp_num = 0
    for noise in noises:
        hyps['noise'] = noise
        for lr in lrs:
            hyps['lr'] = lr
            for l1 in l1s:
                hyps['l1'] = l1
                for l2 in l2s:
                    hyps['l2'] = l2
                    for b1 in b1s:
                        hyps['b1'] = b1
                        hyps['save_folder'] = hyps['exp_name'] +"_"+ str(exp_num) + "_lr"+str(lr) + "_" + "b1" + str(b1)
                        hp.print()            
                        train(hyps, hyps['n_epochs'], hyps['batch_size'], lr, l1, l2, b1, hyps['shuffle'], hyps['save_folder'], hyps['noise'])
                        exp_num += 1



