import os
import argparse
import torch
import torch.nn as nn
from torch.utils.data.dataloader import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
from tqdm import tqdm
from collections import deque
from kinetic.data import *
from kinetic.evaluation import pearsonr_eval
from kinetic.utils import *
from kinetic.models import *
from kinetic.config import get_custom_cfg
from kinetic.LNK_data import *

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, required=True)
parser.add_argument('--hyper', type=str, required=True)
opt = parser.parse_args()

def train(cfg):
    
    device = torch.device('cuda:'+str(opt.gpu))
    
    model = select_model(cfg, device)
        
    if cfg.Model.checkpoint != '':
        checkpoint = torch.load(cfg.Model.checkpoint, map_location=device)
        model = OnePixelModel(cfg, checkpoint['model_state_dict'], 0.01, device)
    else:
        raise Exception('Require a trained model!')
    for para in model.parameters():
        para.requires_grad = False
    model.kinetics.ksi.requires_grad = True
    model.kinetics.ksr.requires_grad = True
    
    #model.kinetics.ka.requires_grad = True
    #model.kinetics.kfi.requires_grad = True
    #model.kinetics.kfr.requires_grad = True
    
    #model.kinetics.ksi.data = torch.rand(cfg.Model.chans[0], 1).abs().to(device)/10
    #model.kinetics.ksr.data = torch.rand(cfg.Model.chans[0], 1).abs().to(device)/10
    print("Initial slow parameters: ", model.kinetics.ksi.data, model.kinetics.ksr.data)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.Optimize.lr, 
                                 weight_decay=cfg.Optimize.l2)
    
    if not os.path.exists(os.path.join(cfg.save_path, cfg.exp_id)):
        os.mkdir(os.path.join(cfg.save_path, cfg.exp_id))
        
    with open(os.path.join(cfg.save_path, cfg.exp_id, 'cfg'), 'w') as f:
        f.write(str(cfg))
    

    start_epoch = 0
        
    model.train()
    
    loss_fn = nn.MSELoss().to(device)
    #loss_fn = nn.PoissonNLLLoss(log_input=False).to(device)
    
    scheduler = ReduceLROnPlateau(optimizer, 'max', factor=0.2, patience=2)
    
    train_dataset = TrainDatasetOnePixel(cfg)
    batch_sampler = BatchRnnOneTimeSampler(length=len(train_dataset), batch_size=cfg.Data.batch_size)
    train_data = DataLoader(dataset=train_dataset, batch_sampler=batch_sampler)
    validation_data =  DataLoader(dataset=ValidationDatasetOnePixel(cfg, train_dataset.stats))
    
    for epoch in range(start_epoch, start_epoch + cfg.epoch):
        epoch_loss = 0
        hs = get_hs(model, cfg.Data.batch_size, device)
        y_preds = []
        y_targs = []
        for idx,(x,y) in enumerate(tqdm(train_data)):
            x = x.to(device)
            y = y.double().to(device)
            out, hs = model(x, hs)
            y_preds.append(out.double())
            y_targs.append(y)
            if idx % cfg.Data.loss_bin == (cfg.Data.loss_bin - 1):
                y_pred = torch.stack(y_preds, dim=2)
                y_targ = torch.stack(y_targs, dim=2)
                loss = temporal_frequency_normalized_loss(y_pred, y_targ, loss_fn, device, num_units=cfg.Model.n_units)
                #loss = loss_fn(y_pred, y_targ)
                optimizer.zero_grad()
                loss.backward(retain_graph=True)
                optimizer.step()
                epoch_loss += loss.detach().cpu().numpy()
                y_preds = []
                y_targs = []
            if idx % cfg.Data.trunc_int == (cfg.Data.trunc_int - 1):
                hs[0] = hs[0].detach()
                hs[1] = deque([h.detach() for h in hs[1]], maxlen=model.seq_len)
                
        epoch_loss = epoch_loss / len(train_dataset) * cfg.Data.batch_size * cfg.Data.loss_bin
        
        pearson = pearsonr_eval(model, validation_data, cfg.Model.n_units, device)
        scheduler.step(pearson)
        
        print('epoch: {:03d}, loss: {:.2f}, pearson correlation: {:.4f}'.format(epoch, epoch_loss, pearson))
        print("Slow parameters: ", model.kinetics.ksi.data, model.kinetics.ksr.data)
        
        update_eval_history(cfg, epoch, pearson, epoch_loss)
        
        if epoch%cfg.save_intvl == 0:

            save_path = os.path.join(cfg.save_path, cfg.exp_id, 
                                     'epoch_{:03d}_loss_{:.2f}_pearson_{:.4f}'
                                     .format(epoch, epoch_loss, pearson)+'.pth')

            torch.save({'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': epoch_loss}, save_path)
    
if __name__ == "__main__":
    cfg = get_custom_cfg(opt.hyper)
    print(cfg)
    train(cfg)