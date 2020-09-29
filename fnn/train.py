import os
import argparse
import torch
import json
import torch.nn as nn
from torch.utils.data.dataloader import DataLoader
import numpy as np
from tqdm import tqdm
from collections import deque
from fnn.models import *
from fnn.data import *
from fnn.evaluation import *
from fnn.utils import *
from fnn.config import get_default_cfg, get_custom_cfg

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, required=True)
parser.add_argument('--hyper', type=str, required=True)
opt = parser.parse_args()

def train(cfg):
    
    if not os.path.exists(os.path.join(cfg.save_path, cfg.exp_id)):
        os.mkdir(os.path.join(cfg.save_path, cfg.exp_id))
        
    with open(os.path.join(cfg.save_path, cfg.exp_id, 'cfg'), 'w') as f:
        f.write(str(cfg))
    
    device = torch.device('cuda:'+str(opt.gpu))
    
    model = select_model(cfg, device)
        
    start_epoch = 0
    
    loss_fn = nn.PoissonNLLLoss(log_input=False).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.Optimize.lr, 
                                 weight_decay=cfg.Optimize.l2)
    
    #scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)
    
    if cfg.Model.checkpoint != '':
        checkpoint = torch.load(cfg.Model.checkpoint, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    model.train()
    
    train_data, test_data = get_data(cfg)
    data_distr = get_model_and_distr(train_data, cfg.Data.val_size, cfg.Data.batch_size)
    
    for epoch in range(cfg.epoch):
        epoch_loss = 0
        loss = 0
        for idx,(x,y) in enumerate(tqdm(data_distr.train_sample())):
            x = x.to(device)
            y = y.double().to(device)
            out = model(x)
            loss += loss_fn(out.double(), y)
            loss += cfg.Optimize.l1 * torch.norm(y, 1).float().mean()
            if idx%cfg.Optimize.trunc_intvl == 0:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_loss += loss.detach().cpu().numpy()
                loss = 0
                
        epoch_loss = epoch_loss / data_distr.train_shape[0] * cfg.Data.batch_size
        
        pearson, _,_,_ = pearsonr_batch_eval(model, data_distr.val_sample(500), cfg.Model.n_units, device, cfg)
        #scheduler.step(eval_loss)
        
        print('epoch: {:03d}, loss: {:.2f}, pearson correlation: {:.4f}'.format(epoch, epoch_loss, pearson))
        
        update_eval_history(cfg, epoch, pearson, epoch_loss)
        
        if epoch % cfg.save_intvl == 0:
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