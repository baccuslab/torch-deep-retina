import os
import torch
import torch.nn as nn
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.sampler import SequentialSampler
import numpy as np
from collections import deque
from models import *
from data import *
from evaluation import pearsonr_eval
from utils import *
from config import get_default_cfg

def train(cfg):
    
    device = torch.device('cuda:'+str(cfg.gpu))
    
    model = BN_CNN_Net(n_units=cfg.Model.n_units, noise=cfg.Model.noise, chans=cfg.Model.chans, 
                       bn_moment=cfg.Model.bn_moment, softplus=cfg.Model.softplus, 
                       img_shape=cfg.img_shape, ksizes=cfg.Model.ksizes).to(device)
    if cfg.Model.checkpoint != '':
        model.load_state_dict(torch.load(cfg.Model.checkpoint, map_location=device))
    model.train()
    
    loss_fn = nn.PoissonNLLLoss(log_input=False).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.Optimize.lr, 
                                 weight_decay=cfg.Optimize.l2)
    
    train_dataset = TrainDataset(cfg.Data.data_path)
    train_data = DataLoader(dataset=train_dataset, batch_size=cfg.Data.batch_size, shuffle=True)
    
    for epoch in range(cfg.epoch):
        epoch_loss = 0
        loss = 0
        for idx,(x,y) in enumerate(train_data):
            x = x.to(device)
            y = y.double().to(device)
            out = model(x)
            loss += loss_fn(out.double(), y)
            if idx%cfg.Optimize.trunc_intvl == 0:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_loss += loss.detach().cpu().numpy()
                loss = 0
                
        epoch_loss = epoch_loss / len(train_dataset) * cfg.Data.batch_size
        
        validation_data =  DataLoader(dataset=ValidationDataset(cfg.Data.data_path))
        pearson = pearsonr_eval(model, validation_data, cfg.Model.n_units, device)
        
        print('epoch: {}, loss: {}, pearson correlation: {}'.format(epoch, epoch_loss, pearson))
        
        if epoch%cfg.save_intvl == 0:
            try:
                os.mkdir(os.path.join(cfg.save_path, cfg.exp_id))
                print("Directory Created ") 
            except FileExistsError:
                pass
            save_path = os.path.join(cfg.save_path, cfg.exp_id, 
                                     'epoch_{}_loss_{}_pearson_{}'
                                     .format(epoch, epoch_loss, pearson)+'.pth')
            torch.save(model.state_dict(), save_path)
    
if __name__ == "__main__":
    cfg = get_default_cfg()
    train(cfg)