import os
import torch
import torch.nn as nn
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.sampler import SequentialSampler
import numpy as np
from tqdm import tqdm
from collections import deque
from models import *
from data import *
from evaluation import pearsonr_eval
from utils import *
from config import get_default_cfg, get_custom_cfg

def train(cfg):
    
    device = torch.device('cuda:'+str(cfg.gpu))
    
    if cfg.Model.name == 'BN_CNN_Net':
        model = BN_CNN_Net(n_units=cfg.Model.n_units, noise=cfg.Model.noise, chans=cfg.Model.chans, 
                       bn_moment=cfg.Model.bn_moment, softplus=cfg.Model.softplus, 
                       img_shape=cfg.img_shape, ksizes=cfg.Model.ksizes).to(device)
    if cfg.Model.name == 'BNCNN_3D':
        model = BNCNN_3D(n_units=cfg.Model.n_units, noise=cfg.Model.noise, chans=cfg.Model.chans, 
                         bn_moment=cfg.Model.bn_moment, softplus=cfg.Model.softplus, 
                         img_shape=cfg.img_shape, ksizes=cfg.Model.ksizes, 
                         strides=cfg.Model.strides).to(device)
        
    start_epoch = 0
    
    loss_fn = nn.PoissonNLLLoss(log_input=False).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.Optimize.lr, 
                                 weight_decay=cfg.Optimize.l2)
    
    if cfg.Model.checkpoint != '':
        checkpoint = torch.load(cfg.Model.checkpoint, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        start_epoch = checkpoint['epoch']
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    train_dataset = TrainDataset(cfg)
    train_data = DataLoader(dataset=train_dataset, batch_size=cfg.Data.batch_size, shuffle=True)
    
    for epoch in range(cfg.epoch):
        epoch_loss = 0
        loss = 0
        for idx,(x,y) in enumerate(tqdm(train_data)):
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
        
        validation_data =  DataLoader(dataset=ValidationDataset(cfg))
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
            
            torch.save({'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': epoch_loss}, save_path)
    
if __name__ == "__main__":
    cfg = get_custom_cfg('3d_conv')
    train(cfg)