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
    
    model = KineticsModel(drop_p=cfg.Model.drop_p, scale_kinet=cfg.Model.scale_kinet, 
                          recur_seq_len=cfg.Model.recur_seq_len, n_units=cfg.Model.n_units, 
                          noise=cfg.Model.noise, bias=cfg.Model.bias, 
                          linear_bias=cfg.Model.linear_bias, chans=cfg.Model.chans, 
                          bn_moment=cfg.Model.bn_moment, softplus=cfg.Model.softplus, 
                          img_shape=cfg.img_shape, ksizes=cfg.Model.ksizes).to(device)
    if cfg.Model.checkpoint != '':
        model.load_state_dict(torch.load(cfg.Model.checkpoint, map_location=device))
    model.train()
    
    loss_fn = nn.PoissonNLLLoss(log_input=False).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.Optimize.lr, 
                                 weight_decay=cfg.Optimize.l2)
    
    train_X, train_y = TrainData(cfg.Data.data_path, cfg.Data.batch_size, 30000)
    
    for epoch in range(cfg.epoch):
        count = 0
        idx = 0
        epoch_loss = 0
        loss = 0
        hs = get_hs(model, cfg.Data.batch_size, device)
        while idx < train_X.shape[0]:
            x = train_X[idx].to(device)
            y = train_y[idx].double().to(device)
            out, hs = model(x, hs)
            loss += loss_fn(out.double(), y)
            if count == 0:
                h_0 = []
                h_0.append(hs[0].detach())
                h_0.append(deque([h.detach() for h in hs[1]], maxlen=model.seq_len))
            if count == model.seq_len - 1:
                count = -1
                idx -= (model.seq_len - 1)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_loss += loss.detach().cpu().numpy()
                loss = 0
                hs[0] = h_0[0].detach()
                hs[1] = deque([h.detach() for h in h_0[1]], maxlen=model.seq_len)
            count += 1
            idx += 1
                
        epoch_loss = epoch_loss / train_X.shape[0] / cfg.Model.recur_seq_len
        
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