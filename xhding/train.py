import torch
import torch.nn as nn
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.sampler import SequentialSampler
import numpy as np
from collections import deque
from models import *
from data import *
from evaluation import pearsonr

def get_hs(model, batch_size, device):
    hs = []
    hs.append(torch.zeros(batch_size, *model.h_shapes[0]).to(device))
    hs[0][:,0] = 1
    hs.append(deque([],maxlen=model.seq_len))
    for i in range(model.seq_len):
        hs[1].append(torch.zeros(batch_size, *model.h_shapes[1]).to(device))
    return hs

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
    
    train_dataset = TrainDataset(cfg.Data.data_path)
    batch_sampler = BatchRnnSampler(SequentialSampler(range(len(train_dataset))), 
                                    batch_size=cfg.Data.batch_size)
    train_data = DataLoader(dataset=train_dataset, batch_sampler=batch_sampler)
    
    for epoch in range(cfg.epoch):
        epoch_loss = 0
        hs = get_hs(model, cfg.Data.batch_size, device)
        for idx,(x,y) in enumerate(train_data):
            loss = 0
            x = x.to(device)
            y = y.to(device)
            out, hs = model(x, hs)
            loss += loss_fn(out, y)
            if idx%cfg.Optimize.trunc_intrvl == 0:
                loss.backward()
                optimizer.step()
                epoch_loss += loss.detach().cpu().numpy()
                loss = 0
                hs = hs.detach()
                hs[0] = hs[0].data.clone()
                hs[1] = deque([h.data.clone() for h in hs[1]], maxlen=model.seq_len)
                
        epoch_loss = epoch_loss / len(train_dataset) * cfg.Data.batch_size
        
        validation_data =  DataLoader(dataset=ValidationDataset(cfg.Data.data_path))
        pearson = pearsonr(model, validation_data, cfg.Model.n_units, device)
        
        print('epoch: {}, loss: {}, pearson correlation: {}'.format(epoch, epoch_loss, pearson))
        
        if epoch%cfg.save_intrvl == 0:
            try:
                os.mkdir(os.path.join(cfg.save_path, cfg.exp_id))
                print("Directory Created ") 
            except FileExistsError:
                print("Directory already exists")
            save_path = os.path.join(cfg.save_path, cfg.exp_id, 
                                     'epoch_{}_loss_{}_pearson_{}'
                                     .format(epoch, epoch_loss, pearson)+'.pth')
            torch.save(model.state_dict(), save_path)
    
    