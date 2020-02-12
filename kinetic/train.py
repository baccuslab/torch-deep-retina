import os
import torch
import torch.nn as nn
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.sampler import SequentialSampler
import numpy as np
from tqdm import tqdm
from collections import deque
from kinetic.data import *
from kinetic.evaluation import pearsonr_eval
from kinetic.utils import *
from kinetic.models import *
from kinetic.config import get_default_cfg, get_custom_cfg

def train(cfg):
    
    device = torch.device('cuda:'+str(cfg.gpu))
    
    if cfg.Model.name == 'KineticsChannelModel':
        model = KineticsChannelModel(drop_p=cfg.Model.drop_p, scale_kinet=cfg.Model.scale_kinet, 
                                  recur_seq_len=cfg.Model.recur_seq_len, n_units=cfg.Model.n_units, 
                                  noise=cfg.Model.noise, bias=cfg.Model.bias, 
                                  linear_bias=cfg.Model.linear_bias, chans=cfg.Model.chans, 
                                  bn_moment=cfg.Model.bn_moment, softplus=cfg.Model.softplus, 
                                  img_shape=cfg.img_shape, ksizes=cfg.Model.ksizes).to(device)
    if cfg.Model.name == 'KineticsChannelModelFilter':
        model = KineticsChannelModelFilter(drop_p=cfg.Model.drop_p, scale_kinet=cfg.Model.scale_kinet, 
                                  recur_seq_len=cfg.Model.recur_seq_len, n_units=cfg.Model.n_units, 
                                  noise=cfg.Model.noise, bias=cfg.Model.bias, 
                                  linear_bias=cfg.Model.linear_bias, chans=cfg.Model.chans, 
                                  bn_moment=cfg.Model.bn_moment, softplus=cfg.Model.softplus, 
                                  img_shape=cfg.img_shape, ksizes=cfg.Model.ksizes).to(device)
    if cfg.Model.name == 'KineticsChannelModelNoBatchnorm':
        model = KineticsChannelModelNoBatchnorm(drop_p=cfg.Model.drop_p, scale_kinet=cfg.Model.scale_kinet, 
                                  recur_seq_len=cfg.Model.recur_seq_len, n_units=cfg.Model.n_units, 
                                  noise=cfg.Model.noise, bias=cfg.Model.bias, 
                                  linear_bias=cfg.Model.linear_bias, chans=cfg.Model.chans, 
                                  softplus=cfg.Model.softplus, 
                                  img_shape=cfg.img_shape, ksizes=cfg.Model.ksizes).to(device)
    if cfg.Model.name == 'KineticsChannelModelLayerNorm':
        model = KineticsChannelModelLayerNorm(drop_p=cfg.Model.drop_p, scale_kinet=cfg.Model.scale_kinet, 
                                  recur_seq_len=cfg.Model.recur_seq_len, n_units=cfg.Model.n_units, 
                                  bias=cfg.Model.bias, 
                                  linear_bias=cfg.Model.linear_bias, chans=cfg.Model.chans, 
                                  softplus=cfg.Model.softplus, 
                                  img_shape=cfg.img_shape, ksizes=cfg.Model.ksizes).to(device)
    if cfg.Model.name == 'KineticsChannelModelInstanceNorm':
        model = KineticsChannelModelInstanceNorm(drop_p=cfg.Model.drop_p, scale_kinet=cfg.Model.scale_kinet, 
                                  recur_seq_len=cfg.Model.recur_seq_len, n_units=cfg.Model.n_units, 
                                  bias=cfg.Model.bias, 
                                  linear_bias=cfg.Model.linear_bias, chans=cfg.Model.chans, 
                                  softplus=cfg.Model.softplus, 
                                  img_shape=cfg.img_shape, ksizes=cfg.Model.ksizes).to(device)
    if cfg.Model.name == 'KineticsModel':
        model = KineticsModel(drop_p=cfg.Model.drop_p, scale_kinet=cfg.Model.scale_kinet, 
                          recur_seq_len=cfg.Model.recur_seq_len, n_units=cfg.Model.n_units, 
                          noise=cfg.Model.noise, bias=cfg.Model.bias, 
                          linear_bias=cfg.Model.linear_bias, chans=cfg.Model.chans, 
                          bn_moment=cfg.Model.bn_moment, softplus=cfg.Model.softplus, 
                          img_shape=cfg.img_shape, ksizes=cfg.Model.ksizes).to(device)
    
    #model = select_model(cfg, device)
    start_epoch = 0
        
    model.train()
    
    loss_fn = nn.PoissonNLLLoss(log_input=False).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.Optimize.lr, 
                                 weight_decay=cfg.Optimize.l2)
    
    if cfg.Model.checkpoint != '':
        checkpoint = torch.load(cfg.Model.checkpoint, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        start_epoch = checkpoint['epoch']
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    train_dataset = TrainDataset(cfg)
    batch_sampler = BatchRnnSampler(length=len(train_dataset), batch_size=cfg.Data.batch_size,
                                    seq_len=cfg.Data.trunc_int)
    train_data = DataLoader(dataset=train_dataset, batch_sampler=batch_sampler)
    
    for epoch in range(start_epoch, start_epoch + cfg.epoch):
        epoch_loss = 0
        loss = 0
        hs = get_hs(model, cfg.Data.batch_size, device)
        for idx,(x,y) in enumerate(tqdm(train_data)):
            x = x.to(device)
            y = y.double().to(device)
            out, hs = model(x, hs)
            loss += loss_fn(out.double(), y)
            if idx % cfg.Data.trunc_int == 0:
                h_0 = []
                h_0.append(hs[0].detach())
                h_0.append(deque([h.detach() for h in hs[1]], maxlen=model.seq_len))
            if idx % cfg.Data.trunc_int == (cfg.Data.trunc_int - 1):
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_loss += loss.detach().cpu().numpy()
                loss = 0
                hs[0] = h_0[0].detach()
                hs[1] = deque([h.detach() for h in h_0[1]], maxlen=model.seq_len)
                
        epoch_loss = epoch_loss / len(train_dataset) * cfg.Data.batch_size
        
        validation_data =  DataLoader(dataset=ValidationDataset(cfg))
        pearson = pearsonr_eval(model, validation_data, cfg.Model.n_units, 600, device)
        
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
    cfg = get_custom_cfg('channel_filter')
    print(cfg)
    train(cfg)