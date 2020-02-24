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

def amacrine_minus_1(key):
    
    assert 'amacrine' in key
    num = int(key[9])
    new_key = key[:9] + str(num-1) + key[10:]
    return new_key

def train(cfg):
    
    if not os.path.exists(os.path.join(cfg.save_path, cfg.exp_id)):
        os.mkdir(os.path.join(cfg.save_path, cfg.exp_id))
        
    with open(os.path.join(cfg.save_path, cfg.exp_id, 'cfg'), 'w') as f:
        f.write(str(cfg))
    
    device = torch.device('cuda:'+str(cfg.gpu))
    
    model = select_model(cfg, device)
    start_epoch = 0
        
    model.train()
    
    loss_fn = nn.PoissonNLLLoss(log_input=False).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.Optimize.lr, 
                                 weight_decay=cfg.Optimize.l2)
    
    checkpoint_path_BNCNN = '/home/xhding/saved_model/BN_CNN_Stack/epoch_030_loss_-2.77_pearson_0.6291_eval_loss_-29.18.pth'
    checkpoint_BNCNN = torch.load(checkpoint_path_BNCNN, map_location=device)
    for key in model.state_dict().keys():
        if ('amacrine' in key) and (key != 'amacrine.1.filter'):
            model.state_dict()[key] = checkpoint_BNCNN['model_state_dict'][amacrine_minus_1(key)]
        if 'ganglion' in key:
            model.state_dict()[key] = checkpoint_BNCNN['model_state_dict'][key]
    '''
    for name, p in model.amacrine.named_parameters():
        if 'filter' not in name:
            p.requires_grad = False
    for name, p in model.ganglion.named_parameters():
        p.requires_grad = False
    '''
    model.amacrine.eval()
    model.ganglion.eval()
    '''
    model.kinetics.ksi.requires_grad = False
    model.kinetics.ksr.requires_grad = False
            
    model.kinetics.ka.requires_grad = False
    model.kinetics.kfi.requires_grad = False
    model.kinetics.kfr.requires_grad = False
    '''
    
    model.kinetics.ksi.data = 0. * torch.ones(model.chans[0], 1).to(device)
    model.kinetics.ksr.data = 0. * torch.ones(model.chans[0], 1).to(device)
    model.kinetics.ka.data = 23. * torch.ones(model.chans[0], 1).to(device)
    model.kinetics.kfi.data = 50. * torch.ones(model.chans[0], 1).to(device)
    model.kinetics.kfr.data = 87. * torch.ones(model.chans[0], 1).to(device)
    
    train_dataset = TrainDataset(cfg)
    batch_sampler = BatchRnnSampler(length=len(train_dataset), batch_size=cfg.Data.batch_size,
                                    seq_len=cfg.Data.trunc_int)
    train_data = DataLoader(dataset=train_dataset, batch_sampler=batch_sampler)
    validation_data =  DataLoader(dataset=ValidationDataset(cfg))
    
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
        
        pearson = pearsonr_eval(model, validation_data, cfg.Model.n_units, 600, device)
        
        print('epoch: {:03d}, loss: {:.2f}, pearson correlation: {:.4f}'.format(epoch, epoch_loss, pearson))
        
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
    cfg = get_custom_cfg('channel_filter_bipolar')
    print(cfg)
    train(cfg)