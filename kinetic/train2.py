import os
import argparse
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
        
    model.train()
    
    loss_fn = nn.PoissonNLLLoss(log_input=False).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.Optimize.lr, 
                                 weight_decay=cfg.Optimize.l2)
    
    if cfg.Model.checkpoint != '':
        checkpoint = torch.load(cfg.Model.checkpoint, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    train_dataset = TrainDataset(cfg)
    batch_sampler = BatchRnnSampler(length=len(train_dataset), batch_size=cfg.Data.batch_size,
                                    seq_len=cfg.Data.trunc_int)
    train_data = DataLoader(dataset=train_dataset, batch_sampler=batch_sampler)
    validation_data =  DataLoader(dataset=ValidationDataset(cfg, train_dataset.stats))
    
    for epoch in range(start_epoch, start_epoch + cfg.epoch):
        epoch_loss = 0
        loss = 0
        hs = get_hs(model, cfg.Data.batch_size, device)
        for idx,(x,y) in enumerate(tqdm(train_data)):
            x = x.to(device)
            fx = model.bipolar(x)
            fx, h0 = model.kinetics(fx, hs[0]) 
            hs[1].append(fx)
            h1 = hs[1]
            hs = [h0, h1]           
            if idx % cfg.Data.trunc_int == 0:
                h_0 = []
                h_0.append(hs[0].detach())
                h_0.append(deque([h.detach() for h in hs[1]], maxlen=model.seq_len))
            if idx % cfg.Data.trunc_int == (cfg.Data.trunc_int - 1):
                y = y.double().to(device)
                fx = torch.stack(list(h1), dim=1) #(B,D,N)
                if model.scale_kinet:
                    fx = model.kinet_scale(fx)
                fx = model.amacrine(fx)
                out = model.ganglion(fx)
                loss = loss_fn(out.double(), y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_loss += loss.detach().cpu().numpy()
                loss = 0
                hs[0] = h_0[0].detach()
                hs[1] = deque([h.detach() for h in h_0[1]], maxlen=model.seq_len)
                
        epoch_loss = epoch_loss / len(train_dataset) * cfg.Data.batch_size
        
        pearson = pearsonr_eval(model, validation_data, cfg.Model.n_units, len(validation_data), device)
        
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
    cfg = get_custom_cfg(opt.hyper)
    print(cfg)
    train(cfg)