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
    
    model.kinetics.ksi.data = 0. * torch.ones(model.chans[0], 1).to(device)
    model.kinetics.ksr.data = 0. * torch.ones(model.chans[0], 1).to(device)
    model.kinetics.ka.data = 23. * torch.ones(model.chans[0], 1).to(device)
    model.kinetics.kfi.data = 50. * torch.ones(model.chans[0], 1).to(device)
    model.kinetics.kfr.data = 87. * torch.ones(model.chans[0], 1).to(device)
    
    model.bipolar[0].convs[6].bias.data = -4. * torch.ones(model.chans[0]).to(device)
    #model.bipolar[0].convs[6].bias.requires_grad = False
    
    train_dataset = TrainDatasetBoth(cfg)
    batch_sampler = BatchRnnSampler(length=len(train_dataset), batch_size=cfg.Data.batch_size,
                                    seq_len=cfg.Data.trunc_int)
    train_data = DataLoader(dataset=train_dataset, batch_sampler=batch_sampler)
    cfg.Data.stim = 'naturalscene'
    validation_data_natural =  DataLoader(dataset=ValidationDataset(cfg, train_dataset.stats_natural))
    cfg.Data.stim = 'fullfield_whitenoise'
    validation_data_noise =  DataLoader(dataset=ValidationDataset(cfg, train_dataset.stats_noise))
    cfg.Data.stim = 'both'
    
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
        
        pearson_natural = pearsonr_eval(model, validation_data_natural, cfg.Model.n_units, device)
        pearson_noise = pearsonr_eval(model, validation_data_noise, cfg.Model.n_units, device, start_idx=4000)
        
        print('epoch: {:03d}, loss: {:.2f}, pearson_natural: {:.4f}, pearson_noise: {:.4f}'.format(epoch, epoch_loss, pearson_natural, pearson_noise))
        
        update_eval_history(cfg, epoch, (pearson_natural, pearson_noise), epoch_loss)
        
        if epoch%cfg.save_intvl == 0:

            save_path = os.path.join(cfg.save_path, cfg.exp_id, 
                                     'epoch_{:03d}_loss_{:.2f}_pearson_natural_{:.4f}_pearson_noise_{:.4f}'
                                     .format(epoch, epoch_loss, pearson_natural, pearson_noise)+'.pth')

            torch.save({'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': epoch_loss}, save_path)
    
if __name__ == "__main__":
    cfg = get_custom_cfg(opt.hyper)
    print(cfg)
    train(cfg)