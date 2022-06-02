import os
import argparse
import torch
import json
import torch.nn as nn
from torch.utils.data.dataloader import DataLoader
import numpy as np
from tqdm import tqdm
from collections import deque
import fnn.models as models
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
    
    model_func = getattr(models, cfg.Model.name)
    model_kwargs = dict(cfg.Model)
    model = model_func(**model_kwargs).to(device)
        
    start_epoch = 0
    
    loss_fn = nn.PoissonNLLLoss(log_input=False).to(device)
    #loss_fn = nn.MSELoss().to(device)
    #loss_fn = nn.L1Loss().to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.Optimize.lr, 
                                 weight_decay=cfg.Optimize.l2)
    
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.2, patience=3)
    
    assert cfg.Model.checkpoint != ''
    checkpoint = torch.load(cfg.Model.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
        
    w = checkpoint['model_state_dict']['ganglion.6.w'].data
    positive = w - torch.min(w,1)[0][:,None]
    normed = positive.permute(1,0) / positive.sum(-1)
    prob = normed.permute(1,0).cpu().numpy()
    loc_ganglion = [np.where(prob[i] == prob[i].max())[0][0] for i in range(prob.shape[0])]
    
    num_locations = (model.image_shape[-1] - model.ksizes[0] - model.ksizes[1] - model.ksizes[2] + 3)**2
    w = np.zeros((model.n_units, num_locations))
    w[range(model.n_units), loc_ganglion] = 1.
    model.ganglion[-1].w.data = torch.from_numpy(w).to(device)
    model.ganglion[-1].w.requires_grad = False
    
    model.train()
    
    test_dataset = TestDataset(cfg)
    train_data = DataLoader(dataset=test_dataset, batch_size=cfg.Data.batch_size, shuffle=True)
    
    train_dataset = TrainValidDataset(cfg, 'train', True)
    perm = train_dataset.perm
    validation_dataset = TrainValidDataset(cfg, 'validation', True, perm)
    validation_data = DataLoader(dataset=validation_dataset, batch_size=cfg.Data.batch_size)
    
    for epoch in range(cfg.epoch):
        epoch_loss = 0
        loss = 0
        for idx,(x,y) in enumerate(tqdm(train_data)):
            x = x.to(device)
            y = y.double().to(device)
            out = model(x)
            loss += loss_fn(out.double(), y)
            loss += cfg.Optimize.l1 * torch.norm(out, 1).float().mean()
            if idx%cfg.Optimize.trunc_intvl == 0:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_loss += loss.detach().cpu().numpy()
                loss = 0
                
        pearson, _,_ = pearsonr_batch_eval(model, train_data, cfg.Model.n_units, device)
        #pearsons = pearsonr_eval_cell(model, data_distr.val_sample(500), cfg.Model.n_units, device)
        scheduler.step(pearson)
        
        pc, _,_ = pearsonr_batch_eval(model, validation_data, cfg.Model.n_units, device)
        
        print('epoch: {:03d}, pearson correlation: {:.4f}, validation pc: {:.4f}'.format(epoch, pearson, pc))
        
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