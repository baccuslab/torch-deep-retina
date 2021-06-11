import os
import argparse
import torch
import torch.nn as nn
from torch.utils.data.dataloader import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
from tqdm import tqdm
from kinetic.data import *
from kinetic.evaluation import pearsonr_eval
from kinetic.utils import *
import kinetic.models as models
from kinetic.config import get_custom_cfg

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
    
    loss_fn = select_lossfn(cfg.Optimize.loss_fn).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.Optimize.lr, 
                                 weight_decay=cfg.Optimize.l2)
    
    checkpoint = torch.load(cfg.Model.checkpoint, map_location=device)

    state_dict = model.state_dict()
    for key in checkpoint['model_state_dict'].keys():
        state_dict[key] = checkpoint['model_state_dict'][key]

    model.load_state_dict(state_dict)
    
    model.kinetics.ksi.data = 6.1832 * torch.ones(1,1).to(device)
    model.kinetics.ksr.data = 0.3129 * torch.ones(1,1).to(device)
    model.kinetics.ksr_2.data = 0. * torch.ones(1,1).to(device)
    
    model.kinetics_inh.ksi.data = 8.31 * torch.ones(1,1).to(device)
    model.kinetics_inh.ksr.data = 0.14 * torch.ones(1,1).to(device)
    model.kinetics_inh.ka.data = 20 * torch.ones(1,1).to(device)
    model.kinetics_inh.kfi.data = 46.7 * torch.ones(1,1).to(device)
    model.kinetics_inh.kfr.data = 79.2 * torch.ones(1,1).to(device)
    model.kinetics_inh.ksr_2.data = 0. * torch.ones(1,1).to(device)
    
    I20 = [None, np.array([1.04])]

    conv_weights = []
    for i in range((cfg.Model.ksizes[0]-1)//2):
        conv_weights.append(checkpoint['model_state_dict']['bipolar.0.convs.{}.weight'.format(i)].cpu().numpy())
    weight = LinearStack(conv_weights)

    tem_filter = weight.mean(0).sum(axis=(-1,-2)) - weight.mean(0).sum(axis=(-1,-2)).mean()
    tem_filter /= 200.

    x, y = np.meshgrid(np.linspace(-1,1,15), np.linspace(-1,1,15))
    dst = np.sqrt(x*x+y*y)
    sigma = 0.9
    gauss = np.exp(-( dst**2 / ( 2.0 * sigma**2 ) ) )

    conv_filter = np.expand_dims(tem_filter, axis=(0, -1, -2)) * gauss
    model.bipolar_inh[0].weight.data = torch.from_numpy(conv_filter).to(device)
    model.bipolar_inh[0].bias.data = -4 * torch.ones(1).to(device)

    model.kinetics_w_inh.data = 15 * torch.tensor([ 1., 1., 1., 1., 1., 1., 1., 1.])[:,None].to(device)
    model.kinetics_b_inh.data = 0 * torch.ones(8,1).to(device)
    model.bipolar[0].convs[6].bias.data = 0 * torch.ones(8).to(device)
    model.bipolar[2].scale_param.data = torch.tensor([ 1., 1./12, 1., 1./4, 1./1.7, 1./10, 1./10, 1./6])[:,None].to(device)
    model.bipolar[2].shift_param.data = -4 * torch.ones(8,1).to(device)

    model.kinetics_w.data = torch.tensor([1.30202385, 2.51288112, 1.40425839, 2.7186066, 2.14456289, 10.91153307, 11.64919857, 9.09428823])[:,None].to(device)
    model.kinetics_b.data = 0 * torch.ones(8,1).to(device)
    
    for para in model.parameters():
        para.requires_grad = False
    for para in model.amacrine.parameters():
        para.requires_grad = True
    for para in model.ganglion.parameters():
        para.requires_grad = True
    for para in model.bipolar[0].parameters():
        para.requires_grad = True
    model.kinetics_w.requires_grad = True
    model.kinetics_b.requires_grad = True
    
    
    model.train()
    model.float()
        
    scheduler_kwargs = dict(cfg.Scheduler)
    scheduler = ReduceLROnPlateau(optimizer, **scheduler_kwargs)
    
    data_kwargs = dict(cfg.Data)
    train_dataset = MyDataset(stim_sec='train', **data_kwargs)
    batch_sampler = BatchRnnSampler(length=len(train_dataset), batch_size=cfg.Data.batch_size,
                                    seq_len=cfg.Data.trunc_int)
    train_data = DataLoader(dataset=train_dataset, batch_sampler=batch_sampler)
    validation_data =  DataLoader(dataset=MyDataset(stim_sec='validation', stats=train_dataset.stats, **data_kwargs))
    seq_len = model.seq_len if cfg.Data.hs_mode == 'multiple' else None
    
    for epoch in range(start_epoch, start_epoch + cfg.epoch):
        epoch_loss = 0
        loss = 0
        hs = get_hs(model, cfg.Data.batch_size, device, I20=I20, mode=cfg.Data.hs_mode)
        for idx,(x,y) in enumerate(tqdm(train_data)):
            x = x.to(device)
            y = y.double().to(device)
            out, hs = model(x, hs)
            loss += loss_fn(out.double(), y)
            if idx % cfg.Data.trunc_int == 0:
                h_0 = detach_hs(hs, cfg.Data.hs_mode, seq_len)
            if idx % cfg.Data.trunc_int == (cfg.Data.trunc_int - 1):
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_loss += loss.detach().cpu().numpy()
                loss = 0
                hs = detach_hs(h_0, cfg.Data.hs_mode, seq_len)
                
        epoch_loss = epoch_loss / len(train_dataset) * cfg.Data.batch_size
        
        pearson = pearsonr_eval(model, validation_data, cfg.Model.n_units, device, 
                                I20=I20, start_idx=cfg.Data.start_idx, hs_mode=cfg.Data.hs_mode)
        scheduler.step(pearson)
        
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