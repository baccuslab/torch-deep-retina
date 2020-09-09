import numpy as np
import torch
from torch.utils.data.sampler import Sampler
from  torch.utils.data.dataset import Dataset
from torchdeepretina.datas import loadexpt


class TrainDataset(Dataset):
    
    def __init__(self, cfg):
        super().__init__()
        data = loadexpt(cfg.Data.date, 'all', cfg.Data.stim, 'train',
                        cfg.img_shape[0], 0, data_path=cfg.Data.data_path)
        val_size = cfg.Data.val_size
        self.X = data.X[:-val_size]
        self.y = data.y[:-val_size]
        self.centers = data.centers
        self.stats = data.stats
        
    def __len__(self):
        return self.y.shape[0]
    
    def __getitem__(self, index):
        inpt = torch.from_numpy(self.X[index])
        trgt = torch.from_numpy(self.y[index])
        return (inpt, trgt)
    
class ValidationDataset(Dataset):
    
    def __init__(self, cfg):
        super().__init__()
        data = loadexpt(cfg.Data.date, 'all', cfg.Data.stim, 'train',
                        cfg.img_shape[0], 0, data_path=cfg.Data.data_path)
        val_size = cfg.Data.val_size
        self.X = data.X[-val_size:]
        self.y = data.y[-val_size:]
        self.centers = data.centers
        self.stats = data.stats
        
    def __len__(self):
        return self.y.shape[0]
    
    def __getitem__(self, index):
        inpt = torch.from_numpy(self.X[index])
        trgt = torch.from_numpy(self.y[index])
        return (inpt, trgt)
    
class TestDataset(Dataset):
    
    def __init__(self, cfg):
        super().__init__()
        data = loadexpt(cfg.Data.date, 'all', cfg.Data.stim, 'test',
                        cfg.img_shape[0], 0, data_path=cfg.Data.data_path)
        val_size = cfg.Data.val_size
        self.X = data.X
        self.y = data.y
        self.centers = data.centers
        self.stats = data.stats
        
    def __len__(self):
        return self.y.shape[0]
    
    def __getitem__(self, index):
        inpt = torch.from_numpy(self.X[index])
        trgt = torch.from_numpy(self.y[index])
        return (inpt, trgt)