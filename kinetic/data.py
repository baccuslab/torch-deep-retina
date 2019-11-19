import numpy as np
import torch
from torch.utils.data.sampler import Sampler
from  torch.utils.data.dataset import Dataset
from torchdeepretina.datas import loadexpt

class BatchRnnSampler(Sampler):
    
    def __init__(self, length, batch_size, seq_len):
        self.length = length
        self.batch_size = batch_size
        self.seq_len = seq_len

    def __iter__(self):
        batch_idx = 0
        count = 0
        while batch_idx < self.length // self.batch_size:
            batch = [batch_idx + n * self.length // self.batch_size for n in range(self.batch_size)]
            yield batch
            batch_idx += 1
            count += 1
            if count == self.seq_len:
                count = 0
                batch_idx -= (self.seq_len - 1)

    def __len__(self):
        return (self.length // self.batch_size - self.seq_len + 1) * self.seq_len + self.seq_len -1
    
class TrainDataset(Dataset):
    
    def __init__(self, cfg):
        super().__init__()
        data = loadexpt('15-10-07', [0,1,2,3,4], 'naturalscene', 'train',
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
        data = loadexpt('15-10-07', [0,1,2,3,4], 'naturalscene', 'train',
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
        data = loadexpt('15-10-07', [0,1,2,3,4], 'naturalscene', 'test',
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