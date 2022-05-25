import numpy as np
import torch
from torch.utils.data.sampler import Sampler
from  torch.utils.data.dataset import Dataset
from torchdeepretina.datas import loadexpt

class TrainValidDataset(Dataset):
    
    def __init__(self, cfg, dataset, shuffle, perm=None):
        super().__init__()
        data = loadexpt(cfg.Data.date, 'all', cfg.Data.stim, 'train',
                        cfg.img_shape[0], 0, data_path=cfg.Data.data_path)
        val_size = cfg.Data.val_size
        
        if type(perm) == np.ndarray:
            self.perm = perm
        elif shuffle:
            self.perm = np.random.permutation(data.X.shape[0]).astype('int')
        else:
            self.perm = np.arange(data.X.shape[0]).astype('int')
        self.train_idxs = self.perm[:-val_size]
        self.val_idxs = self.perm[-val_size:]
        self.X = data.X
        self.y = data.y
        self.centers = data.centers
        self.stats = data.stats
        self.dataset = dataset
        
    def __len__(self):
        if self.dataset == 'train':
            return self.train_idxs.shape[0]
        if self.dataset == 'validation':
            return self.val_idxs.shape[0]
    
    def __getitem__(self, index):
        if self.dataset == 'train':
            idx = self.train_idxs[index]
        if self.dataset == 'validation':
            idx = self.val_idxs[index]
        inpt = torch.from_numpy((self.X[idx].astype('float32') - self.stats['mean']) / self.stats['std'])
        trgt = torch.from_numpy(self.y[idx])
        return (inpt, trgt)

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
        inpt = torch.from_numpy((self.X[index].astype('float32') - self.stats['mean']) / self.stats['std'])
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
        inpt = torch.from_numpy((self.X[index].astype('float32') - self.stats['mean']) / self.stats['std'])
        trgt = torch.from_numpy(self.y[index])
        return (inpt, trgt)
    
class TestDataset(Dataset):
    
    def __init__(self, cfg):
        super().__init__()
        data = loadexpt(cfg.Data.date, 'all', cfg.Data.stim, 'test',
                        cfg.img_shape[0], 0, data_path=cfg.Data.data_path)
        self.X = data.X
        self.y = data.y
        self.centers = data.centers
        self.stats = data.stats
        
    def __len__(self):
        return self.y.shape[0]
    
    def __getitem__(self, index):
        inpt = torch.from_numpy((self.X[index].astype('float32') - self.stats['mean']) / self.stats['std'])
        trgt = torch.from_numpy(self.y[index])
        return (inpt, trgt)