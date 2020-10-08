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
            batch = [batch_idx + n * (self.length // self.batch_size) for n in range(self.batch_size)]
            yield batch
            batch_idx += 1
            count += 1
            if count == self.seq_len:
                count = 0
                batch_idx -= (self.seq_len - 1)

    def __len__(self):
        return (self.length // self.batch_size - self.seq_len + 1) * self.seq_len + self.seq_len -1
    
class BatchRnnOneTimeSampler(Sampler):
    
    def __init__(self, length, batch_size):
        self.length = length
        self.batch_size = batch_size

    def __iter__(self):
        batch_idx = 0
        while batch_idx < self.length // self.batch_size:
            batch = [batch_idx + n * (self.length // self.batch_size) for n in range(self.batch_size)]
            yield batch
            batch_idx += 1

    def __len__(self):
        return self.length // self.batch_size
    
def index_both_mask(len_natural, len_noise, n_split=5, val_size=30000):
    
    each_len_natural =  len_natural // n_split
    each_len_noise = len_noise // n_split
    mask = np.concatenate((np.ones(each_len_natural), np.zeros(each_len_noise)))
    mask = np.tile(mask, n_split-1)
    mask = np.concatenate((mask, np.ones(len_natural-val_size-(n_split-1)*each_len_natural)))
    mask = np.concatenate((mask, np.zeros(len_noise-val_size-(n_split-1)*each_len_noise)))
    return mask
    
def index_both_idx(index, mask):

    stim_type = mask[index]
    
    if mask[index] == 0.:
        stimulus = 'noise'
    if mask[index] == 1.:
        stimulus = 'natural'
    idx = (mask[:index] == stim_type).sum()
    
    return stimulus, idx

def XY(data, stim_type, img_shape, stim_sec, val_size):
    if stim_type == 'full':
        X = data.X[:]
    elif stim_type == 'one_pixel':
        X = data.X[:, :, img_shape[1]//2, img_shape[2]//2]
    else:
        raise Exception('Invalid stimulus type')
    y = data.y[:]
    if stim_sec == 'train':
        X = X[:-val_size]
        y = y[:-val_size]
    elif stim_sec == 'validation':
        X = X[-val_size:]
        y = y[-val_size:]
    elif stim_sec == 'test':
        pass
    else:
        raise Exception('Invalid stimlus section')
    return X, y
    
class MyDataset(Dataset):
    
    def __init__(self, stim_sec, img_shape, data_path, date, stim, val_size, 
                 stats=None, cells='all', stim_type='full', **kwargs):
        super().__init__()
        if stim_sec == 'train' or stim_sec == 'validation':
            data = loadexpt(date, cells, stim, 'train', img_shape[0], 0, norm_stats=stats, data_path=data_path)
        elif stim_sec == 'test':
            data = loadexpt(date, cells, stim, 'test', img_shape[0], 0, norm_stats=stats, data_path=data_path)
        else:
            raise Exception('Invalid stimulus section')
        self.X, self.y = XY(data, stim_type, img_shape, stim_sec, val_size)
        self.centers = data.centers
        self.stats = data.stats
        
    def __len__(self):
        return self.y.shape[0]
    
    def __getitem__(self, index):
        inpt = torch.from_numpy(self.X[index])
        trgt = torch.from_numpy(self.y[index])
        return (inpt, trgt)
    
class TrainDatasetBoth(Dataset):
    
    def __init__(self, cfg, cells='all'):
        super().__init__()
        assert cfg.Data.stim == 'both'
        data_natural = loadexpt(cfg.Data.date, cells, 'naturalscene', 'train',
                        cfg.img_shape[0], 0, data_path=cfg.Data.data_path)
        data_noise = loadexpt(cfg.Data.date, cells, 'fullfield_whitenoise', 'train',
                        cfg.img_shape[0], 0, data_path=cfg.Data.data_path)
        
        self.val_size = cfg.Data.val_size
        self.len_natural = data_natural.y.shape[0]
        self.len_noise = data_noise.y.shape[0]
        self.n_split = 5
        
        self.X_natural = data_natural.X[:-self.val_size]
        self.y_natural = data_natural.y[:-self.val_size]
        self.X_noise = data_noise.X[:-self.val_size]
        self.y_noise = data_noise.y[:-self.val_size]
        
        self.stats_natural = data_natural.stats
        self.stats_noise = data_noise.stats
        self.index_mask = index_both_mask(self.len_natural, self.len_noise, self.n_split, self.val_size)
        
    def __len__(self):
        return self.y_natural.shape[0] + self.y_noise.shape[0]
    
    def __getitem__(self, index):
        stimulus, idx = index_both_idx(index, self.index_mask)
        if stimulus == 'natural':
            inpt = torch.from_numpy(self.X_natural[idx])
            trgt = torch.from_numpy(self.y_natural[idx])
        if stimulus == 'noise':
            inpt = torch.from_numpy(self.X_noise[idx])
            trgt = torch.from_numpy(self.y_noise[idx])
                
        return (inpt, trgt)