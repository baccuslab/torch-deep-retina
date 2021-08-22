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

def interleave(a, b, each_len_a, each_len_b):
    n_split = a.shape[0] // each_len_a
    n_split_b = b.shape[0] // each_len_b
    assert n_split_b == n_split
    result = []
    for i in range(n_split):
        result.append(a[i*each_len_a:(i+1)*each_len_a])
        result.append(b[i*each_len_b:(i+1)*each_len_b])
    result.append(a[n_split*each_len_a:])
    result.append(b[n_split*each_len_b:])
    result = np.concatenate(result, axis=0)
    return result

def XY(data, stim_type, img_shape, stim_sec, val_size):
    if stim_type == 'full':
        X = data.X[:]
    elif stim_type == 'one_pixel':
        X = data.X[:, :, img_shape[1]//2, img_shape[2]//2]
    elif stim_type == 'code_bar':
        X = data.X[:, :, img_shape[1]//2, :]
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
        inpt = torch.from_numpy((self.X[index].astype('float32') - self.stats['mean']) / self.stats['std'])
        trgt = torch.from_numpy(self.y[index])
        return (inpt, trgt)
    
class TrainDatasetBoth(Dataset):
    
    def __init__(self, img_shape, data_path, date, stim, val_size, cells='all', **kwargs):
        super().__init__()
        assert stim == 'both'
        data_natural = loadexpt(date, cells, 'naturalscene', 'train',
                                img_shape[0], 0, data_path=data_path)
        data_noise = loadexpt(date, cells, 'fullfield_whitenoise', 'train',
                              img_shape[0], 0, data_path=data_path)
        
        self.val_size = val_size
        self.len_natural = data_natural.y.shape[0]
        self.len_noise = data_noise.y.shape[0]
        self.n_split = 5
        
        X_natural = data_natural.X[:-self.val_size]
        y_natural = data_natural.y[:-self.val_size]
        X_noise = data_noise.X[:-self.val_size]
        y_noise = data_noise.y[:-self.val_size]
        
        self.stats_natural = data_natural.stats
        self.stats_noise = data_noise.stats
        
        each_len_natural = self.len_natural // self.n_split
        each_len_noise = self.len_noise // self.n_split
        self.X = interleave(X_noise, X_natural, each_len_noise, each_len_natural)
        self.y = interleave(y_noise, y_natural, each_len_noise, each_len_natural)
        
        self.stats = {}
        self.stats['mean'] = (data_natural.stats['mean'] + data_noise.stats['mean']) / 2
        self.stats['std'] = np.sqrt((data_natural.stats['std']**2 + data_noise.stats['std']**2) / 2)
        
        del X_natural
        del y_natural
        del X_noise
        del y_noise
        
    def __len__(self):
        return self.y.shape[0]
    
    def __getitem__(self, index):
        
        inpt = torch.from_numpy((self.X[index].astype('float32') - self.stats['mean']) / self.stats['std'])
        trgt = torch.from_numpy(self.y[index])
                
        return (inpt, trgt)
    
class TrainDatasetBoth2(Dataset):
    
    def __init__(self, img_shape, data_path, date, stim, val_size, cells='all', **kwargs):
        super().__init__()
        assert stim == 'both'
        data_natural = loadexpt(date, cells, 'naturalscene', 'train',
                                img_shape[0], 0, data_path=data_path)
        data_noise = loadexpt(date, cells, 'fullfield_whitenoise', 'train',
                              img_shape[0], 0, data_path=data_path)
        
        self.val_size = val_size
        self.len_natural = data_natural.y.shape[0]
        self.len_noise = data_noise.y.shape[0]
        self.n_split = 5
        
        self.stats_natural = data_natural.stats
        self.stats_noise = data_noise.stats
        
        X_natural = data_natural.X[:-self.val_size]
        y_natural = data_natural.y[:-self.val_size]
        X_noise = data_noise.X[:-self.val_size]
        y_noise = data_noise.y[:-self.val_size]
        
        mean_array_natural = np.ones((X_natural.shape[0],1,1,1)).astype('float32') * self.stats_natural['mean']
        mean_array_noise = np.ones((X_noise.shape[0],1,1,1)).astype('float32') * self.stats_noise['mean']
        std_array_natural = np.ones((X_natural.shape[0],1,1,1)).astype('float32') * self.stats_natural['std']
        std_array_noise = np.ones((X_noise.shape[0],1,1,1)).astype('float32') * self.stats_noise['std']
        
        each_len_natural = self.len_natural // self.n_split
        each_len_noise = self.len_noise // self.n_split
        self.X = interleave(X_noise, X_natural, each_len_noise, each_len_natural)
        self.y = interleave(y_noise, y_natural, each_len_noise, each_len_natural)
        self.mean_array = interleave(mean_array_noise, mean_array_natural, each_len_noise, each_len_natural)
        self.std_array = interleave(std_array_noise, std_array_natural, each_len_noise, each_len_natural)
        
        del X_natural
        del y_natural
        del X_noise
        del y_noise
        del mean_array_natural
        del mean_array_noise
        del std_array_natural
        del std_array_noise
        
    def __len__(self):
        return self.y.shape[0]
    
    def __getitem__(self, index):
        
        inpt = torch.from_numpy((self.X[index].astype('float32') - self.mean_array[index]) / self.std_array[index])
        trgt = torch.from_numpy(self.y[index])
                
        return (inpt, trgt)