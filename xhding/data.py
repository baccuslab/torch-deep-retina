import numpy as np
import sys
sys.path.insert(0, '/home/xhding/workspaces/torch-deep-retina/')
import torch
from torch.utils.data.sampler import Sampler
from  torch.utils.data.dataset import Dataset
from torchdeepretina.datas import loadexpt

class BatchRnnSampler(Sampler):
    
    def __init__(self, sampler, batch_size):
        self.sampler = sampler
        self.batch_size = batch_size

    def __iter__(self):
        for batch_idx in range(self.__len__()):
            batch = [batch_idx + n * self.__len__() for n in range(self.batch_size)]
            yield batch

    def __len__(self):
        return len(self.sampler) // self.batch_size
    
class TrainDataset(Dataset):
    
    def __init__(self, data_path='/home/salamander/experiments/data/', val_size=30000):
        super().__init__()
        data = loadexpt('15-10-07', [0,1,2,3,4], 'naturalscene', 'train',
                        40, 0, data_path=data_path)
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
    
    def __init__(self, data_path='/home/salamander/experiments/data/', val_size=30000):
        super().__init__()
        data = loadexpt('15-10-07', [0,1,2,3,4], 'naturalscene', 'train',
                        40, 0, data_path=data_path)
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