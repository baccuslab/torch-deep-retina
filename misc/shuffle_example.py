from tqdm import tqdm as tq
import torch
import numpy as np
from deepretina.experiments import loadexpt
import sys

sys.path.append('../')
sys.path.append('../utils/')
from models.BN_CNN import BNCNN

def load_data():
    train_data = loadexpt('15-10-07',[0,1,2,3,4],'naturalscene','train',40,0)
    return train_data.X, train_data.y

def epoch_loop(x,y,bs,num_epochs,fraction_validate=0.1,DEVICE='cuda:1'):
    model = BNCNN()
    model = model.to(DEVICE)

    assert(x.shape[0] == y.shape[0])
    indices = np.random.permutation(x.shape[0])
    print('Shape of data: {}, Shape of permuted indices: {}'.format(x.shape,indices.shape))
   

    num_val = int(x.shape[0]*fraction_validate)
    val_indxs = indices[:num_val]
    
    val_x = x[val_indxs]
    val_y = y[val_indxs]
    
    test = np.in1d(indices,val_indxs)
    test = np.where(test)[0]
    no_val_indices = np.delete(indices,test,axis=0)
    print(no_val_indices.shape)

    print('Validation X Shape: {}, Validation y Shape: {}'.format(val_x.shape,val_y.shape))

    

    for epoch in tq(range(num_epochs)):
        indices = np.random.permutation(no_val_indices.shape[0])
        indices = no_val_indices[indices] 


        print('Train X Shape: {}, Train y Shape: {}'.format(x.shape,y.shape))
        num_batches, leftover = divmod(x.shape[0], bs)

        for batch in tq(range(num_batches)):
            batch_idx = indices[bs*batch:bs*(batch+1)]


            batch_x = x[batch_idx]
            batch_y = y[batch_idx]
            
            batch_x = torch.FloatTensor(batch_x)
            batch_y = torch.FloatTensor(batch_y)

            batch_x = batch_x.to(DEVICE)
            batch_y = batch_y.to(DEVICE)

            batch_out = model(batch_x)


if __name__ == '__main__':
    x,y = load_data()
    epoch_loop(x,y,bs=512,num_epochs=1,fraction_validate=0.1,DEVICE='cuda:0')
