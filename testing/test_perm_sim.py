import numpy as np
import torch
import torchdeepretina as tdr


if __name__=="__main__":
    N = 100
    T = 10000

    x = np.random.randn(T,N)
    x = x*(10**np.arange(N))

    perm = np.random.permutation(N).astype(np.int)
    y = x[:,perm]
    #y = np.concatenate([x[:,perm],np.random.random((T,10))],axis=-1)
    sim = tdr.utils.perm_similarity(y, x, grad_fit=False, patience=10)
    print(sim)
    
    
