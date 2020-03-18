import numpy as np
import torch
import torchdeepretina as tdr


if __name__=="__main__":
    N = 1000
    T = 10000

    # Fails at i==11ish
    x = np.random.random((T,N))

    perm = np.random.permutation(N).astype(np.int)
    y = np.concatenate([x[:,perm],np.random.random((T,10))],axis=-1)
    sim = tdr.utils.perm_similarity(y, x, grad_fit=True, patience=10)
    print(sim)
    
    
