import numpy as np
import torch
import torchdeepretina as tdr


if __name__=="__main__":
    N = 1000
    T = 10000

    for i in range(20):
        # Fails at i==11ish
        x = np.random.random((T,N))*10**i

        perm = np.random.permutation(N).astype(np.int)
        y = x[:,perm]
        sim = tdr.utils.perm_similarity(x, y, grad_fit=True, patience=10)
        print(i,sim)
    
    
