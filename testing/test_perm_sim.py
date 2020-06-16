import numpy as np
import torch
import torchdeepretina as tdr


if __name__=="__main__":
    N = 4
    T = 10000
    S = 5

    x = np.random.randn(T,N)
    #x = x*(10**np.arange(N))
    perm = np.random.permutation(N).astype(np.int)
    y = x[:,perm]
    sim = tdr.utils.perm_similarity(x, y, grad_fit=False, patience=10,
                                                        vary_space=False)
    print("sim1:", sim)

    print("perm",perm)
    y = np.random.random((T,N,S))
    ss = np.random.randint(0,S,N)
    print("ss:", ss)
    for i in range(N):
        s = ss[i]
        y[:,i,s] = x[:,perm[i]]

    #y = np.concatenate([x[:,perm],np.random.random((T,10))],axis=-1)
    sim = tdr.utils.perm_similarity(x, y, grad_fit=False, patience=10,
                                                        vary_space=True)
    print(sim)
    
    
