import sys
import torchdeepretina as tdr
import os
import torch
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import scipy.stats
import numpy as np

if __name__ == "__main__":
    m1 = np.random.randn(50,100)
    m2 = np.random.randn(50,100)
    mtx = np.zeros((m1.shape[1],m2.shape[1]))
    for i in range(m1.shape[1]):
        for j in range(m2.shape[1]):
            r,_ = scipy.stats.pearsonr(m1[:,i],m2[:,j])
            mtx[i,j] = r
    plt.figure()
    plt.subplot(2,1,1)
    plt.imshow(mtx)
    print(mtx[:10,:10])

    test_mtx = tdr.utils.mtx_cor(m1,m2,batch_size=10,to_numpy=True)
    print("sqr error:", ((mtx-test_mtx)**2).sum())
    print(test_mtx[:10,:10])
    plt.subplot(2,1,2)
    plt.imshow(test_mtx)
    plt.show()
