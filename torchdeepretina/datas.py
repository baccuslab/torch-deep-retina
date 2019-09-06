"""
Preprocessing utility functions for loading and formatting experimental data
"""
from collections import namedtuple

import h5py
import numpy as np
from os.path import join, expanduser
from scipy.stats import zscore
import pyret.filtertools as ft
import torch

NUM_BLOCKS = {
    '15-10-07': 6,
    '15-11-21a': 6,
    '15-11-21b': 6,
    '16-01-07': 3,
    '16-01-08': 3,
}
CELLS = {
    '15-10-07': [0, 1, 2, 3, 4],
    '15-11-21a': [6, 10, 12, 13],
    '15-11-21b': [0, 1, 3, 5, 8, 9, 13, 14, 16, 17, 18, 20, 21, 22, 23, 24, 25],
    '16-01-07': [0, 2, 7, 10, 11, 12, 31],
    '16-01-08': [0, 3, 7, 9, 11],
    '16-05-31': [2, 3, 4, 14, 16, 18, 20, 25, 27]
}

Exptdata = namedtuple('Exptdata', ['X', 'y', 'spkhist', 'stats', "cells"])
__all__ = ['loadexpt', 'stimcut', 'CELLS', "DataContainer","DataObj", "DataDistributor"]

class DataContainer():
    def __init__(self, data):
        self.X = data.X
        self.y = data.y
        self.stats = data.stats

def loadexpt(expt, cells, filename, train_or_test, history, nskip, cutout_width=None, norm_stats=None, data_path="~/experiments/data"):
    """Loads an experiment from an h5 file on disk

    Parameters
    ----------
    expt : str
        The date of the experiment to load in YY-MM-DD format (e.g. '15-10-07')

    cells : list of ints (or string "all")
        Indices of the cells to load from the experiment
        If "all" is argued, then all cells for the argued expt are used.

    filename : string
        Name of the hdf5 file to load (e.g. 'whitenoise' or 'naturalscene')

    train_or_test : string
        The key in the hdf5 file to load ('train' or 'test')

    history : int
        Number of samples of history to include in the toeplitz stimulus

    nskip : float, optional
        Number of samples to skip at the beginning of each repeat

    cutout_width : int, optional
        If not None, cuts out the stimulus around the STA (assumes cells is a scalar)

    norm_stats : listlike of floats i.e. [mean, std], optional
        If a list of len 2 is argued, idx 0 will be used as the mean and idx 1 the std for
        data normalization

    data_path : string
        path to the data folders
    """
    assert history > 0 and type(history) is int, "Temporal history must be a positive integer"
    assert train_or_test in ('train', 'test'), "train_or_test must be 'train' or 'test'"
    if type(cells) == type(str()) and cells=="all":
        cells = CELLS[expt]

    # get whitenoise STA for cutout stimulus
    if cutout_width is not None:
        assert len(cells) == 1, "cutout must be used with single cells"
        wn = _loadexpt_h5(expt, 'whitenoise')
        sta = np.array(wn['train/stas/cell{:02d}'.format(cells[0]+1)]).copy()
        py, px = ft.filterpeak(sta)[1]

    # load the hdf5 file
    with _loadexpt_h5(expt, filename, root=data_path) as f:

        expt_length = f[train_or_test]['time'].size

        # load the stimulus into memory as a numpy array, and z-score it
        if cutout_width is None:
            stim = np.asarray(f[train_or_test]['stimulus']).astype('float32')
        else:
            stim = ft.cutout(np.asarray(f[train_or_test]['stimulus']), idx=(px, py), width=cutout_width).astype('float32')
        stats = {}
        if norm_stats is not None:
            stats['mean'] = norm_stats[0]
            stats['std'] = norm_stats[1]
        else:
            stats['mean'] = stim.mean()
            stats['std'] = stim.std()+1e-7
        stim = (stim-stats['mean'])/stats['std']

        # apply clipping to remove the stimulus just after transitions
        num_blocks = NUM_BLOCKS[expt] if train_or_test == 'train' and nskip > 0 else 1
        valid_indices = np.arange(expt_length).reshape(num_blocks, -1)[:, nskip:].ravel()

        # reshape into the Toeplitz matrix (nsamples, history, *stim_dims)
        stim_reshaped = rolling_window(stim[valid_indices], history, time_axis=0)

        # get the response for this cell (nsamples, ncells)
        resp = np.array(f[train_or_test]['response/firing_rate_10ms'][cells]).T[valid_indices]
        resp = resp[history:]

        # get the spike history counts for this cell (nsamples, ncells)
        binned = np.array(f[train_or_test]['response/binned'][cells]).T[valid_indices]
        spk_hist = rolling_window(binned, history, time_axis=0)

    return Exptdata(stim_reshaped, resp, spk_hist, stats, cells)


def _loadexpt_h5(expt, filename, root="~/experiments/data"):
    """Loads an h5py reference to an experiment on disk"""
    filepath = join(expanduser(root), expt, filename + '.h5')
    return h5py.File(filepath, mode='r')

def stimcut(data, expt, ci, width=11):
    """Cuts out a stimulus around the whitenoise receptive field"""

    # get the white noise STA for this cell
    wn = _loadexpt_h5(expt, 'whitenoise')
    sta = np.array(wn['train/stas/cell{:02d}'.format(ci+1)]).copy()

    # find the peak of the sta
    xc, yc = ft.filterpeak(sta)[1]

    # cutout stimulus
    X, y = data
    Xc = ft.cutout(X, idx=(yc, xc), width=width)
    yc = y[:, ci].reshape(-1, 1)
    return Exptdata(Xc, yc)


def rolling_window(array, window, time_axis=0):
    """
    Make an ndarray with a rolling window of the last dimension

    Parameters
    ----------
    array : array_like
        Array to add rolling window to

    window : int
        Size of rolling window

    time_axis : int, optional
        The axis of the temporal dimension, either 0 or -1 (Default: 0)

    Returns
    -------
    Array that is a view of the original array with a added dimension
    of size w.

    Examples
    --------
    >>> x=np.arange(10).reshape((2,5))
    >>> rolling_window(x, 3)
    array([[[0, 1, 2], [1, 2, 3], [2, 3, 4]],
           [[5, 6, 7], [6, 7, 8], [7, 8, 9]]])

    Calculate rolling mean of last dimension:

    >>> np.mean(rolling_window(x, 3), -1)
    array([[ 1.,  2.,  3.],
           [ 6.,  7.,  8.]])
    """
    if time_axis == 0:
        array = array.T

    elif time_axis == -1:
        pass

    else:
        raise ValueError('Time axis must be 0 (first dimension) or -1 (last)')

    assert window >= 1, "`window` must be at least 1."
    assert window < array.shape[-1], "`window` is too long."

    # with strides
    shape = array.shape[:-1] + (array.shape[-1] - window, window)
    strides = array.strides + (array.strides[-1],)
    arr = np.lib.stride_tricks.as_strided(array, shape=shape, strides=strides)

    if time_axis == 0:
        return np.rollaxis(arr.T, 1, 0)
    else:
        return arr

class DataObj:
    def __init__(self, data, idxs):
        self.data = data
        self.idxs = idxs
        self.shape = [len(idxs), *data.shape[1:]]

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self,idxs):
        return self.data[self.idxs[idxs]]

    def __call__(self,idxs):
        return self.data[self.idxs[idxs]]

class DataDistributor:
    """
    This class is used to abstract away the manipulations required for shuffling or organizing data for rnns.
    """

    def __init__(self, data, val_size=30000, batch_size=512, seq_len=1, shuffle=True, rand_sample=None, recurrent=False, shift_labels=False):
        """
        data - a class or named tuple containing an X and y member variable.
        val_size - the number of samples dedicated to validation
        batch_size - size of batches yielded by train_sample generator
        seq_len - int describing how long the data sequences should be.
                    if not doing recurrent training, set to 1
        shuffle - bool describing if data should be shuffled (the shuffling
                    preserves the frame order within each data point)
        rand_sample - bool similar to shuffle but specifies the sampling method rather
                        than the storage method. Allows for the data to be
                        unshuffled but sampled randomly. Defaults to the state of shuffle
                        if left as None
        recurrent - bool describing if model is recurrent
        shift_labels - bool describing if labels should be shifted for null model training
        """
        self.batch_size = batch_size
        self.is_torch = False
        self.seq_len = seq_len
        self.shuffle = shuffle
        self.rand_sample = rand_sample
        self.recurrent = recurrent
        self.X = data.X
        self.y = data.y
        rand_sample = shuffle if rand_sample is None else rand_sample

        if seq_len > 1:
            self.X = rolling_window(self.X, seq_len)
            self.y = rolling_window(self.y, seq_len)
        if recurrent and not shuffle:
            self.X = self.order_into_batches(self.X, batch_size)
            self.y = self.order_into_batches(self.y, batch_size)
        if type(self.X) == type(np.array([])):
            if shuffle:
                self.perm = np.random.permutation(self.X.shape[0]).astype('int')
            else:
                self.perm = np.arange(self.X.shape[0]).astype('int')
        else:
            if shuffle:
                self.perm = torch.randperm(self.X.shape[0]).long()
            else:
                self.perm = torch.arange(self.X.shape[0]).long()

        if shift_labels:
            self.y = np.roll(self.y, len(self.y)//2, axis=0)
        if val_size > len(self.perm):
            val_size = int(len(self.perm)*0.05)
        self.train_idxs = self.perm[:-val_size]
        self.val_idxs = self.perm[-val_size:]
        self.train_X = DataObj(self.X, self.train_idxs)
        self.train_y = DataObj(self.y, self.train_idxs)
        self.val_X = DataObj(self.X, self.val_idxs)
        self.val_y = DataObj(self.y, self.val_idxs)
        self.train_shape = (len(self.train_idxs), *self.X.shape[1:])
        self.val_shape = (len(self.val_idxs), *self.X.shape[1:])
        if self.recurrent and not self.shuffle:
            self.n_loops = self.train_shape[0]
        else:
            self.n_loops = self.train_shape[0]//batch_size

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[self.perm[idx]]

    def order_into_batches(self, data, batch_size):
        switch_back = False
        if type(data) != type(np.array([])):
            data = data.numpy()
            switch_back = True
        length = data.shape[0]//batch_size
        tot_len = length*batch_size
        trunc = data[:tot_len].reshape(batch_size, length, *data.shape[1:])
        trans_order = list(range(len(trunc.shape)))
        trans_order[0], trans_order[1] = trans_order[1], trans_order[0]
        trunc = trunc.transpose(trans_order)
        if switch_back:
            trunc = torch.from_numpy(trunc)
        return trunc

    def undo_batch_order(self, data):
        switch_back = False
        if type(data) == type(np.array([])):
            data = torch.from_numpy(data)
            switch_back = True
        trans_idxs = list(range(len(data.shape)))
        trans_idxs[0], trans_idxs[1] = trans_idxs[1], trans_idxs[0]
        data = data.permute(*trans_idxs)
        data = data.view(-1, *data.shape[2:])
        if switch_back:
            data = data.numpy()
        return data

    def set_batch_size(self, batch_size):
        self.batch_size = batch_size
        self.n_loops = self.train_shape[0]//batch_size

    def val_sample(self, step_size):
        val_X = self.val_X
        val_y = self.val_y
        n_loops = self.val_shape[0]//step_size
        arange = torch.arange(n_loops*step_size).long()
        for i in range(n_loops):
            if self.recurrent:
                idxs = i
            else:
                idxs = arange[i*step_size:(i+1)*step_size]
            yield val_X[idxs], val_y[idxs]

    def train_sample(self, batch_size=None):
        n_loops = self.n_loops
        # Allow for new batchsize argument
        if batch_size is None:
            batch_size = self.batch_size
        elif not (self.recurrent and not self.shuffle):
            n_loops = self.train_shape[0]//batch_size
        else:
            print("batch_size arg ignored")
            batch_size = self.batch_size
        if self.rand_sample:
            batch_perm = torch.randperm(self.train_shape[0]).long()
        else:
            batch_perm = torch.arange(self.train_shape[0]).long()
        for i in range(n_loops):
            if self.recurrent and not self.shuffle:
                idxs = i
            else:
                idxs = batch_perm[i*batch_size:(i+1)*batch_size]
            yield self.train_X[idxs], self.train_y[idxs]
    
    def torch(self):
        self.is_torch = True
        self.X = torch.FloatTensor(self.X)
        self.y = torch.FloatTensor(self.y)
        self.perm = torch.LongTensor(self.perm)
        self.train_idxs = self.perm[:-self.val_shape[0]]
        self.val_idxs = self.perm[-self.val_shape[0]:]
        self.train_X = DataObj(self.X, self.train_idxs)
        self.train_y = DataObj(self.y, self.train_idxs)
        self.val_X = DataObj(self.X, self.val_idxs)
        self.val_y = DataObj(self.y, self.val_idxs)

    def numpy(self):
        self.is_torch = False
        self.X = np.asarray(self.X)
        self.y = np.asarray(self.y)
        self.perm = np.asarray(self.perm).astype('int')
        self.train_idxs = self.perm[:-self.val_shape[0]]
        self.val_idxs = self.perm[-self.val_shape[0]:]
        self.train_X = DataObj(self.X, self.train_idxs)
        self.train_y = DataObj(self.y, self.train_idxs)
        self.val_X = DataObj(self.X, self.val_idxs)
        self.val_y = DataObj(self.y, self.val_idxs)

