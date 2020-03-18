"""
Preprocessing utility functions for loading and formatting
experimental data
"""
from collections import namedtuple

import h5py
import numpy as np
import os
import re
from os.path import join, expanduser
from scipy.stats import zscore
import pyret.filtertools as ft
import torch
import torchdeepretina.stimuli as tdrstim
from torchdeepretina.stimuli import rolling_window
import torchdeepretina.utils as utils

NUM_BLOCKS = {
    '15-10-07': 6,
    '15-11-21a': 6,
    '15-11-21b': 6,
    'arbfilt': 1,
    '16-01-07': 3,
    '16-01-08': 3,
}
CELLS = {
    '15-10-07': [0, 1, 2, 3, 4],
    '15-11-21a': [6, 10, 12, 13],
    '15-11-21b': [0, 1, 3, 5, 8, 9, 13, 14, 16, 17, 18, 20, 21, 22,
                                                       23, 24, 25],
    '16-01-07': [0, 2, 7, 10, 11, 12, 31],
    '16-01-08': [0, 3, 7, 9, 11],
    '16-05-31': [2, 3, 4, 14, 16, 18, 20, 25, 27],
    'arbfilt': [0,1,2,3,4,5,6,7,8,9]
}
CENTERS = {
    "bipolars_late_2012":  [(19, 22), (18, 20), (20, 21), (18, 19)],
    "bipolars_early_2012": [(17, 23), (17, 23), (18, 23)],
    "amacrines_early_2012":[(18, 23), (19, 24), (19, 23), (19, 23),
                                                          (18, 23)],
    "amacrines_late_2012":[ (20, 20), (22, 19), (20, 20), (19, 22),
                            (22, 25), (19, 21), (19, 17), (20, 19),
                            (17, 20), (19, 23), (17, 20), (17, 20),
                            (20, 19), (18, 18), (19, 17), (17, 17),
                            (18, 20), (20, 19), (17, 19), (19, 18),
                            (17, 17), (25, 15)
                          ],
    '15-10-07':  [[21,18], [24,20], [22,18], [27,18], [31,20]],
    '15-11-21a': [[37,3], [39,6], [20,15], [41,2]],
    '15-11-21b': [[16,13], [20,12], [19,7],[19,3], [21,7], [22,6],
                  [21,7], [25,3], [23,6], [26,4], [24,6], [26,7],
                  [27,9], [26,9], [27,11], [26,13], [24,10]],
    '16-01-07': None,
    '16-01-08': None,
    'arbfilt': None,
}
CENTERS_DICT = {
     "bipolars_late_2012":  {i:c for i,c in\
                          enumerate(CENTERS["bipolars_late_2012"])},
     "bipolars_early_2012":  {i:c for i,c in\
                          enumerate(CENTERS["bipolars_early_2012"])},
     "amacrines_early_2012":  {i:c for i,c in\
                          enumerate(CENTERS["amacrines_early_2012"])},
     "amacrines_late_2012":  {i:c for i,c in\
                          enumerate(CENTERS["amacrines_late_2012"])},
        '15-10-07':  {0:[21,18], 1:[24,20], 2:[22,18], 3:[27,18],
                      4:[31,20]},
        '15-11-21a': {6:[37,3], 10:[39,6], 12:[20,15], 13:[41,2]},
        '15-11-21b': {0:[16,13], 1:[20,12], 3:[19,7], 5:[19,3],
                      8:[21,7], 9:[22,6], 13:[21,7], 14:[25,3],
                      16:[23,6],17:[26,4],18:[24,6],20:[26,7],
                      21:[27,9], 22:[26,9], 23:[27,11], 24:[26,13],
                      25:[24,10]},
        '16-01-07': dict(),
        '16-01-08': dict(),
        'arbfilt': dict(),
}

INTR_FILES = {'bipolars_late_2012', 'bipolars_early_2012',
              'amacrines_early_2012','amacrines_late_2012',
              'horizontals_early_2012', 'horizontals_late_2012'}

Exptdata = namedtuple('Exptdata', ['X','y','spkhist','stats',"cells",
                                                          "centers"])
__all__ = ['loadexpt','CELLS',"CENTERS","DataContainer","DataObj",
                                                "DataDistributor"]

class DataContainer():
    def __init__(self, data):
        self.X = data.X
        self.y = data.y
        self.centers = data.centers
        self.stats = data.stats
    
    def __len__(self):
        return len(self.X)

def loadexpt(expt, cells, filename, train_or_test, history, nskip=0,
                                     cutout_width=None, 
                                     norm_stats=None, 
                                     data_path="~/experiments/data"):
    """
    Loads an experiment from an h5 file on disk

    Parameters
    ----------
    expt : str
        The date of the experiment to load in YY-MM-DD format
        (e.g. '15-10-07')

    cells : list of ints (or string "all")
        Indices of the cells to load from the experiment
        If "all" is argued, all cells for the argued expt are used.

    filename : string
        Name of the hdf5 file to load (e.g. 'whitenoise' or
        'naturalscene')

    train_or_test : string
        The key in the hdf5 file to load ('train' or 'test')

    history : int
        Number of samples of history to include in the toeplitz
        stimulus. If None, no history dimension is created.

    nskip : float, optional
        Number of samples to skip at the beginning of each repeat

    cutout_width : int, optional
        If not None, cuts out the stimulus around the STA 
        (assumes cells is a scalar)

    norm_stats : listlike of floats or dict i.e. [mean, std], optional
        If a list of len 2 is argued, idx 0 will be used as the mean 
        and idx 1 the std for data normalization. if dict, use 'mean' 
        and 'std' as keys

    data_path : string
        path to the data folders
    """
    assert train_or_test in ('train', 'test'), "train_or_test must\
                                              be 'train' or 'test'"
    if type(cells) == type(str()) and cells=="all":
        cells = utils.try_key(CELLS,expt,default=None)

    # load the hdf5 file
    with _loadexpt_h5(expt, filename, root=data_path) as f:

        expt_length = f[train_or_test]['time'].size

        # load the stimulus into memory as a numpy array, and z-score
        if cutout_width is None:
            stim = np.asarray(f[train_or_test]['stimulus'])
            stim = stim.astype('float32')
        else:
            arr = np.asarray(f[train_or_test]['stimulus'])
            assert len(cells) == 1, "only 1 cell allowed for cutout"
            center = CENTERS_DICT[expt][cells[0]]
            stim = tdrstim.get_cutout(arr, center=center,
                                       span=cutout_width,
                                       pad_to=arr.shape[-1])
            stim = stim.astype('float32')
        stats = {}
        if norm_stats is not None:
            if isinstance(norm_stats, dict):
                stats = {k:v for k,v in norm_stats.items()}
            else:
                stats['mean'] = norm_stats[0]
                stats['std'] = norm_stats[1]
        else:
            stats['mean'] = stim.mean()
            stats['std'] = stim.std()+1e-7
        stim = (stim-stats['mean'])/stats['std']

        # apply clipping to remove the stimulus just after transitions
        num_blocks = 1 if not (train_or_test=='train' and nskip>0)\
                              else utils.try_key(NUM_BLOCKS,expt,1)
        valid_indices = np.arange(expt_length).reshape(num_blocks,-1)
        valid_indices = valid_indices[:, nskip:].ravel()

        # reshape into the Toeplitz matrix (nsamps, hist, # *stim_dim)
        stim_reshaped = stim[valid_indices]
        if history is not None and history > 0:
            stim_reshaped = rolling_window(stim_reshaped, history,
                                                          time_axis=0)

        # get the response for this cell (nsamples, ncells)
        s = 'response/firing_rate_10ms'
        if cells is None:
            cells = list(range(f[train_or_test][s].shape[0]))
        stream = f[train_or_test][s][cells]
        resp = np.array(stream).T[valid_indices]
        if history is not None and history > 0:
            resp = resp[history:]

        # get the spike history counts for this cell (nsamp, ncells)
        stream = f[train_or_test]['response/binned'][cells]
        spk_hist = np.array(stream).T[valid_indices]
        if history is not None and history > 0:
            spk_hist = rolling_window(spk_hist, history, time_axis=0)

        # get the ganglion cell receptive field centers
        if expt in CENTERS:
            centers = np.asarray([CENTERS_DICT[expt][c] for c in cells])
        else:
            centers = None

    return Exptdata(stim_reshaped, resp, spk_hist, stats, cells,
                                                       centers)

def _loadexpt_h5(expt, filename, root="~/experiments/data"):
    """Loads an h5py reference to an experiment on disk"""
    filepath = join(expanduser(root), expt, filename + '.h5')
    return h5py.File(filepath, mode='r')

def prepare_stim(stim, stim_type):
    """
    Used to prepare the interneuron stimulus for the model

    stim: ndarray
    stim_type: str
        preparation method
    """
    if stim_type == 'boxes':
        return 2*stim - 1
    elif stim_type == 'flashes':
        stim = stim.reshape(stim.shape[0], 1, 1)
        return np.broadcast_to(stim, (stim.shape[0], 38, 38))
    elif stim_type == 'movingbar':
        stim = block_reduce(stim, (1,6), func=np.mean)
        stim = stim.reshape(stim.shape[0], stim.shape[1], 1)
        stim = pyret.stimulustools.upsample(stim, 5)[0]
        return np.broadcast_to(stim, (stim.shape[0], stim.shape[1],
                                                    stim.shape[1]))
    elif stim_type == 'lines':
        fxn = lambda m: np.convolve(m,0.5*np.ones((2,)),mode='same')
        stim_averaged = np.apply_along_axis(fxn, axis=1, arr=stim)
        stim = stim_averaged[:,::2]
        # now stack stimulus to convert 1d to 2d spatial stimulus
        shape = (-1,1,stim.shape[-1])
        return stim.reshape(shape).repeat(stim.shape[-1], axis=1)
    else:
        print("Invalid stim type")
        assert False

def loadintrexpt(expt, cells, stim_type, history, H=None, W=None,
                                     cutout_width=None,
                                     norm_stats=None,
                                     train_or_test="train",
                                     data_path="~/interneuron_data",
                                     cross_val_idx=None,
                                     n_cv_folds=None):
    """
    Loads an experiment from an h5 file on disk

    Parameters
    ----------
    expt : str
        The name of the interneuron file with or without the extension
        (e.g. 'bipolars_late_2012')

    cells : list of ints (or string "all")
        Indices of the cells to load from the experiment
        If "all" is argued, all cells for the argued expt are used.

    stim_type : string
        Name of the hdf5 file to load (e.g. 'whitenoise' or
        'naturalscene')

    history : int
        Number of samples of history to include in the toeplitz
        stimulus. If None, no history dimension is created.

    H : int
        height of stimulus

    W : int
        width of stimulus. if none, defaults to H

    cutout_width : int, optional
        If not None, cuts out the stimulus around the STA 
        (assumes `cells` is a scalar)

    norm_stats : listlike of floats or dict i.e. [mean, std], optional
        If a list of len 2 is argued, idx 0 will be used as the mean 
        and idx 1 the std for data normalization. if dict, use 'mean' 
        and 'std' as keys

    train_or_test: str (valid: 'train','test')
        determines if test data or train data will be returned. if
        test data is selected and it is interneuron data being
        returned then lines stimulus is included in data. if using
        cross_val_idx and n_cv_folds, 'test' will return the isolated
        portion of the data. otherwise the large portion will be
        returned.

    data_path : string
        path to the data folders

    cross_val_idx : int or None
        if int, the data will be split such that 1 of the n_cv_folds
        splits is kept seperate from the rest of the data. this
        cross_val_idx parameter determines the index of the held out
        portion. depending on the value of train_or_test, the isolated
        portion or the large portion will be returned. 'test' returns
        the isolated portion.

    n_cv_folds : int or None
        if None, cross validation will be ignored. if int, denotes the
        number of splits of the data to be completed.
    """
    if ".h5" != expt[-3:]: expt = expt + ".h5"
    files = [expt]
    if stim_type == "naturalscene" or stim_type == "whitenoise":
        stim_keys = {"boxes"}
    elif stim_type == "all":
        stim_keys = {"boxes", "lines"}
    else:
        stim_keys = {stim_type}
    if train_or_test == "test":
        stim_keys.add("lines")

    if history is None: history = 0
    train_stim,mem_pots,_ = load_interneuron_data(data_path,
                                                files=files,
                                                filter_length=history,
                                                stim_keys=stim_keys,
                                                join_stims=True,
                                                window=True)
    if H is None: H = train_stim[files[0]].shape[1]
    if W is None: W = H
    train_stim = tdrstim.spatial_pad(train_stim[files[0]], H=H, W=W)
    mem_pot = mem_pots[files[0]].T

    if n_cv_folds is not None and cross_val_idx is not None:
        train_stims, mem_pots = cv_split([train_stim, mem_pot],
                                         cv_idx=cross_val_idx,
                                         n_folds=n_cv_folds)
        idx = 1 if train_or_test == "test" else 0
        train_stims, mem_pots = train_stims[idx], mem_pots[idx]

    if cells=="all":
        cells = list(range(mem_pots.shape[1]))
    elif isinstance(cells,  int):
        cells = [cells]
    mem_pot = mem_pot[:,cells]

    stats = norm_stats
    if stats is None:
        stats = {"mean":train_stim.mean(),
                 "std":train_stim.std()+1e-5}
    elif isinstance(stats, list):
        keys = ['mean', 'std']
        stats = {k:v for k,v in zip(keys, stats)}
    train_stim = (train_stim - stats['mean'])/stats['std']

    centers = [CENTERS[expt[:-3]][c] for c in cells]
    if cutout_width is not None and cutout_width > 0:
        assert len(cells) == 1, "only 1 cell allowed for cutout"
        center = CENTERS[expt][cells[0]]
        train_stim = tdrstim.get_cutout(train_stim, center=center,
                                   span=cutout_width,
                                   pad_to=train_stim.shape[-1])
        train_stim = train_stim.astype('float32')
    return Exptdata(train_stim, mem_pot, None, stats, cells, centers)

def load_test_data(hyps,verbose=False):
    """
    Used to load testing data for models either trained on ganglion
    cells or interneurons.

    hyps: dict
        keys: str
            'dataset'
            
    """
    if hyps['dataset'][-3:] == ".h5":
        hyps['dataset'] = hyps['dataset'][:-3]
    if verbose:
        print("Dataset:", hyps['dataset'])
    if hyps['dataset'] in INTR_FILES:
        data = loadintrexpt(expt=hyps['dataset'],
                                  cells=hyps['cells'],
                                  stim_type=hyps['stim_type'],
                                  train_or_test="test",
                                  history=hyps['img_shape'][0],
                                  H=hyps['img_shape'][1],
                                  norm_stats=hyps['norm_stats'])
    else:
        data = loadexpt(expt=hyps['dataset'],
                              cells=hyps['cells'],
                              filename=hyps['stim_type'],
                              train_or_test="test",
                              history=hyps['img_shape'][0],
                              norm_stats=hyps['norm_stats'])
    return data

def load_interneuron_data(root_path="~/interneuron_data/",
                          files=None,
                          filter_length=40,
                          stim_keys={"boxes"},
                          join_stims=False,
                          trunc_join=True,
                          window=False):
    """ 
    Load data

    root_path: str
        path to folder that contains the interneuron h5 files
    files: list
        a list of the desired interneuron h5 file names
    filter_length: int
        length of first layer filters of model
    stim_keys: set of str
        the desired stimulus types
    join_stims: bool
       combines the stimuli listed in stim_keys
    trunc_join: bool
       truncates the joined stimuli to be of equal length. 
       Only applies if join_stims is true.
    window: bool
        if true, windows the stimulus with the argued filter_length

    Returns:
        if using join_stims then no stim_type key exists!!!

        stims - dict
            keys are the cell files, vals are dicts
            keys of subdicts are stim type with vals of ndarray
            stimuli (T,H,W)
        mem_pots - dict
            keys are the cell files, vals are dicts
                keys of subdicts are stim type with values of ndarray
                membrane potentials for each cell within the
                file (N_CELLS, T-filter_length)
    """
    if filter_length==0:
        window=False
    if files is None:
        files = [f+".h5" for f in INTR_FILES]
    full_files = [os.path.expanduser(os.path.join(root_path, name))\
                                                  for name in files]
    file_ids = []
    for f in full_files:
        file_ids.append(re.split('_|\.', f)[0])
    num_pots = []
    stims = dict()
    mem_pots = dict()
    for fi,file_name in zip(full_files,files):
        stims[file_name] = None if join_stims else dict()
        mem_pots[file_name] = None if join_stims else dict()
        if join_stims:
            shapes = []
            mem_shapes = []
        with h5py.File(fi,'r') as f:
            for k in f.keys():
                if k in stim_keys:
                    if join_stims:
                        arr = np.asarray(f[k+'/stimuli'])
                        shape = prepare_stim(arr, k).shape
                        shapes.append(shape)
                        s = '/detrended_membrane_potential'
                        mem_pot = np.asarray(f[k+s])
                        mem_shapes.append(mem_pot.shape)
                        del mem_pot
                    else:
                        try:
                            temp = f[k+'/stimuli']
                            temp = np.asarray(temp, dtype=np.float32)
                            stims[file_name][k]=prepare_stim(temp, k)
                            if window:
                                s = rolling_window(stims[file_name][k],
                                                   filter_length)
                                stims[file_name][k] = s
                            s = 'detrended_membrane_potential'
                            temp = np.asarray(f[k][s])
                            temp = temp[:,filter_length:]
                            temp = temp.astype(np.float32)
                            mem_pots[file_name][k] = temp
                            del temp
                        except Exception as e:
                            print(e)
                            print("stim error at", k)
            if join_stims:
                # Summing up length of first dimension of all stimuli
                if trunc_join:
                    trunc_len = np.min([s[0] for s in shapes])
                    zero_dim = [trunc_len*len(shapes)]
                else:
                    zero_dim=[s[0] for i,s in enumerate(shapes)]
                one_dim = [s[1] for s in shapes]
                two_dim = [s[2] for s in shapes]
                if window:
                    shape = [np.sum(zero_dim)-filter_length*len(shapes),
                                              filter_length,
                                              np.max(one_dim),
                                              np.max(two_dim)]
                else:
                    shape = [np.sum(zero_dim), np.max(one_dim),
                                               np.max(two_dim)]
                stims[file_name] = np.empty(shape, dtype=np.float32)

                zero_dim = [s[0] for s in mem_shapes] # Num cells
                mem_shape = [np.max(zero_dim), shape[0]]
                if not window: 
                    mem_shape[1] -= filter_length
                mem_pots[file_name] = np.empty(mem_shape,
                                        dtype=np.float32)

                startx = 0
                mstartx = 0
                for i,k in enumerate(stim_keys):
                    prepped = np.asarray(f[k+'/stimuli'])
                    prepped = prepare_stim(prepped, k)
                    if trunc_join:
                        prepped = prepped[:trunc_len]
                    # In case stim have varying spatial dimensions
                    if not (prepped.shape[-2] ==\
                                      stims[file_name].shape[-2] and\
                                      prepped.shape[-1] ==\
                                      stims[file_name].shape[-1]):
                        prepped = tdrstim.spatial_pad(prepped,
                                          stims[file_name].shape[-2],
                                          stims[file_name].shape[-1])
                    if window:
                        prepped = rolling_window(prepped,
                                                 filter_length)
                    endx = startx+len(prepped)
                    stims[file_name][startx:endx] = prepped
                    s = 'detrended_membrane_potential'
                    mem_pot = np.asarray(f[k][s])
                    if trunc_join:
                        mem_pot = mem_pot[:,:trunc_len]
                    if i == 0 or window:
                        mem_pot = mem_pot[:,filter_length:]
                    mendx = mstartx+mem_pot.shape[1]
                    mem_pots[file_name][:,mstartx:mendx] = mem_pot
                    startx = endx
                    mstartx = mendx
    return stims, mem_pots, full_files

def cv_split(Xs, cv_idx, n_folds):
    """
    Splits indices into validation and training

    Xs: list of list-likes
        a list of data lists to be split in the same way. ie the
        first element of Xs is X and the second element is
        Y. Must all be of same length in first dimension
    cv_idx: int or None
        cross validation index. if int, the data will be split
        such that 1 of the n_folds splits is kept seperate from
        the rest of the data. this cross_val_idx parameter
        determines the index of the held out portion. depending on
        the value of train_or_test, the isolated portion or the
        large portion will be returned. 'test' returns the
        isolated portion.
    n_folds: int or None
        if None, cross validation will be ignored. if int, denotes the
        number of splits of the data to be completed.
    """
    data_len = len(Xs[0])
    cv_p = 1/n_folds
    fold_size = data_len*cv_p
    splits = []
    startx = int(cv_idx*fold_size)
    endx = int(startx+fold_size)
    for X in Xs:
        s_ = np.zeros(len(X)).astype(np.bool)
        s_[startx:endx] = True
        not_fold = X[~s_]
        fold = X[s_]
        splits.append([not_fold, fold])
    return splits

class DataObj:
    def __init__(self, data, idxs):
        self.data = data
        self.idxs = idxs
        self.shape = [len(idxs), *data.shape[1:]]

    def reshape(self, *args):
        args = list(args)
        if type(args[0]) == type(list()) or\
                                    type(args[0]) == type(tuple()):
            args = [*args[0]]
        # Reshape must not change indexing axis
        assert args[0] == -1 or args[0] == len(self.idxs) 
        others = []
        for i,arg in enumerate(args):
            if arg != -1:
                others.append(arg)
        if len(others) < len(args):
            num = int(np.prod(self.shape)/np.prod(others))
            args = [arg if arg != -1 else num for arg in args]
        self.shape = [len(self.idxs),*args[1:]]
        self.data = self.data.reshape(-1,*args[1:])
        return self.data[self.idxs]

    def mean(self, dim=None, axis=None):
        if dim is not None:
            return self.data[self.idxs].mean(dim)
        if axis is not None:
            return self.data[self.idxs].mean(axis)
        return self.data[self.idxs].mean()

    def std(self, dim=None, axis=None):
        if dim is not None:
            return self.data[self.idxs].std(dim)
        if axis is not None:
            return self.data[self.idxs].std(axis)
        return self.data[self.idxs].std()

    def roll(self, amt, idxs):
        """
        rolls the data before indexing. Used for creating null models.

        amt - int
            the amount to roll the dataset
        idxs - list of ints
            the indexes to return from the rolled data
        """
        rolled_idxs = (self.idxs[idxs]+amt)%len(self.data)
        return self.data[rolled_idxs]

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self,idxs):
        return self.data[self.idxs[idxs]]

    def __call__(self,idxs):
        return self.data[self.idxs[idxs]]

class DataDistributor:
    """
    This class is used to abstract away the manipulations required for
    shuffling or organizing data for rnns.
    """

    def __init__(self, data, batch_size=512, seq_len=1,
                                    shuffle=True,
                                    cross_val_idx=0,
                                    n_cv_folds=10,
                                    rand_sample=None,
                                    recurrent=False,
                                    shift_labels=False,
                                    zscorey=False):
        """
        data: a class or named tuple containing an X and y member
            variable.
        val_size - the number of samples dedicated to validation
        batch_size - size of batches yielded by train_sample generator
        seq_len - int describing how long the data sequences should
            be. if not doing recurrent training, set to 1
        shuffle - bool describing if data should be shuffled
            (the shuffling preserves the frame order within each
            data point)
        cross_val_idx - int describing the cross validation rotation.
        n_cv_folds - int descibing the number of cross validation
            folds to be used in cross_validation.
        rand_sample - bool similar to shuffle but specifies the
            sampling method rather than the storage method. Allows
            for the data to be unshuffled but sampled randomly.
            Defaults to the state of shuffle if left as None
        recurrent - bool describing if model is recurrent
        shift_labels - bool describing if labels should be shifted
            for null model training
        """
        self.batch_size = batch_size
        self.is_torch = False
        self.seq_len = seq_len
        self.shuffle = shuffle
        self.rand_sample = shuffle if rand_sample is None else\
                                                    rand_sample
        self.recurrent = recurrent
        self.shift_labels = shift_labels
        self.X = data.X
        self.y = data.y
        if isinstance(self.X, torch.Tensor):
            self.X = np.asarray(self.X)
        if isinstance(self.y, torch.Tensor):
            self.y = np.asarray(self.y)

        if shift_labels:
            self.y = self.shift_in_groups(self.y, group_size=200)
        if zscorey:
            self.y_mean = self.y.mean()
            self.y_std = self.y.std()
            self.y = (self.y-self.y_mean)/(self.y_std+1e-5)
        else:
            self.y_mean = None
            self.y_std =  None

        if seq_len > 1:
            self.X = rolling_window(self.X, seq_len)
            self.y = rolling_window(self.y, seq_len)
        if recurrent:
            self.X = self.order_into_batches(self.X, batch_size)
            self.y = self.order_into_batches(self.y, batch_size)
        if shuffle:
            self.perm = np.random.permutation(self.X.shape[0])
            self.perm = self.perm.astype('int')
        else:
            self.perm = np.arange(self.X.shape[0]).astype('int')

        # Split indices into validation and training
        # If using shuffled indices, seeding is handled before cross
        # validation setup, so the random index order is preserved
        # between different folds.
        if cross_val_idx is None:
            cross_val_idx = 0
        idxs = cv_split([self.perm], cv_idx=cross_val_idx,
                                     n_folds=n_cv_folds)
        self.train_idxs, self.val_idxs = idxs[0]

        self.train_X = DataObj(self.X, self.train_idxs)
        self.train_y = DataObj(self.y, self.train_idxs)
        self.val_X = DataObj(self.X, self.val_idxs)
        self.val_y = DataObj(self.y, self.val_idxs)
        self.train_shape = (len(self.train_idxs), *self.X.shape[1:])
        self.val_shape = (len(self.val_idxs), *self.X.shape[1:])
        if self.recurrent:
            self.n_loops = self.train_shape[0]
        else:
            self.n_loops = self.train_shape[0]//batch_size

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[self.perm[idx]]

    def shift_in_groups(self, arr, group_size=200):
        """
        Shifts the labels in groups by a random amount. i.e. if the
        group_size is 200, then the first 200 samples take on the
        value of the 200 samples shifted by some random amount. This
        method of grouping ensures any temporal correlations are
        broken.

        arr: ndarray (N,C)
            the data to be shifted. Only shifts along the first
            dimension N
        group_size: int
            the size of the chunks for shifting
        """
        shifted = np.empty(arr.shape, dtype=np.float32)
        for i in range(0,len(arr),group_size):
            shift = int(np.random.randint(len(arr)))
            arange = np.arange(i,min(i+group_size,len(arr)))
            idxs = (arange+shift)%len(arr)
            shifted[i:i+group_size] = arr[idxs]
        return shifted

    def order_into_batches(self, data, batch_size):
        """
        Rearranges the data into batches, preserving the order of the
        samples along the batch dimension.

        Ex:
            data = 1 2 3 4 5 6 7 8 9 10
            batch_size = 2
            return: 1 2 3 4 5
                    6 7 8 9 10

        data: ndarray or torch FloatTensor (N, ...)
        batch_size: int
            size of the batching
        """
        switch_back = False
        if type(data) != type(np.array([])):
            data = data.numpy()
            switch_back = True
        length = data.shape[0]//batch_size
        tot_len = length*batch_size
        trunc = data[:tot_len].reshape(batch_size, length,
                                          *data.shape[1:])
        trans_order = list(range(len(trunc.shape)))
        trans_order[0],trans_order[1]=trans_order[1],trans_order[0]
        trunc = trunc.transpose(trans_order)
        if switch_back:
            trunc = torch.from_numpy(trunc)
        return trunc

    def undo_batch_order(self, data):
        """
        Returns data that is shaped (B,L,...) in which the sequential
        order is preserved along L, into (N,...) in which the
        sequential order is preserved along N.

        data: ndarray (B,L, ...)
        """
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

    def val_sample(self, step_size):
        """
        Sample batches of data from the validation set.

        step_size: int
            the number of val samples to be returned

        Returns:
            val_X: ndarray or FloatTensor (S, ...)
            val_Y: ndarray or FloatTensor (S, C)
        """
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
        """
        Sample batches of data from the validation set.

        batch_size: int
            the number of train samples to be returned

        Returns:
            train_X: ndarray or FloatTensor (B, ...)
            train_Y: ndarray or FloatTensor (B, C)
        """
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
            if self.recurrent:
                idxs = batch_perm[i]
            else:
                idxs = batch_perm[i*batch_size:(i+1)*batch_size]

            yield self.train_X[idxs], self.train_y[idxs]
    
    def torch(self):
        """
        Converts all data to torch datatype
        """
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
        """
        Converts all data to numpy datatype
        """
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

class ChunkedData:
    """
    Used for LN cross validation in figures
    """
    def __init__(self, X, y, n_chunks=8, shuffle=False):
        self.shuffle = shuffle
        self.n_chunks = n_chunks
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
        
        idxs = torch.arange(len(X)).long() if not shuffle else \
                          torch.randperm(len(X)).long()
        
        self.chunks = []
        seg_size = len(X)//n_chunks
        for i in range(n_chunks-1):
            self.chunks.append(idxs[i*seg_size:(i+1)*seg_size])
        self.chunks.append(idxs[(n_chunks-1)*seg_size:])

    def get_mean(self, test_chunk):
        """
        returns mean along dimension 0. Created to reduce RAM
        footprint.

        test_chunk - idx of test chunk
        """
        n_samples = 0
        cumu_sum = 0
        for i in range(self.n_chunks):
            if i != test_chunk:
                temp_x = self.X[self.chunks[i]]
                cumu_sum = cumu_sum + temp_x.sum(0)
                n_samples += len(self.chunks[i])
        return cumu_sum/n_samples

    def get_std(self, test_chunk):
        """
        returns std along dimension 0. Created to reduce RAM
        footprint.

        test_chunk - idx of test chunk
        """
        mean = self.get_mean(test_chunk)
        cumu_sum = torch.zeros_like(mean)
        n_samples = 0
        for i in range(self.n_chunks):
            if i != test_chunk:
                temp_x = self.X[self.chunks[i]]
                cumu_sum = cumu_sum + ((temp_x-mean)**2).sum(0)
                n_samples += len(self.chunks[i])
        return torch.sqrt(cumu_sum / n_samples)

    def get_norm_stats(self, test_chunk):
        """
        Returns the mean and standard deviation of the test_chunk.
        Created to reduce RAM footprint.

        test_chunk: int
            the chunk of data to exclude in the normalization
            statistics
        """
        mean = self.get_mean(test_chunk)
        cumu_sum = torch.zeros_like(mean)
        n_samples = 0
        for i in range(self.n_chunks):
            if i != test_chunk:
                temp_x = self.X[self.chunks[i]]
                cumu_sum = cumu_sum + ((temp_x-mean)**2).sum(0)
                n_samples += len(self.chunks[i])
        return mean, torch.sqrt(cumu_sum / n_samples)

    def get_train_data(self,test_chunk):
        """
        Returns all chunks of data except for the argued test_chunk.

        test_chunk: int
            the chunk of data to exclude in the training data
        """
        X,y = [],[]
        for i in range(self.n_chunks):
            if i != test_chunk:
                X.append(self.X[self.chunks[i]])
                y.append(self.y[self.chunks[i]])
        if type(self.X) == type(np.array([])):
            return np.concatenate(X,axis=0), np.concatenate(y,axis=0)
        else:
            return torch.cat(X,dim=0), torch.cat(y,dim=0)
    
