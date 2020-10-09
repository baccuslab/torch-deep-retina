import os
from yacs.config import CfgNode

_C = CfgNode()

_C.exp_id = 'random'
_C.img_shape = (40,50,50)
_C.epoch = 50
_C.save_intvl = 5
_C.save_path = '/home/xhding/saved_model'

_C.Model = CfgNode()
_C.Model.img_shape = _C.img_shape
_C.Model.name = 'KineticsModel'
_C.Model.checkpoint = ''
_C.Model.scale_kinet = False
_C.Model.recur_seq_len = 8
_C.Model.n_units = 5
_C.Model.bias = True
_C.Model.linear_bias = False
_C.Model.chans = [8,8]
_C.Model.bn_moment = 0.01
_C.Model.softplus = True
_C.Model.ksizes = (15,11)
_C.Model.k_chan = True
_C.Model.ka_offset = False
_C.Model.ksr_gain = False
_C.Model.k_inits = {'ka':23., 'ka_2':None, 'kfi':50., 'kfr':87., 'ksi':0., 'ksr':0., 'ksr_2':0.}
_C.Model.dt = 0.01
_C.Model.scale_shift_chan = True

_C.Data = CfgNode()
_C.Data.img_shape = _C.img_shape
_C.Data.data_path = '/home/TRAIN_DATA'
_C.Data.date = '15-10-07'
_C.Data.stim = 'naturalscene'
_C.Data.batch_size = 512
_C.Data.val_size = 30000
_C.Data.trunc_int = 8
_C.Data.loss_bin = 1000
_C.Data.cells = 'all'
_C.Data.stim_type = 'full'
_C.Data.hs_mode = 'single'
_C.Data.I20 = None
_C.Data.start_idx = 0

_C.Optimize = CfgNode()
_C.Optimize.loss_fn = 'poisson'
_C.Optimize.lr = 1e-3
_C.Optimize.l2 = 1e-4