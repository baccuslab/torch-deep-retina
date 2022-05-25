import os
from yacs.config import CfgNode

_C = CfgNode()
_C.save_path = ''
_C.img_shape = [40,50,50]

_C.Model = CfgNode()
_C.Model.name = 'BN_CNN_Stack'
_C.Model.t_list = []
_C.Model.checkpoint = ''
_C.Model.binomial_para = []
_C.Model.intv = 0.5
_C.Model.thre = 1.
_C.Model.curve_thre = 0.4
_C.Model.sigma = True
_C.Model.n_units = 6
_C.Model.bias = True
_C.Model.linear_bias = True
_C.Model.chans = [8,8]
_C.Model.bn_moment = 0.01
_C.Model.ksizes = (15,11)
_C.Model.noise_locs = [3,4,2]

_C.Data = CfgNode()
_C.Data.data_path = '/home/xhding/tem_stim'
_C.Data.repeats_path = ''
_C.Data.date = '21-03-15'
_C.Data.stim = 'naturalscene'
_C.Data.cells = []
_C.Data.batch_size = 500
_C.Data.num_trials = 15

_C.Eval = CfgNode()
_C.Eval.range = [0., 3., 0., 1., 0., 1.]
_C.Eval.num_points = [11, 11, 11]
_C.Eval.weight = 0.
_C.Eval.ignore_idxs = []
_C.Eval.num_repeats = 100
