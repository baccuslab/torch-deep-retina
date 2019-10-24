import os
from yacs.config import CfgNode

_C = CfgNode()

_C.exp_id = ''
_C.img_shape = (40,50,50)
_C.gpu = 0
_C.epoch = 50
_C.save_intrvl = 5
_C.save_path = ''

_C.Model = CfgNode()
_C.Model.checkpoint = ''
_C.Model.drop_p = 0.
_C.Model.scale_kinet = True
_C.Model.recur_seq_len = 5
_C.Model.n_units = 5
_C.Model.noise = 0.05
_C.Model.bias = True
_C.Model.linear_bias = None
_C.Model.chans = [8,8]
_C.Model.bn_moment = 0.01
_C.Model.softplus = True
_C.Model.ksizes = (15,11)

_C.Data = CfgNode()
_C.Data.data_path = '/home/salamander/experiments/data/'
_C.Data.batch_size = 512

_C.Optimize = CfgNode()
_C.Optimize.trunc_intvl = 8
_C.Optimize.lr = 1e-3
_C.Optimize.l2 = 1e-4
