import os
from yacs.config import CfgNode

_C = CfgNode()

_C.exp_id = 'random'
_C.img_shape = [40,50,50]
_C.epoch = 50
_C.save_intvl = 5
_C.save_path = '/home/xhding/saved_model'

_C.Model = CfgNode()
_C.Model.name = 'BN_CNN_Stack'
_C.Model.checkpoint = ''
_C.Model.n_units = 5
_C.Model.bias = True
_C.Model.linear_bias = False
_C.Model.chans = [8,8]
_C.Model.bn_moment = 0.01
_C.Model.ksizes = (15,11)

_C.Data = CfgNode()
_C.Data.data_path = '/home/TRAIN_DATA'
_C.Data.date = '15-10-07'
_C.Data.stim = 'naturalscene'
_C.Data.batch_size = 512
_C.Data.val_size = 30000

_C.Optimize = CfgNode()
_C.Optimize.trunc_intvl = 1
_C.Optimize.lr = 1e-3
_C.Optimize.l2 = 1e-4
_C.Optimize.l1 = 0.
_C.Optimize.oh_loss_scale = 7.
