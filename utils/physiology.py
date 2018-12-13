import torch
import torch.nn as nn
#Use: phys = Physio(myNet)
#	  activity_dict = phys.inspect(stim)
#     activity_dict.['conv2d_1'] <--- gets the conv2d_1 layer activity

class Physio:
	def __init__(self, net):
		self.net = net
		self.dict = {};

	def layer_activity(self, module, inp, out):
		self.dict[module.__class__.__name__] = out

	def inspect(self, stim):
		for module in self.net.modules():
			module.register_forward_hook(layer_activity)
		self.dict['output'] = self.net(stim)
		return self.dict








