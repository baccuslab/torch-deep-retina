import torch
import torch.nn as nn
#Use: phys = Physio(myNet)
#	  activity_dict = phys.inspect(stim)
#     activity_dict['conv2d_1'] <--- gets the conv2d_1 layer activity

class Physio:
	def __init__(self, net):
		self.net = net
		self.dict = {};

	def layer_activity(self, name):
		def hook(module, inp, out):
			self.dict[name] = out.cpu().detach().numpy()
		return hook

	def inspect(self, stim):
		for name, module in self.net.named_modules():
			module.register_forward_hook(self.layer_activity(name))
		self.dict['output'] = self.net(stim)
		return self.dict









