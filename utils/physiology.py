import torch
import torch.nn as nn
#Use: phys = Physio(myNet)
#	  

class Physio:
	def __init__(self, net):
		self.net = net
		self.dict = {}
		self.inspect_hooks = False

	def layer_activity(self, name):
		def hook(module, inp, out):
			self.dict[name] = out.cpu().detach().numpy()
		return hook

	def injection(self, subtype, constant):
		def hook(module, inp, out):
			module.weight[subtype, :, :] = module.weight[subtype, :, :] * constant
			module.bias[subtype] = module.bias[subtype] * constant

	# activity_dict = phys.inspect(stim)
	# activity_dict['conv1'] <--- gets the conv2d_1 layer activity
	def inspect(self, stim):
		if(not self.inspect_hooks):
			for name, module in self.net.named_modules():
				module.register_forward_hook(self.layer_activity(name))
			self.inspect_hooks = True
		self.dict['output'] = self.net(stim).cpu().detach().numpy()
		return self.dict

	# phys.inject('conv1', 1, 2)
	# then do a forward pass
	def inject(self, layer, subtype, constant):
		for name, module in self.net.named_modules():
			if name == layer:
				module.register_forward_hook(self.injection(subtype, constant))













