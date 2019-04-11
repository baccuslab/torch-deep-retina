import torch
import torch.nn as nn

class Physio:
    def __init__(self, net):
        self.net = net
        self.dict = {}
        self.inspect_hooks = False
        self.hooks = []
    
    def layer_activity(self, name):
        def hook(module, inp, out):
            self.dict[name] = out.cpu().detach().numpy()
        return hook
      
    def layer_grad(self, name):
        def hook(module, inp, out):
            self.dict[name+'_grad'] = out[0].cpu().detach().numpy()
        return hook
    
    def injection(self, subtype, constant):
        def hook(module, inp, out):
            module.weight[subtype, :, :] = module.weight[subtype, :, :] * constant
            module.bias[subtype] = module.bias[subtype] * constant
    
    # activity_dict = phys.inspect(stim)
    # activity_dict['conv1'] <--- gets the conv2d_1 layer activity
    def inspect(self, stim, insp_keys={"all"}):
        if len(self.hooks) <= 0:
            for name, module in self.net.named_modules():
                if name in insp_keys or "all" in insp_keys:
                    self.hooks.append(module.register_forward_hook(self.layer_activity(name)))
                    self.hooks.append(module.register_backward_hook(self.layer_grad(name)))
        self.dict['output'] = self.net(stim)
        return self.dict
    
    # phys.inject('conv1', 1, 2)
    # then do a forward pass
    def inject(self, layer, subtype, constant):
        for name, module in self.net.named_modules():
            if name == layer:
                module.register_forward_hook(self.injection(subtype, constant))

    def remove_hooks(self):
        for h in self.hooks:
            h.remove()
        self.hooks = []

    def remove_refs(self):
        for k,v in list(self.dict.items()):
            try:
                self.dict[k].detach().cpu()
                del self.dict[k]
            except:
                pass














