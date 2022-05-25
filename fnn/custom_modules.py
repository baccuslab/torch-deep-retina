import torch
import torch.nn as nn

class OneHot(nn.Module):
    def __init__(self,shape):
        super(OneHot, self).__init__()
        self.shape = shape
        self.w = nn.Parameter(torch.rand(shape[0],shape[1]*shape[2]))
        self.prob = None


    def forward(self, x):
        positive = self.w - torch.min(self.w,1)[0][:,None]
        normed = positive.permute(1,0) / positive.sum(-1)
        self.prob = normed.permute(1,0)

        x = x.reshape(*x.shape[:2],-1)
        out = torch.sum(x*self.prob,dim=-1)

        return out

def semantic_loss(prob):
    wmc_tmp = torch.zeros_like(prob)

    for i in range(prob.shape[1]):
        one_situation = torch.ones_like(prob).scatter_(1,torch.zeros_like(prob[:,0]).fill_(i).unsqueeze(-1).long(),0)
        wmc_tmp[:,i] = torch.abs((one_situation - prob).prod(dim=1))

    wmc_tmp = -1.0*torch.log(wmc_tmp.sum(dim=1))
    total_loss = torch.sum(wmc_tmp)
    return total_loss