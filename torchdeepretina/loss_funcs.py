import torch
import torch.nn as nn

class MinMax:
    def __init__(self):
        pass

    def __call__(self, x1, x2):
        """
        Minimizes value of x1 while maximizing value of x2
        """
        loss = x1.mean() - x2.mean()
        return loss
