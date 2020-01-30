import torch
import torch.nn as nn

class CNN(nn.Module):
    def __init__(self, out):
        super(CNN, self).__init__()
        self.name = 'CNN'
        self.conv1 = nn.Conv2d(60, 6, kernel_size=8, bias=True)
        self.relu1 = nn.Softplus()
        self.conv2 = nn.Conv2d(6, 4, kernel_size=11, bias=True)
        self.relu2 = nn.Softplus()
        self.conv3 = nn.Conv2d(4, out, kernel_size=15, bias=True)
        self.relu3 = nn.Softplus()

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.conv3(x)
        x = self.relu3(x)
        return x

class LN(nn.Module):
    def __init__(self, out):
        super(LN, self).__init__()
        self.name = 'LN'
        self.conv = nn.Conv2d(60, out, kernel_size=10, bias=False)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        return x
