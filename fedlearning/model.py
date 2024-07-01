import numpy as np

# Pytorch libraries
import torch
import torch.nn as nn
import torch.nn.functional as F

class NaiveMLP(nn.Module):
    def __init__(self, in_dims, in_channels, out_dims=10, dim_hidden=100):
        super(NaiveMLP, self).__init__()
        self.fc1 = nn.Linear(in_dims, dim_hidden)
        self.fc2 = nn.Linear(dim_hidden, out_dims)
        # self.fc3 = nn.Linear(dim_hidden, out_dims)

        # self.bn1 = nn.BatchNorm1d(dim_hidden, track_running_stats=False, affine=False)
        # self.bn2 = nn.BatchNorm1d(dim_hidden, track_running_stats=False, affine=False)

    def forward(self, x):
        x = x.view(x.shape[0], -1)

        activate = F.relu
        # activate = torch.erf
        # activate = torch.tanh

        out = activate(self.fc1(x))
        out = self.fc2(out)
        # out = self.fc3(out)

        return out

def init_weights(module, init_type='kaiming', gain=0.01):
    '''
    initialize network's weights
    init_type: normal | uniform | kaiming
    '''
    classname = module.__class__.__name__
    if hasattr(module, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
        if init_type == 'normal':
            nn.init.normal_(module.weight.data, 0.0, gain)
        elif init_type == "uniform":
            nn.init.uniform_(module.weight.data)
        elif init_type == 'kaiming':
            nn.init.kaiming_normal_(module.weight.data, a=0, mode='fan_in')

        if hasattr(module, 'bias') and module.bias is not None:
            nn.init.constant_(module.bias.data, 0.0)

    elif (classname.find('BatchNorm') != -1 and module.weight is not None):
        nn.init.normal_(module.weight.data, 1.0, gain)
        nn.init.constant_(module.bias.data, 0.0)

    elif (classname.find("GroupNorm") != -1 and module.weight is not None):
        nn.init.normal_(module.weight.data, 1.0, gain)
        nn.init.constant_(module.bias.data, 0.0)

class NaiveCNN(nn.Module):
    def __init__(self, in_dims, in_channels, out_dims=10):
        self.in_channels = in_channels
        self.in_dims = in_dims
        self.out_dims = out_dims
        super(NaiveCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, stride=1, padding="same")
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding="same")
        # For example, for a 28x28 image, the input to the fc1 will be 64*7*7
        print(in_dims)
        self.fc1 = nn.Linear(64*in_dims[0]*in_dims[1]//16, 128)
        self.fc2 = nn.Linear(128, out_dims)
    
    def forward(self, x):
        x = x.view(x.size(0), self.in_channels, self.in_dims[0], self.in_dims[1])
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, stride=2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, stride=2)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x