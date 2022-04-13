"""Two-layer CNN for MNIST"""
import torch
from torch import nn
from torch.nn import functional as F


class BabyCNN(nn.Module):
    def __init__(self, cfg):
        super(BabyCNN, self).__init__()
        self.cfg = cfg
        self.conv1 = nn.Conv2d(self.cfg.in_channels, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(self.cfg.dropout1)
        self.dropout2 = nn.Dropout(self.cfg.dropout2)
        self.fc1 = nn.Linear(9216, self.cfg.fc_units)
        self.fc2 = nn.Linear(self.cfg.fc_units, self.cfg.out_features)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.softmax(x, dim=1)
        return output
