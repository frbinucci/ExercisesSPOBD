import torch
from torch import nn


class CompressionNetwork(nn.Module):
    def __init__(self,hidden_filters,compressed_size):
        super(CompressionNetwork, self).__init__()
        self.conv1 = nn.Conv2d(kernel_size=3,in_channels=3,out_channels=hidden_filters,stride=2)
        self.conv2 = nn.Conv2d(kernel_size=3,in_channels=hidden_filters,out_channels=1,stride=2)
        self.linear1 = nn.Linear(in_features=49,out_features=compressed_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = torch.flatten(x,1)
        x = self.linear1(x)
        x = self.relu(x)
        return x

