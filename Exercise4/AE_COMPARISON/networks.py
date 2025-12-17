import torch
from torch import nn
import torch.nn.functional as F

class GIBNetwork(nn.Module):
    def __init__(self):
        super(GIBNetwork, self).__init__()
        self.intermediate_output = False
        self.test_bottleneck = False
        #Linear stage
        self.conv1 = nn.Conv2d(3, 16, kernel_size=(3, 3), padding=1,stride=1)
        self.conv2 = nn.Conv2d(16,1,kernel_size=(3,3),padding=1,stride=2)
        self.bn1 = nn.BatchNorm2d(16)
        self.bn2 = nn.BatchNorm2d(1)
        self.max_pool = nn.MaxPool2d(2)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.25)
        self.fc = nn.Linear(256, 43)

    def set_intermediate_output(self,intermediate_output):
        self.intermediate_output = intermediate_output

    def set_test_bottleneck(self,test_bottleneck):
        self.test_bottleneck = test_bottleneck

    def forward(self, x):
        if self.test_bottleneck == False:
            x = self.conv2(self.bn1(self.conv1(x)))
            x = self.bn2(x)
        if self.intermediate_output == False:
            x = self.relu(x)
            x = x.view(x.size(0), -1)
            x = self.fc(x)
        return x

class AENetwork(nn.Module):
    def __init__(self,compressed_size):
        super(AENetwork, self).__init__()
        self.intermediate_output = False
        self.test_bottleneck = False
        #Linear stage
        self.junction_stage= nn.Linear(compressed_size,16**2)
        self.reshape = nn.Unflatten(1, (1, 16, 16))
        self.relu = nn.ReLU()


        self.conv2 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )

        self.fc = nn.Linear(2048, 43)

    def set_intermediate_output(self,intermediate_output):
        self.intermediate_output = intermediate_output

    def set_test_bottleneck(self,test_bottleneck):
        self.test_bottleneck = test_bottleneck

    def forward(self, x):
        x = self.junction_stage(x)
        x = self.reshape(x)
        #x = self.relu(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class ClassificationNetwork(nn.Module):
    def __init__(self,in_features,mid_features,out_features):
        super(ClassificationNetwork, self).__init__()
        self.linear1 = nn.Linear(in_features=in_features,out_features=mid_features)
        self.linear2 = nn.Linear(in_features=mid_features,out_features=out_features)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.linear1(x))
        x = self.linear2(x)
        return x




