import torch
from torch import nn

class Net(nn.Module):
    def __init__(self,**kwargs):
        super(Net, self).__init__()
        first_layer_size = kwargs.get('first_layer_size',784)
        self.l0 = nn.Linear(784,first_layer_size,bias=True)
        self.l1 = nn.Linear(first_layer_size,128)
        self.l2 = nn.Linear(128,64)
        self.l3 = nn.Linear(64,16)
        self.l4 = nn.Linear(16,10)
        self.relu = nn.ReLU()
        self.intermediate_output = False
        self.test_bottleneck = False

    def set_intermediate_output(self,intermediate_output):
        self.intermediate_output = intermediate_output

    def set_test_bottleneck(self,test_bottleneck):
        self.test_bottleneck = test_bottleneck

    def forward(self, x):
        x = torch.flatten(x,1)
        if self.test_bottleneck == False:
            x = self.l0(x)
        if self.intermediate_output == False:
            x = self.relu(x)
            x = self.relu(self.l1(x))
            x = self.relu(self.l2(x))
            x = self.relu(self.l3(x))
            x = self.l4(x)
        return x