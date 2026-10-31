
'''
CNN for MNIST: two conv blocks (conv -> ReLU -> max pool), then two fully connected layers.
'''

from local_code.stage_3_code.Method_CNN import Method_CNN
import torch
from torch import nn

class Method_CNN_MNIST(Method_CNN):
    data = None

    def __init__(self, mName, mDescription, device='auto'):
        Method_CNN.__init__(self, mName, mDescription, device)

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        # conv2: 32 -> 64 channels, kernel 3, padding 1
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2)
        self.relu = nn.ReLU()
        # fc1: flattened size -> 128
        self.fc1 = nn.Linear(3136, 128)
        # fc2: 128 -> 10
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)

        return x