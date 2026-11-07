
'''
CNN for CIFAR
'''

import torch
from torch import nn

from local_code.stage_3_code.Method_CNN import Method_CNN


class Method_CNN_CIFAR(Method_CNN):
    data = None

    def __init__(self, mName, mDescription, device='auto'):
        Method_CNN.__init__(self, mName, mDescription, device)

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1) 
        
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1) 

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)

        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)

        self.bn3 = nn.BatchNorm2d(128)


        self.pool = nn.MaxPool2d(2) 
        self.relu = nn.ReLU()

        self.fc1 = nn.Linear(2048, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 10)

        self.drop = nn.Dropout(0.2)

    def forward(self, x):
        x = self.pool(self.relu(self.bn1(self.conv1(x)))) # 32 x 32 x 32 -> 16 x 16 x 32
        x = self.pool(self.relu(self.bn2(self.conv2(x)))) # 16 x 16 x 64 -> 8 x 8 x 64
        x = self.pool(self.relu(self.bn3(self.conv3(x)))) # 8 x 8 x 128 -> 4 x 4 x 128

        x = torch.flatten(x, 1) # 4 x 4 x 128 = 2048

        x = self.drop(self.relu(self.fc1(x)))
        x = self.drop(self.relu(self.fc2(x)))
        x = self.fc3(x)

        return x