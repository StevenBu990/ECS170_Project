
'''
CNN for ORL
'''

import torch
from torch import nn

from local_code.stage_3_code.Method_CNN import Method_CNN


class Method_CNN_ORL(Method_CNN):
    data = None

    def __init__(self, mName, mDescription, device='auto'):
        Method_CNN.__init__(self, mName, mDescription, device)

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1) 
        
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1) 

        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1)

        self.pool = nn.MaxPool2d(2) 
        self.relu = nn.ReLU()

        self.fc1 = nn.Linear(9856, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 40)

        self.drop = nn.Dropout(0.2)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x))) # 112 x 92 x 32 -> 56 x 46 x 32
        x = self.pool(self.relu(self.conv2(x))) # 56 x 46 x 64 -> 28 x 23 x 64
        x = self.pool(self.relu(self.conv3(x))) # 28 x 23 x 64 -> 14 x 11 x 64

        x = torch.flatten(x, 1) # 14 x 11 x 64 = 9856

        x = self.drop(self.relu(self.fc1(x)))
        x = self.drop(self.relu(self.fc2(x)))
        x = self.fc3(x)

        return x