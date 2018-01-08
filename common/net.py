'''
@file: net.py
@version: v1.0
@date: 2018-01-07
@author: ruanxiaoyi
@brief: Design the network
@remark: {when} {email} {do what}
'''

import torch.nn as nn

class VGGNet(nn.Module):
    def __init__(self):
        super(VGGNet, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1, 0, 1, False),
            nn.ReLU(),
            nn.BatchNorm2d(num_features=64, affine=False),
            nn.Conv2d(64, 64, 3, 1, 1, 0, 1, False),
            nn.ReLU(),
            nn.BatchNorm2d(num_features=64, affine=False),
            nn.Conv2d(64, 64, 3, 1, 1, 0, 1, False),
            nn.ReLU(),
            nn.BatchNorm2d(num_features=64, affine=False)
        )

        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2, dilation=0)

        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, 1, 1, 0, 1, False),
            nn.ReLU(),
            nn.BatchNorm2d(num_features=128, affine=False),
            nn.Conv2d(128, 128, 3, 1, 1, 0, 1, False),
            nn.ReLU(),
            nn.BatchNorm2d(num_features=128, affine=False),
            nn.Conv2d(128, 128, 3, 1, 1, 0, 1, False),
            nn.ReLU(),
            nn.BatchNorm2d(num_features=128, affine=False)
        )

        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2, dilation=0)

        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 256, 3, 1, 1, 0, 1, False),
            nn.ReLU(),
            nn.BatchNorm2d(num_features=256, affine=False),
            nn.Conv2d(256, 256, 3, 1, 1, 0, 1, False),
            nn.ReLU(),
            nn.BatchNorm2d(num_features=256, affine=False),
            nn.Conv2d(256, 256, 3, 1, 1, 0, 1, False),
            nn.ReLU(),
            nn.BatchNorm2d(num_features=256, affine=False)
        )

        self.conv4 = nn.Sequential(
            nn.Conv2d(256, 512, 3, 1, 1, 0, 1, False),
            nn.ReLU(),
            nn.BatchNorm2d(num_features=512, affine=False),
            nn.Conv2d(512, 512, 3, 1, 1, 0, 1, False),
            nn.ReLU(),
            nn.BatchNorm2d(num_features=512, affine=False),
            nn.Conv2d(512, 512, 3, 1, 1, 0, 1, False),
            nn.ReLU(),
            nn.BatchNorm2d(num_features=512, affine=False)
        )

        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2, dilation=0)

        self.flatten = lambda x: x.view(x.size(0), -1)

        self.fcn = nn.Sequential(
            nn.Linear(8 * 8 * 512, 4096),
            nn.Dropout(),
            nn.Linear(4096, 2048),
            nn.Dropout(),
            nn.Linear(2048, 200),
            nn.Softmax()
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.pool3(x)
        x = self.flatten(x)
        x = self.fcn(x)
        return x
