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
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1, dilation=1, groups=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(num_features=64, affine=False),
            nn.Conv2d(64, 64, 3, 1, 1, 1, 1, False),
            nn.ReLU(),
            nn.BatchNorm2d(num_features=64, affine=False),
            nn.Conv2d(64, 64, 3, 1, 1, 1, 1, False),
            nn.ReLU(),
            nn.BatchNorm2d(num_features=64, affine=False)
        )

        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2, dilation=1)

        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, 1, 1, 1, 1, False),
            nn.ReLU(),
            nn.BatchNorm2d(num_features=128, affine=False),
            nn.Conv2d(128, 128, 3, 1, 1, 1, 1, False),
            nn.ReLU(),
            nn.BatchNorm2d(num_features=128, affine=False),
            nn.Conv2d(128, 128, 3, 1, 1, 1, 1, False),
            nn.ReLU(),
            nn.BatchNorm2d(num_features=128, affine=False)
        )

        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2, dilation=1)

        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 256, 3, 1, 1, 1, 1, False),
            nn.ReLU(),
            nn.BatchNorm2d(num_features=256, affine=False),
            nn.Conv2d(256, 256, 3, 1, 1, 1, 1, False),
            nn.ReLU(),
            nn.BatchNorm2d(num_features=256, affine=False),
            nn.Conv2d(256, 256, 3, 1, 1, 1, 1, False),
            nn.ReLU(),
            nn.BatchNorm2d(num_features=256, affine=False)
        )

        self.conv4 = nn.Sequential(
            nn.Conv2d(256, 512, 3, 1, 1, 1, 1, False),
            nn.ReLU(),
            nn.BatchNorm2d(num_features=512, affine=False),
            nn.Conv2d(512, 512, 3, 1, 1, 1, 1, False),
            nn.ReLU(),
            nn.BatchNorm2d(num_features=512, affine=False),
            nn.Conv2d(512, 512, 3, 1, 1, 1, 1, False),
            nn.ReLU(),
            nn.BatchNorm2d(num_features=512, affine=False)
        )

        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2, dilation=1)

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

class Vgg19(nn.Module):
    def __init__(self):
        super(Vgg19, self).__init__()
        self.conv1 = nn.Sequential(
                        nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
                        nn.ReLU(),
                        nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
                        nn.ReLU(),
                        nn.BatchNorm2d(64),
                        nn.MaxPool2d(kernel_size=2, stride=2),
                        )
        self.conv2 = nn.Sequential(
                        nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
                        nn.ReLU(),
                        nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
                        nn.ReLU(),
                        nn.BatchNorm2d(128),
                        )
        self.conv3 = nn.Sequential(
                        nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
                        nn.ReLU(),
                        nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
                        nn.ReLU(),
                        nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
                        nn.ReLU(),
                        nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
                        nn.ReLU(),
                        nn.BatchNorm2d(256),
                        nn.MaxPool2d(kernel_size=2, stride=2),
                        )
        self.conv4 = nn.Sequential(
                        nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
                        nn.ReLU(),
                        nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
                        nn.ReLU(),
                        nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
                        nn.ReLU(),
                        nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
                        nn.ReLU(),
                        nn.BatchNorm2d(512),
                        )
        self.conv5 = nn.Sequential(
                        nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
                        nn.ReLU(),
                        nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
                        nn.ReLU(),
                        nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
                        nn.ReLU(),
                        nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
                        nn.ReLU(),
                        nn.BatchNorm2d(512),
                        nn.MaxPool2d(kernel_size=2, stride=2),
                        )

        self.flatten = lambda x: x.view(x.size(0), -1)

        self.fc = nn.Sequential(
                        nn.Linear(8*8*512, 4096),
                        nn.ReLU(),
                        nn.Linear(4096, 2048),
                        nn.ReLU(),
                        nn.Linear(2048, 2048),
                        nn.ReLU(),
                        nn.Linear(2048, 1024),
                        nn.ReLU(),
                        nn.Linear(1024, 200),
                        nn.Softmax(),
                        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x