'''
@file: net.py
@version: v1.0
@date: 2018-01-07
@author: ruanxiaoyi
@brief: Design the network
@remark: {when} {email} {do what}
'''

import torch.nn as nn
import torch

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


CONF = {
    # 'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'], origin
    'A': [64, 128, 'M', 256, 256, 512, 512, 'M', 512, 512, 'M'],
    # 'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 256, 256, 'M', 512, 512, 512, 512, 'M'],
}

class Vgg11(nn.Module):
    def __init__(self):
        super(Vgg11, self).__init__()
        self.features = self._make_layers(CONF['A'], True)
        self.fc = nn.Sequential(
            nn.Linear(512*8*8, 4096),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(4096, 200)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

    def _make_layers(self, conf, batch_norm=False):
        layers = []
        in_channels = 3
        for v in conf:
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
                if batch_norm:
                    layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
                else:
                    layers += [conv2d, nn.ReLU(inplace=True)]
                in_channels = v
        return nn.Sequential(*layers)

class GoogleNet(nn.Module):
    def __init__(self):
        super(GoogleNet, self).__init__()
        self.pre_layer = nn.Sequential(
                            nn.Conv2d(3, 192, kernel_size=3, padding=1),
                            nn.BatchNorm2d(192),
                            nn.ReLU(inplace=True),
                            )
        self.inception = nn.Sequential(
                            Inception(192, 64, 96, 128, 16, 32, 32),
                            Inception(256, 128, 128, 192, 32, 96, 64),
                            nn.MaxPool2d(3, stride=2, padding=1),    #shape 32x32x480
                            Inception(480, 192, 96, 208, 16, 48, 64),
                            Inception(512, 160, 112, 224, 24, 64, 64),
                            # nn.MaxPool2d(3, stride=2, padding=1),   #shape 16x16x512
                            # Inception(512, 128, 128, 256, 24,  64,  64),
                            # Inception(512, 112, 144, 288, 32,  64,  64),
                            nn.AvgPool2d(3, stride=2),    #shape 15x15x512
                            )
        self.fc = nn.Sequential(
                        nn.Linear(15*15*512, 4096),
                        nn.ReLU(),
                        nn.Dropout(),
                        nn.Linear(4096, 4096),
                        nn.ReLU(),
                        nn.Dropout(),
                        nn.Linear(4096, 200),
                        )
    def forward(self, x):
        x = self.pre_layer(x)
        x = self.inception(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


        

class Inception(nn.Module):
    def __init__(self, in_chanle, n1x1, n3x3b, n3x3, n5x5b, n5x5, pool_chanel):
        super(Inception, self).__init__()
        # 1x1 conv branch
        self.branch1 = nn.Sequential(
                            nn.Conv2d(in_chanle, n1x1, kernel_size=1),
                            nn.BatchNorm2d(n1x1),
                            nn.ReLU(inplace=True),
                            )
        # 1x1 conv -> 3x3 conv branch
        self.branch2 = nn.Sequential(
                            nn.Conv2d(in_chanle, n3x3b, kernel_size=1),
                            nn.BatchNorm2d(n3x3b),
                            nn.ReLU(inplace=True),
                            nn.Conv2d(n3x3b, n3x3, kernel_size=3, padding=1),
                            nn.BatchNorm2d(n3x3),
                            nn.ReLU(inplace=True),
                            )
        # 1x1 conv -> 5x5 conv branch
        self.branch3 = nn.Sequential(
                            nn.Conv2d(in_chanle, n5x5b, kernel_size=1),
                            nn.BatchNorm2d(n5x5b),
                            nn.ReLU(inplace=True),
                            nn.Conv2d(n5x5b, n5x5, kernel_size=3, padding=1),
                            nn.BatchNorm2d(n5x5),
                            nn.ReLU(inplace=True),
                            )
        # 3x3 pool -> 1x1 conv branch 
        self.branch4 = nn.Sequential(
                            nn.MaxPool2d(3, stride=1, padding=1),
                            nn.Conv2d(in_chanle, pool_chanel, kernel_size=1),
                            nn.BatchNorm2d(pool_chanel),
                            nn.ReLU(inplace=True),
                            )

    def forward(self, x):
        y1 = self.branch1(x)
        y2 = self.branch2(x)
        y3 = self.branch3(x)
        y4 = self.branch4(x)
        return torch.cat([y1, y2, y3, y4], 1)
