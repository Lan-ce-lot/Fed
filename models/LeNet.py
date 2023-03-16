from collections import OrderedDict

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.utils.data as Data
import torchvision
import torch.nn.functional as func

class LeNet(nn.Module):
    def __init__(self, in_features=1, num_classes=10, dim=1024):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_features,
                        32,
                        kernel_size=5,
                        padding=0,
                        stride=1,
                        bias=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 2))
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32,
                        64,
                        kernel_size=5,
                        padding=0,
                        stride=1,
                        bias=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 2))
        )
        self.fc1 = nn.Sequential(
            nn.Linear(dim, 512),
            nn.ReLU(inplace=True)
        )
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = torch.flatten(out, 1)
        out = self.fc1(out)
        out = self.fc(out)
        return out


class DigitModel(nn.Module):
    """
    Model for benchmark experiment on Digits.
    """

    def __init__(self, num_classes=10, dim=6272, **kwargs):
        super(DigitModel, self).__init__()
        self.conv1 = nn.Sequential(
            OrderedDict([
                ('conv1', nn.Conv2d(3, 64, 5, 1, 2)),
                ('bn1', nn.BatchNorm2d(64)),
                ('relu1', nn.ReLU(inplace=True)),
                ('maxpool1',nn.MaxPool2d(kernel_size=2))
            ])
        )
        self.conv2 = nn.Sequential(
            OrderedDict([
                ('conv2', nn.Conv2d(64, 64, 5, 1, 2)),
                ('bn2', nn.BatchNorm2d(64)),
                ('relu2', nn.ReLU(inplace=True)),
                ('maxpool2', nn.MaxPool2d(kernel_size=2))
            ])
        )
        self.conv3 = nn.Sequential(
            OrderedDict([
                ('conv3', nn.Conv2d(64, 128, 5, 1, 2)),
                ('bn3', nn.BatchNorm2d(128)),
                ('relu3', nn.ReLU(inplace=True)),
            ])
        )
        self.fc1 = nn.Sequential(
            OrderedDict([
                ('fc1', nn.Linear(dim, 2048)),
                ('bn4', nn.BatchNorm1d(2048)),
                ('relu4', nn.ReLU(inplace=True)),

                ('fc2', nn.Linear(2048, 512)),
                ('bn5', nn.BatchNorm1d(512)),
                ('relu5', nn.ReLU(inplace=True)),
            ])
        )
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = out.view(out.shape[0], -1)
        out = self.fc1(out)
        out = self.fc(out)

        return out


