from torch import nn
import torch.nn.functional as F


class ResNet18(nn.Module):
    def __init__(self, num_classes=10):
        super(ResNet18, self).__init__()
        self.pre = nn.Sequential(
            nn.Conv2d(1, 64, 3, 1, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2, 1)
        )
        # make layer for multiple residual block
        self.layer1 = _make_layer(64, 64, 2)
        self.layer2 = _make_layer(64, 128, 2, stride=2)
        self.layer3 = _make_layer(128, 256, 2, stride=2)
        self.layer4 = _make_layer(256, 512, 2, stride=2)

        # fc layer
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        x = x.view(-1, 1, 28, 28)
        x = self.pre(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = F.avg_pool2d(x, 2)
        x = x.view(x.size(0), -1)
        return self.fc(x)

    def __str__(self):
        return 'ResNet18'


class ResNet34(nn.Module):
    def __init__(self, num_classes=10):
        super(ResNet34, self).__init__()
        self.pre = nn.Sequential(
            nn.Conv2d(1, 64, 3, 1, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2, 1)
        )
        # make layer for multiple residual block
        self.layer1 = _make_layer(64, 128, 3)
        self.layer2 = _make_layer(128, 256, 4, stride=2)
        self.layer3 = _make_layer(256, 512, 6, stride=2)
        self.layer4 = _make_layer(512, 512, 3, stride=2)

        # fc layer
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        x = x.view(-1, 1, 28, 28)
        x = self.pre(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = F.avg_pool2d(x, 2)
        x = x.view(x.size(0), -1)
        return self.fc(x)

    def __str__(self):
        return 'ResNet34'


class ResidualBlock(nn.Module):
    def __init__(self, inchannel, outchannel, stride=1, shortcut=None):
        super(ResidualBlock, self).__init__()
        self.left = nn.Sequential(
            nn.Conv2d(inchannel, outchannel, 3, stride, 1, bias=False),
            nn.BatchNorm2d(outchannel),
            nn.ReLU(inplace=True),
            nn.Conv2d(outchannel, outchannel, 3, 1, 1, bias=False),
            nn.BatchNorm2d(outchannel)
        )
        self.right = shortcut

    def forward(self, x):
        out = self.left(x)
        residual = x if self.right is None else self.right(x)
        out += residual
        return F.relu(out)


def _make_layer(inchannel, outchannel, block_num, stride=1):
    shortcut = nn.Sequential(
        nn.Conv2d(inchannel, outchannel, 1, stride, bias=False),
        nn.BatchNorm2d(outchannel)
    )

    layers = [ResidualBlock(inchannel, outchannel, stride, shortcut)]

    for i in range(1, block_num):
        layers.append(ResidualBlock(outchannel, outchannel))
    return nn.Sequential(*layers)
