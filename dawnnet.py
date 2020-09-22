import torch
import torch.nn as nn
import torch.nn.functional as F


class Block(nn.Module):

    def __init__(self, c_in, c_out, stride=1):
        super(Block, self).__init__()
        self.bn1 = nn.BatchNorm2d(c_in)
        self.conv1 = nn.Conv2d(c_in, c_out, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(c_out)
        self.conv2 = nn.Conv2d(c_out, c_out, kernel_size=3, stride=1, padding=1, bias=False)

        self.proj = (c_in != c_out or stride > 1)
        if self.proj:
            self.conv_proj = nn.Conv2d(c_in, c_out, kernel_size=1, stride=stride, padding=0, bias=False)

    def forward(self, x):
        x = F.relu(self.bn1(x))

        identity = x.clone().to(x.device)

        x = F.relu(self.bn2(self.conv1(x)))

        if self.proj:
            identity = self.conv_proj(identity)

        x += identity

        return x


class DawnNet(nn.Module):

    def __init__(self, in_channels=3, out_channels=10, c=64):
        super(DawnNet, self).__init__()

        c = [c, 2*c, 4*c, 4*c]

        self.conv0 = nn.Conv2d(in_channels, c[0], kernel_size=3, stride=1, padding=1, bias=False)
        self.bn0 = nn.BatchNorm2d(c[0])
        self.layer1 = nn.ModuleList([
            Block(c[0], c[0]),
            Block(c[0], c[0])
        ])
        self.layer2 = nn.ModuleList([
            Block(c[0], c[1], stride=2),
            Block(c[1], c[1])
        ])
        self.layer3 = nn.ModuleList([
            Block(c[1], c[2], stride=2),
            Block(c[2], c[2])
        ])
        self.layer4 = nn.ModuleList([
            Block(c[2], c[3], stride=2),
            Block(c[3], c[3])
        ])

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(c[3], out_channels)

    def forward(self, x):

        x = F.relu(self.bn0(self.conv0(x)))

        for i in range(len(self.layer1)):
            x = self.layer1[i](x)
        for i in range(len(self.layer2)):
            x = self.layer2[i](x)
        for i in range(len(self.layer3)):
            x = self.layer3[i](x)
        for i in range(len(self.layer4)):
            x = self.layer4[i](x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x


