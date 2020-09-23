import torch
import torch.nn as nn
import torch.nn.functional as F
import time


class Block(nn.Module):

    def __init__(self, c_in, c_out, stride=1, **kwargs):
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


class BottleneckBlock(nn.Module):

    def __init__(self, c_in, c_out, stride=1, width=1):
        super(BottleneckBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(c_in)
        self.conv1 = nn.Conv2d(c_in, c_out, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(c_out)
        self.conv2 = nn.Conv2d(c_out, c_out * width, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(c_out * width)
        self.conv3 = nn.Conv2d(c_out * width, c_out, kernel_size=1, bias=False)

        self.proj = (c_in != c_out or stride > 1)
        if self.proj:
            self.conv_proj = nn.Conv2d(c_in, c_out, kernel_size=1, stride=stride, padding=0, bias=False)

    def forward(self, x):

        identity = x.clone().to(x.device)

        x = self.bn1(x)
        x = F.relu(x)
        x = self.conv1(x)

        x = self.bn2(x)
        x = F.relu(x)
        x = self.conv2(x)

        x = self.bn3(x)
        x = F.relu(x)
        x = self.conv3(x)

        if self.proj:
            identity = self.conv_proj(identity)

        x += identity

        return x


class PreActResNet(nn.Module):

    def __init__(self, block, num_blocks, width=1, in_channels=3, out_channels=10, c=64):
        super(PreActResNet, self).__init__()

        assert len(num_blocks) == 4, "Network must have four layers"

        c = [c, 2 * c, 4 * c, 8 * c]

        self.conv0 = nn.Conv2d(in_channels, c[0], kernel_size=3, stride=1, padding=1, bias=False)
        self.bn0 = nn.BatchNorm2d(c[0])

        self.layer1 = self.make_layer(block, num_blocks[0], in_dim=c[0], out_dim=c[0], width=width)
        self.layer2 = self.make_layer(block, num_blocks[1], in_dim=c[0], out_dim=c[1], width=width)
        self.layer3 = self.make_layer(block, num_blocks[2], in_dim=c[1], out_dim=c[2], width=width)
        self.layer4 = self.make_layer(block, num_blocks[3], in_dim=c[2], out_dim=c[3], width=width)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(c[3], out_channels)

    def make_layer(self, block, num_blocks, in_dim, out_dim, width):

        layer = nn.ModuleList([])
        if in_dim == out_dim:
            for i in range(num_blocks):
                layer.append(block(in_dim, out_dim, width=width))
        else:
            for i in range(num_blocks):
                if i == 0:
                    layer.append(block(in_dim, out_dim, stride=2, width=width))
                else:
                    layer.append(block(out_dim, out_dim, width=width))

        return layer

    def forward(self, x):

        start_time = time.time()

        x = F.relu(self.bn0(self.conv0(x)))

        for block in self.layer1:
            x = block(x)
        for block in self.layer2:
            x = block(x)
        for block in self.layer3:
            x = block(x)
        for block in self.layer4:
            x = block(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        print("-> Forward: {}".format(time.time() - start_time))

        return x

