import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import actfuns
import util


class BottleneckBlock(nn.Module):

    expansion = 4

    def __init__(self, c_in, c_out, hyper_params, stride=1):
        super(BottleneckBlock, self).__init__()

        # -------- Calculating number of input channels for each layer after applying activations
        actfun = hyper_params['actfun'] if 'actfun' in hyper_params else 'relu'
        self.actfun = actfuns.actfun_name2factory(actfun)()
        divisor = getattr(self.actfun, "divisor", 1)
        feature_factor = getattr(self.actfun, "feature_factor", 1)
        width = hyper_params['width'] if 'width' in hyper_params else 1

        out = int(int(round((c_out * width) / divisor)) * divisor)

        conv1_in = int(c_in * feature_factor)
        conv2_in = int(out * feature_factor)
        conv3_in = int(out * feature_factor)

        self.bn1 = nn.BatchNorm2d(c_in)
        self.conv1 = nn.Conv2d(conv1_in, out, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out)
        self.conv2 = nn.Conv2d(conv2_in, out, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out)
        self.conv3 = nn.Conv2d(conv3_in, c_out * self.expansion, kernel_size=1, bias=False)

        self.proj = (c_in != self.expansion * c_out or stride > 1)
        if self.proj:
            self.conv_proj = nn.Conv2d(c_in, self.expansion * c_out, kernel_size=1, stride=stride, padding=0, bias=False)

    def forward(self, x):

        identity = x.clone().to(x.device)

        alpha_primes = None
        x = self.bn1(x)
        x = self.actfun(x)
        x = self.conv1(x)

        alpha_primes = None
        x = self.bn2(x)
        x = self.actfun(x)
        x = self.conv2(x)

        alpha_primes = None
        x = self.bn3(x)
        x = self.actfun(x)
        x = self.conv3(x)

        if self.proj:
            identity = self.conv_proj(identity)

        x += identity

        return x


class PreActResNet(nn.Module):

    def __init__(self, resnet_ver, **kwargs):
        super(PreActResNet, self).__init__()

        # -------- Retrieving model architecture
        block, num_blocks = self.get_version(resnet_ver)
        self.hyper_params = kwargs

        # -------- Error handling
        assert len(num_blocks) == 4, "Network must have four layers"
        assert all(i >= 0 for i in num_blocks), "All layers must have one or more block(s)"

        # -------- Setting number of layer channels
        self.actfun = kwargs['actfun'] if 'actfun' in kwargs else 'relu'
        self.k = kwargs['k'] if 'k' in kwargs else 1
        self.p = kwargs['p'] if 'p' in kwargs else 1
        self.g = kwargs['g'] if 'g' in kwargs else 1
        self.width = kwargs['width'] if 'width' in kwargs else 1
        if self.actfun == 'relu':
            assert self.k == 1, "k = {} with ReLU activation. ReLU cannot have k != 1".format(self.k)

        c = kwargs['c'] if 'c' in kwargs else 64
        c = [c, 2 * c, 4 * c, 8 * c]
        for i, curr_num_params in enumerate(c):
            c[i] = self.k * self.g * int(curr_num_params / (self.k * self.g))
        self.inplanes = c[0]
        # print("c = {}".format(c))

        # -------- Defining layers in network
        in_channels = kwargs['in_channels'] if 'in_channels' in kwargs else 3
        out_channels = kwargs['out_channels'] if 'out_channels' in kwargs else 10
        self.conv0 = nn.Conv2d(in_channels, c[0], kernel_size=3, stride=1, padding=1, bias=False)
        self.bn0 = nn.BatchNorm2d(c[0])
        self.layer1 = self.make_layer(block, num_blocks[0], c=c[0], hyper_params=kwargs)
        self.layer2 = self.make_layer(block, num_blocks[1], c=c[1], hyper_params=kwargs, stride=2)
        self.layer3 = self.make_layer(block, num_blocks[2], c=c[2], hyper_params=kwargs, stride=2)
        self.layer4 = self.make_layer(block, num_blocks[3], c=c[3], hyper_params=kwargs, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(self.inplanes, out_channels)

    def get_version(self, resnet_ver):
        block = BottleneckBlock
        if resnet_ver == 18:
            num_blocks = [2, 2, 2, 2]
        elif resnet_ver == 34:
            num_blocks = [3, 4, 6, 3]
        elif resnet_ver == 50:
            num_blocks = [3, 4, 6, 3]
        elif resnet_ver == 101:
            num_blocks = [3, 4, 23, 3]
        elif resnet_ver == 152:
            num_blocks = [3, 8, 36, 3]
        else:
            num_blocks = [2, 2, 2, 2]

        return block, num_blocks

    def make_layer(self, block, num_blocks, c, hyper_params, stride=1):

        layer = nn.ModuleList([])
        for i in range(num_blocks):
            layer.append(block(self.inplanes, c, hyper_params=hyper_params, stride=stride))
            if i == 0:
                self.inplanes = c * block.expansion
                stride = 1

        return layer

    def forward(self, x):

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

        return x
