'''Pre-activation ResNet in PyTorch.

Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Identity Mappings in Deep Residual Networks. arXiv:1603.05027
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import activation_functions as actfuns
import util
import math


class PreActBlock(nn.Module):
    '''Pre-activation version of the BasicBlock.'''
    expansion = 1

    def __init__(self, actfun, p, k, g, permute_type, alpha_dist, reduce_actfuns,
                 in_planes, planes, stride=1):
        super(PreActBlock, self).__init__()

        self.actfun = actfun
        self.p, self.k, self.g = p, k, g
        self.permute_type = permute_type
        self.alpha_dist = alpha_dist
        self.reduce_actfuns = reduce_actfuns

        pk_ratio = util.get_pk_ratio(self.actfun, self.p, self.k, self.g)
        if actfun == 'bin_partition_full':
            post_act_in_planes = int((in_planes * self.p) + (2 * math.floor((in_planes * self.p) / (3 * self.k)) * (
                    1 - self.k)))
            post_act_planes = int((planes * self.p) + (2 * math.floor((planes * self.p) / (3 * self.k)) * (
                    1 - self.k)))
        else:
            post_act_in_planes = int(in_planes * pk_ratio)
            post_act_planes = int(planes * pk_ratio)

        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(post_act_in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False,
                               groups=self.g)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(post_act_planes, planes, kernel_size=3, stride=1, padding=1, bias=False, groups=self.g)

        self.conv_layers = nn.ModuleList([self.conv1, self.conv2])
        self.batch_norms = nn.ModuleList([self.bn1, self.bn2])

        self.shuffle_maps = []
        self.shuffle_maps = util.add_shuffle_map(self.shuffle_maps, in_planes, self.p)
        self.shuffle_maps = util.add_shuffle_map(self.shuffle_maps, planes, self.p)

        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False)
            )
            self.conv_layers.append(self.shortcut)

        self.all_alpha_primes = nn.ParameterList()
        self.alpha_dist = alpha_dist
        if self.actfun == "combinact":
            self.num_combinact_actfuns = len(
                actfuns.get_combinact_actfuns(reduce_actfuns))  # Number of actfuns used by combinact
            if alpha_dist == "per_cluster":
                self.all_alpha_primes.append(nn.Parameter(torch.zeros(post_act_in_planes, self.num_combinact_actfuns)))
                self.all_alpha_primes.append(nn.Parameter(torch.zeros(post_act_planes, self.num_combinact_actfuns)))
            if alpha_dist == "per_perm":
                for layer in range(2):
                    self.all_alpha_primes.append(nn.Parameter(torch.zeros(self.p, self.num_combinact_actfuns)))

    def forward(self, x):
        alpha_primes = None
        if self.actfun == 'combinact':
            alpha_primes = self.all_alpha_primes[0]
        out = self.bn1(x)
        out = actfuns.activate(out, actfun=self.actfun,
                               k=self.k, p=self.p, M=x.shape[1],
                               layer_type='conv',
                               permute_type=self.permute_type,
                               shuffle_maps=self.shuffle_maps[0],
                               alpha_primes=alpha_primes,
                               alpha_dist=self.alpha_dist,
                               reduce_actfuns=self.reduce_actfuns)
        out = self.conv1(out)

        if self.actfun == 'combinact':
            alpha_primes = self.all_alpha_primes[1]
        out = self.bn2(out)
        out = actfuns.activate(out, actfun=self.actfun,
                               k=self.k, p=self.p, M=x.shape[1],
                               layer_type='conv',
                               permute_type=self.permute_type,
                               shuffle_maps=self.shuffle_maps[1],
                               alpha_primes=alpha_primes,
                               alpha_dist=self.alpha_dist,
                               reduce_actfuns=self.reduce_actfuns)
        out = self.conv2(out)

        shortcut = self.shortcut(x) if hasattr(self, 'shortcut') else x
        out += shortcut

        return out


class PreActBottleneck(nn.Module):
    '''Pre-activation version of the original Bottleneck module.'''
    expansion = 4

    def __init__(self, actfun, p, k, g, permute_type, alpha_dist, reduce_actfuns,
                 in_planes, planes, stride=1):
        super(PreActBottleneck, self).__init__()

        self.actfun = actfun
        self.p, self.k, self.g = p, k, g
        self.permute_type = permute_type
        self.alpha_dist = alpha_dist
        self.reduce_actfuns = reduce_actfuns

        pk_ratio = util.get_pk_ratio(self.actfun, self.p, self.k, self.g)
        if actfun == 'bin_partition_full':
            post_act_in_planes = int((in_planes * self.p) + (2 * math.floor((in_planes * self.p) / (3 * self.k)) * (
                    1 - self.k)))
            post_act_planes = int((planes * self.p) + (2 * math.floor((planes * self.p) / (3 * self.k)) * (
                    1 - self.k)))
        else:
            post_act_in_planes = int(in_planes * pk_ratio)
            post_act_planes = int(planes * pk_ratio)

        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(post_act_in_planes, planes, kernel_size=1, bias=False, groups=self.g)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(post_act_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False,
                               groups=self.g)
        self.bn3 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(post_act_planes, self.expansion * planes, kernel_size=1, bias=False, groups=self.g)

        self.conv_layers = nn.ModuleList([self.conv1, self.conv2, self.conv3])
        self.batch_norms = nn.ModuleList([self.bn1, self.bn2, self.bn3])

        self.shuffle_maps = []
        self.shuffle_maps = util.add_shuffle_map(self.shuffle_maps, in_planes, self.p)
        self.shuffle_maps = util.add_shuffle_map(self.shuffle_maps, planes, self.p)
        self.shuffle_maps = util.add_shuffle_map(self.shuffle_maps, planes, self.p)

        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False)
            )
            self.conv_layers.append(self.shortcut)

        self.all_alpha_primes = nn.ParameterList()  # List of our trainable alpha prime values
        self.alpha_dist = alpha_dist  # Reference to chosen alpha distribution
        if self.actfun == "combinact":
            self.num_combinact_actfuns = len(
                actfuns.get_combinact_actfuns(reduce_actfuns))  # Number of actfuns used by combinact
            if alpha_dist == "per_cluster":
                self.all_alpha_primes.append(nn.Parameter(torch.zeros(post_act_in_planes, self.num_combinact_actfuns)))
                self.all_alpha_primes.append(nn.Parameter(torch.zeros(post_act_planes, self.num_combinact_actfuns)))
                self.all_alpha_primes.append(nn.Parameter(torch.zeros(post_act_planes, self.num_combinact_actfuns)))
            if alpha_dist == "per_perm":
                for layer in range(3):
                    self.all_alpha_primes.append(nn.Parameter(torch.zeros(self.p, self.num_combinact_actfuns)))

    def forward(self, x):
        alpha_primes = None
        if self.actfun == 'combinact':
            alpha_primes = self.all_alpha_primes[0]
        out = self.bn1(x)
        out = actfuns.activate(out, actfun=self.actfun,
                               k=self.k, p=self.p, M=x.shape[1],
                               layer_type='conv',
                               permute_type=self.permute_type,
                               shuffle_maps=self.shuffle_maps[0],
                               alpha_primes=alpha_primes,
                               alpha_dist=self.alpha_dist,
                               reduce_actfuns=self.reduce_actfuns)
        out = self.conv1(out)

        if self.actfun == 'combinact':
            alpha_primes = self.all_alpha_primes[1]
        out = self.bn2(out)
        out = actfuns.activate(out, actfun=self.actfun,
                               k=self.k, p=self.p, M=x.shape[1],
                               layer_type='conv',
                               permute_type=self.permute_type,
                               shuffle_maps=self.shuffle_maps[1],
                               alpha_primes=alpha_primes,
                               alpha_dist=self.alpha_dist,
                               reduce_actfuns=self.reduce_actfuns)
        out = self.conv2(out)

        alpha_primes = None
        if self.actfun == 'combinact':
            alpha_primes = self.all_alpha_primes[2]
        out = self.bn3(out)
        out = actfuns.activate(out, actfun=self.actfun,
                               k=self.k, p=self.p, M=x.shape[1],
                               layer_type='conv',
                               permute_type=self.permute_type,
                               shuffle_maps=self.shuffle_maps[2],
                               alpha_primes=alpha_primes,
                               alpha_dist=self.alpha_dist,
                               reduce_actfuns=self.reduce_actfuns)
        out = self.conv3(out)

        shortcut = self.shortcut(x) if hasattr(self, 'shortcut') else x
        out += shortcut
        return out


class PreActResNet(nn.Module):
    def __init__(self, block, num_blocks,
                 actfun,
                 num_input_channels=3,
                 num_outputs=10,
                 k=2, p=1, g=1,
                 alpha_dist="per_cluster",
                 permute_type="shuffle",
                 reduce_actfuns=False):
        super(PreActResNet, self).__init__()
        self.actfun = actfun
        self.p, self.k, self.g = p, k, g
        actfuns_1d = ['relu', 'abs', 'swish', 'leaky_relu']
        if actfun in actfuns_1d:
            self.k = 1
        self.permute_type = permute_type
        self.alpha_dist = alpha_dist
        self.reduce_actfuns = reduce_actfuns

        block_sizes = [64, 128, 256, 512]
        for i, block_size in enumerate(block_sizes):
            block_sizes[i] = self.k * self.g * int(block_size / (self.k * self.g))
        self.in_planes = block_sizes[0]

        self.conv_layers = nn.ModuleList([
            nn.Conv2d(num_input_channels, block_sizes[0], kernel_size=3, stride=1, padding=1, bias=False)
        ])
        self.batch_norms = nn.ModuleList([])

        self.all_alpha_primes = nn.ParameterList()
        self.layer1 = self._make_layer(block, block_sizes[0], num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, block_sizes[1], num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, block_sizes[2], num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, block_sizes[3], num_blocks[3], stride=2)

        self.linear_layers = nn.ModuleList([
            nn.Linear(block_sizes[3] * block.expansion, num_outputs)
        ])

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            curr_layer = block(self.actfun, self.p, self.k, self.g,
                               self.permute_type, self.alpha_dist, self.reduce_actfuns,
                               self.in_planes, planes, stride)
            self.conv_layers += curr_layer.conv_layers
            self.batch_norms += curr_layer.batch_norms
            self.all_alpha_primes += curr_layer.all_alpha_primes
            layers.append(curr_layer)
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv_layers[0](x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear_layers[0](out)
        return out


def ResNet(resnet_ver,
           actfun,
           num_input_channels=3,
           num_outputs=10,
           k=2, p=1, g=1,
           alpha_dist="per_cluster",
           permute_type="shuffle",
           reduce_actfuns=False,
           wide=False):
    #
    # if wide == True:

    if resnet_ver == 18:
        block = PreActBlock
        num_blocks = [2, 2, 2, 2]
    elif resnet_ver == 34:
        block = PreActBlock
        num_blocks = [3, 4, 6, 3]
    elif resnet_ver == 50:
        block = PreActBottleneck
        num_blocks = [3, 4, 6, 3]
    elif resnet_ver == 101:
        block = PreActBottleneck
        num_blocks = [3, 4, 23, 3]
    elif resnet_ver == 152:
        block = PreActBottleneck
        num_blocks = [3, 8, 36, 3]

    return PreActResNet(block, num_blocks, actfun,
                        num_input_channels,
                        num_outputs,
                        k, p, g,
                        alpha_dist,
                        permute_type,
                        reduce_actfuns)
