import torch
import torch.utils.data
import torch.nn as nn
import activation_functions as actfuns
import util
import math
import time


class CNN(nn.Module):

    def __init__(self, actfun,
                 num_input_channels=3,
                 input_dim=32,
                 num_outputs=10,
                 k=2, p=1, g=1,
                 alpha_dist="per_cluster",
                 permute_type="shuffle",
                 reduce_actfuns=False,
                 num_params=3000000):
        super(CNN, self).__init__()

        if permute_type == 'invert' and p % k != 0:
            p = k

        self.actfun = actfun
        self.p, self.k, self.g = p, k, g
        self.permute_type = permute_type
        self.alpha_dist = alpha_dist
        self.shuffle_maps = []
        self.reduce_actfuns = reduce_actfuns

        pk_ratio = util.get_pk_ratio(self.actfun, self.p, self.k, self.g)
        pre_acts = util.calc_cnn_preacts(num_params, num_input_channels, num_outputs, input_dim,
                                         pk_ratio, self.p, self.k, self.g)

        post_acts = []
        for i, pre_act in enumerate(pre_acts):
            pre_acts[i] = self.k * self.g * int(pre_act / (self.k * self.g))
            if pre_acts[i] == 0:
                pre_acts[i] = self.k * self.g
            if actfun == 'bin_partition_full':
                post_acts.append(int((pre_acts[i] * self.p) + (2 * math.floor((pre_acts[i] * self.p) / (3 * self.k)) * (
                        1 - self.k))))
            else:
                post_acts.append(int(pre_acts[i] * pk_ratio))

        self.conv_layers = nn.ModuleList([
            nn.ModuleList([nn.Conv2d(in_channels=num_input_channels, out_channels=int(pre_acts[0]),
                                     kernel_size=3, padding=1),
                           nn.Conv2d(in_channels=int(post_acts[0]), out_channels=int(pre_acts[1]),
                                     kernel_size=3, padding=1, groups=self.g)]),
            nn.ModuleList([nn.Conv2d(in_channels=int(post_acts[1]), out_channels=int(pre_acts[2]),
                                     kernel_size=3, padding=1, groups=self.g),
                           nn.Conv2d(in_channels=int(post_acts[2]), out_channels=int(pre_acts[2]),
                                     kernel_size=3, padding=1, groups=self.g)]),
            nn.ModuleList([nn.Conv2d(in_channels=int(post_acts[2]), out_channels=int(pre_acts[3]),
                                     kernel_size=3, padding=1, groups=self.g),
                           nn.Conv2d(in_channels=int(post_acts[3]), out_channels=int(pre_acts[3]),
                                     kernel_size=3, padding=1, groups=self.g)])
        ])

        self.shuffle_maps = util.add_shuffle_map(self.shuffle_maps, int(pre_acts[0]), self.p)
        self.shuffle_maps = util.add_shuffle_map(self.shuffle_maps, int(pre_acts[1]), self.p)
        self.shuffle_maps = util.add_shuffle_map(self.shuffle_maps, int(pre_acts[2]), self.p)
        self.shuffle_maps = util.add_shuffle_map(self.shuffle_maps, int(pre_acts[2]), self.p)
        self.shuffle_maps = util.add_shuffle_map(self.shuffle_maps, int(pre_acts[3]), self.p)
        self.shuffle_maps = util.add_shuffle_map(self.shuffle_maps, int(pre_acts[3]), self.p)

        self.batch_norms = nn.ModuleList([
            nn.BatchNorm2d(int(pre_acts[0])),
            nn.BatchNorm2d(int(pre_acts[2])),
            nn.BatchNorm2d(int(pre_acts[3])),
        ])

        self.pooling = nn.ModuleList([
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.MaxPool2d(kernel_size=2, stride=2)
        ])

        self.linear_layers = nn.ModuleDict()
        self.linear_layers['l1'] = nn.ModuleList()
        self.linear_layers['l2'] = nn.ModuleList()

        for group in range(g):
            self.linear_layers['l1'].append(nn.Linear(int(post_acts[3] * (int(input_dim / 8) ** 2) / self.g),
                                                      int(pre_acts[5] / self.g)))
            self.linear_layers['l2'].append(nn.Linear(int(post_acts[5] / self.g), int(pre_acts[4] / self.g)))
        self.linear_layers['l3'] = nn.Linear(int(post_acts[4]), int(num_outputs))

        self.shuffle_maps = util.add_shuffle_map(self.shuffle_maps, int(pre_acts[5]), self.p)
        self.shuffle_maps = util.add_shuffle_map(self.shuffle_maps, int(pre_acts[4]), self.p)

        self.all_alpha_primes = nn.ParameterList()  # List of our trainable alpha prime values
        self.alpha_dist = alpha_dist  # Reference to chosen alpha distribution
        if self.actfun == "combinact":
            self.num_combinact_actfuns = len(
                actfuns.get_combinact_actfuns(reduce_actfuns))  # Number of actfuns used by combinact
            if alpha_dist == "per_cluster":
                self.all_alpha_primes.append(nn.Parameter(torch.zeros(int(post_acts[0]), self.num_combinact_actfuns)))
                self.all_alpha_primes.append(nn.Parameter(torch.zeros(int(post_acts[1]), self.num_combinact_actfuns)))
                self.all_alpha_primes.append(nn.Parameter(torch.zeros(int(post_acts[2]), self.num_combinact_actfuns)))
                self.all_alpha_primes.append(nn.Parameter(torch.zeros(int(post_acts[2]), self.num_combinact_actfuns)))
                self.all_alpha_primes.append(nn.Parameter(torch.zeros(int(post_acts[3]), self.num_combinact_actfuns)))
                self.all_alpha_primes.append(nn.Parameter(torch.zeros(int(post_acts[3]), self.num_combinact_actfuns)))
                self.all_alpha_primes.append(nn.Parameter(torch.zeros(int(post_acts[5]), self.num_combinact_actfuns)))
                self.all_alpha_primes.append(nn.Parameter(torch.zeros(int(post_acts[4]), self.num_combinact_actfuns)))
            if alpha_dist == "per_perm":
                for layer in range(8):
                    self.all_alpha_primes.append(nn.Parameter(torch.zeros(self.p, self.num_combinact_actfuns)))

    def forward(self, x):

        # ------------- Conv layers
        if self.actfun == 'l2_lae':
            actfun = 'l2'
        else:
            actfun = self.actfun

        if len(x.shape) != 4:
            x = x.unsqueeze(3)

        x = torch.transpose(x, 1, 3)
        x = torch.transpose(x, 2, 3)

        for block in range(3):
            x = self.conv_layers[block][0](x)
            x = self.batch_norms[block](x)
            if actfun == 'combinact':
                alpha_primes = self.all_alpha_primes[block * 2]
            else:
                alpha_primes = None
            x = actfuns.activate(x, actfun=actfun,
                                 k=self.k, p=self.p, M=x.shape[1],
                                 layer_type='conv',
                                 permute_type=self.permute_type,
                                 shuffle_maps=self.shuffle_maps[block * 2],
                                 alpha_primes=alpha_primes,
                                 alpha_dist=self.alpha_dist,
                                 reduce_actfuns=self.reduce_actfuns)
            x = self.conv_layers[block][1](x)
            if actfun == 'combinact':
                alpha_primes = self.all_alpha_primes[(block * 2) + 1]
            else:
                alpha_primes = None
            x = actfuns.activate(x, actfun=actfun,
                                 k=self.k, p=self.p, M=x.shape[1],
                                 layer_type='conv',
                                 permute_type=self.permute_type,
                                 shuffle_maps=self.shuffle_maps[(block * 2) + 1],
                                 alpha_primes=alpha_primes,
                                 alpha_dist=self.alpha_dist,
                                 reduce_actfuns=self.reduce_actfuns)
            x = self.pooling[block](x)

        x = x.view(x.size(0), -1)

        # ------------- Linear layers
        if self.actfun == 'l2_lae':
            self.actfun = 'lae'

        x = self.grouped_fc(x, self.linear_layers['l1'])
        if self.actfun == 'combinact':
            alpha_primes = self.all_alpha_primes[6]
        else:
            alpha_primes = None
        x = actfuns.activate(x, actfun=self.actfun,
                             k=self.k, p=self.p, M=x.shape[1],
                             layer_type='linear',
                             permute_type=self.permute_type,
                             shuffle_maps=self.shuffle_maps[6],
                             alpha_primes=alpha_primes,
                             alpha_dist=self.alpha_dist,
                             reduce_actfuns=self.reduce_actfuns)

        x = self.grouped_fc(x, self.linear_layers['l2'])
        if self.actfun == 'combinact':
            alpha_primes = self.all_alpha_primes[7]
        else:
            alpha_primes = None
        x = actfuns.activate(x, actfun=self.actfun,
                             k=self.k, p=self.p, M=x.shape[1],
                             layer_type='linear',
                             permute_type=self.permute_type,
                             shuffle_maps=self.shuffle_maps[7],
                             alpha_primes=alpha_primes,
                             alpha_dist=self.alpha_dist,
                             reduce_actfuns=self.reduce_actfuns)

        x = self.linear_layers['l3'](x)

        return x

    def grouped_fc(self, x, linear_layers):
        all_outputs = None
        for group_idx, group_fc in enumerate(linear_layers):
            group_idx_start = group_idx * int(x.shape[1] / self.g)
            group_idx_end = (group_idx + 1) * int(x.shape[1] / self.g)
            curr_inputs = x[:, group_idx_start:group_idx_end]
            curr_outputs = group_fc(curr_inputs)
            if group_idx == 0:
                all_outputs = curr_outputs
            else:
                all_outputs = torch.cat((all_outputs, curr_outputs), dim=1)
        return all_outputs