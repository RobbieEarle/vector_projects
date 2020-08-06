import torch
import torch.utils.data
import torch.nn as nn
import activation_functions as actfuns
import util
import math


class CombinactNN(nn.Module):

    def __init__(self, actfun,
                 input_dim=784,
                 output_dim=10,
                 num_layers=2,
                 k=2, p=1,
                 alpha_dist="per_cluster",
                 permute_type="shuffle",
                 reduce_actfuns=False,
                 pfact=1):
        super(CombinactNN, self).__init__()

        # Validate input
        # error = util.test_net_inputs(actfun, net_struct)
        # if error is not None:
        #     raise ValueError(error)

        self.input_dim = input_dim
        self.actfun = actfun
        self.k, self.p = k, p
        if actfun == 'relu' or actfun == 'abs':
            self.k = 1
        self.permute_type = permute_type
        self.alpha_dist = alpha_dist
        self.shuffle_maps = []
        self.reduce_actfuns = reduce_actfuns

        pre_acts = [int(250 * pfact),
                    int(200 * pfact),
                    int(100 * pfact)]
        post_acts = []
        for i, pre_act in enumerate(pre_acts):
            pre_acts[i] = self.k * int(pre_act / self.k)
            post_acts.append(0)
            if actfun == 'binary_ops_partition':
                post_acts[i] = int((pre_acts[i] * self.p) + (2 * math.floor((pre_acts[i] * self.p) / (3 * self.k)) * (
                    1 - self.k)))
            elif actfun == 'binary_ops_all':
                post_acts[i] = int(pre_acts[i] * self.p * ((2 / self.k) + 1))
            else:
                post_acts[i] = int(pre_acts[i] * self.p / self.k)

        if num_layers == 2:
            self.linear_layers = nn.ModuleList([
                nn.Linear(input_dim, pre_acts[0]),
                nn.Linear(post_acts[0], pre_acts[1]),
                nn.Linear(post_acts[1], output_dim)
            ])
            self.batch_norms = nn.ModuleList([
                nn.BatchNorm1d(pre_acts[0]),
                nn.BatchNorm1d(pre_acts[1])
            ])

            self.shuffle_maps = util.add_shuffle_map(self.shuffle_maps, pre_acts[0], self.p)
            self.shuffle_maps = util.add_shuffle_map(self.shuffle_maps, pre_acts[1], self.p)

        elif num_layers == 3:
            self.linear_layers = nn.ModuleList([
                nn.Linear(input_dim, pre_acts[0]),
                nn.Linear(post_acts[0], pre_acts[1]),
                nn.Linear(post_acts[1], pre_acts[2]),
                nn.Linear(post_acts[2], output_dim)
            ])
            self.batch_norms = nn.ModuleList([
                nn.BatchNorm1d(pre_acts[0]),
                nn.BatchNorm1d(pre_acts[1]),
                nn.BatchNorm1d(pre_acts[1]),
            ])

            self.shuffle_maps = util.add_shuffle_map(self.shuffle_maps, pre_acts[0], self.p)
            self.shuffle_maps = util.add_shuffle_map(self.shuffle_maps, pre_acts[1], self.p)
            self.shuffle_maps = util.add_shuffle_map(self.shuffle_maps, pre_acts[2], self.p)

        self.all_alpha_primes = nn.ParameterList()
        self.alpha_dist = alpha_dist
        if self.actfun == "combinact":
            self.num_combinact_actfuns = len(actfuns.get_combinact_actfuns(reduce_actfuns))
            if alpha_dist == "per_cluster":
                self.all_alpha_primes.append(nn.Parameter(torch.zeros(post_acts[0], self.num_combinact_actfuns)))
                self.all_alpha_primes.append(nn.Parameter(torch.zeros(post_acts[1], self.num_combinact_actfuns)))
                if num_layers == 3:
                    self.all_alpha_primes.append(
                        nn.Parameter(torch.zeros(post_acts[2], self.num_combinact_actfuns)))
            if alpha_dist == "per_perm":
                for layer in range(num_layers):
                    self.all_alpha_primes.append(nn.Parameter(torch.zeros(self.p, self.num_combinact_actfuns)))

    def forward(self, x):

        x = x.reshape(x.size(0), self.input_dim)

        for i, fc in enumerate(self.linear_layers):
            x = fc(x)
            if i < len(self.linear_layers) - 1:
                if self.actfun == 'combinact':
                    alpha_primes = self.all_alpha_primes[i]
                else:
                    alpha_primes = None
                x = self.batch_norms[i](x)
                x = actfuns.activate(x, actfun=self.actfun,
                                     k=self.k, p=self.p, M=x.shape[1],
                                     layer_type='linear',
                                     permute_type=self.permute_type,
                                     shuffle_maps=self.shuffle_maps[i],
                                     alpha_primes=alpha_primes,
                                     alpha_dist=self.alpha_dist,
                                     reduce_actfuns=self.reduce_actfuns)

        return x


class CombinactCNN(nn.Module):

    def __init__(self, actfun,
                 num_input_channels=3,
                 input_dim=32,
                 num_outputs=10,
                 k=2, p=1,
                 alpha_dist="per_cluster",
                 permute_type="shuffle",
                 reduce_actfuns=False,
                 pfact=1):
        super(CombinactCNN, self).__init__()

        self.actfun = actfun
        self.k, self.p = k, p
        if actfun == 'relu' or actfun == 'abs':
            self.k = 1
        self.permute_type = permute_type
        self.alpha_dist = alpha_dist
        self.shuffle_maps = []
        self.reduce_actfuns = reduce_actfuns

        pre_acts = [int(32 * pfact),
                 int(64 * pfact),
                 int(128 * pfact),
                 int(256 * pfact),
                 int(512 * pfact),
                 int(1024 * pfact)]
        post_acts = []
        for i, pre_act in enumerate(pre_acts):
            pre_acts[i] = self.k * int(pre_act / self.k)
            post_acts.append(0)
            if actfun == 'binary_ops_partition':
                post_acts[i] = int((pre_acts[i] * self.p) + (2 * math.floor((pre_acts[i] * self.p) / (3 * self.k)) * (
                        1 - self.k)))
            elif actfun == 'binary_ops_all':
                post_acts[i] = int(pre_acts[i] * self.p * ((2 / self.k) + 1))
            else:
                post_acts[i] = int(pre_acts[i] * self.p / self.k)

        self.conv_layers = nn.ModuleList([
            nn.ModuleList([nn.Conv2d(in_channels=num_input_channels, out_channels=pre_acts[0], kernel_size=3, padding=1),
                           nn.Conv2d(in_channels=int(post_acts[0]), out_channels=pre_acts[1], kernel_size=3, padding=1)]),
            nn.ModuleList([nn.Conv2d(in_channels=int(post_acts[1]), out_channels=pre_acts[2], kernel_size=3, padding=1),
                           nn.Conv2d(in_channels=int(post_acts[2]), out_channels=pre_acts[2], kernel_size=3, padding=1)]),
            nn.ModuleList([nn.Conv2d(in_channels=int(post_acts[2]), out_channels=pre_acts[3], kernel_size=3, padding=1),
                           nn.Conv2d(in_channels=int(post_acts[3]), out_channels=pre_acts[3], kernel_size=3, padding=1)])
        ])

        self.shuffle_maps = util.add_shuffle_map(self.shuffle_maps, pre_acts[0], self.p)
        self.shuffle_maps = util.add_shuffle_map(self.shuffle_maps, pre_acts[1], self.p)
        self.shuffle_maps = util.add_shuffle_map(self.shuffle_maps, pre_acts[2], self.p)
        self.shuffle_maps = util.add_shuffle_map(self.shuffle_maps, pre_acts[2], self.p)
        self.shuffle_maps = util.add_shuffle_map(self.shuffle_maps, pre_acts[3], self.p)
        self.shuffle_maps = util.add_shuffle_map(self.shuffle_maps, pre_acts[3], self.p)

        self.batch_norms = nn.ModuleList([
            nn.BatchNorm2d(pre_acts[0]),
            nn.BatchNorm2d(pre_acts[2]),
            nn.BatchNorm2d(pre_acts[3]),
        ])

        self.pooling = nn.ModuleList([
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.MaxPool2d(kernel_size=2, stride=2)
        ])

        self.linear_layers = nn.ModuleList([
            nn.Linear(int(post_acts[3]) * (int(input_dim / 8) ** 2), pre_acts[5]),
            nn.Linear(int(post_acts[5]), pre_acts[4]),
            nn.Linear(int(post_acts[4]), num_outputs)
        ])

        self.shuffle_maps = util.add_shuffle_map(self.shuffle_maps, pre_acts[5], self.p)
        self.shuffle_maps = util.add_shuffle_map(self.shuffle_maps, pre_acts[4], self.p)

        self.all_alpha_primes = nn.ParameterList()  # List of our trainable alpha prime values
        self.alpha_dist = alpha_dist  # Reference to chosen alpha distribution
        if self.actfun == "combinact":
            self.num_combinact_actfuns = len(actfuns.get_combinact_actfuns(reduce_actfuns))  # Number of actfuns used by combinact
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

        if self.actfun == 'l2_lae':
            actfun = 'l2'
        else:
            actfun = self.actfun

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

        if self.actfun == 'l2_lae':
            actfun = 'lae'

        for i, fc in enumerate(self.linear_layers):
            x = fc(x)
            if i < len(self.linear_layers) - 1:
                if actfun == 'combinact':
                    alpha_primes = self.all_alpha_primes[6 + i]
                else:
                    alpha_primes = None
                x = actfuns.activate(x, actfun=actfun,
                                     k=self.k, p=self.p, M=x.shape[1],
                                     layer_type='linear',
                                     permute_type=self.permute_type,
                                     shuffle_maps=self.shuffle_maps[6+i],
                                     alpha_primes=alpha_primes,
                                     alpha_dist=self.alpha_dist,
                                     reduce_actfuns=self.reduce_actfuns)

        return x
