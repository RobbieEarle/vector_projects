import torch
import torch.utils.data
import torch.nn as nn
import activation_functions as actfuns
import util
import time


class CombinactNN(nn.Module):
    def __init__(self, actfun,
                 input_dim=784,
                 output_dim=10,
                 num_layers=2,
                 k=2, p=1,
                 alpha_dist="per_cluster",
                 permute_type="shuffle"):
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
            self.p = 1
        self.permute_type = permute_type
        self.alpha_dist = alpha_dist
        self.shuffle_maps = []

        pk_ratio = self.p / self.k
        if num_layers == 2:
            self.linear_layers = nn.ModuleList([
                nn.Linear(input_dim, 250),
                nn.Linear(int(250 * pk_ratio), 200),
                nn.Linear(int(200 * pk_ratio), output_dim)
            ])
            self.batch_norms = nn.ModuleList([
                nn.BatchNorm1d(250),
                nn.BatchNorm1d(200)
            ])

            self.shuffle_maps = util.add_shuffle_map(self.shuffle_maps, 250, self.p)
            self.shuffle_maps = util.add_shuffle_map(self.shuffle_maps, 200, self.p)

        elif num_layers == 3:
            self.linear_layers = nn.ModuleList([
                nn.Linear(input_dim, 250),
                nn.Linear(int(250 * pk_ratio), 200),
                nn.Linear(int(200 * pk_ratio), 100),
                nn.Linear(int(100 * pk_ratio), output_dim)
            ])
            self.batch_norms = nn.ModuleList([
                nn.BatchNorm1d(250),
                nn.BatchNorm1d(200),
                nn.BatchNorm1d(100),
            ])

            self.shuffle_maps = util.add_shuffle_map(self.shuffle_maps, 250, self.p)
            self.shuffle_maps = util.add_shuffle_map(self.shuffle_maps, 200, self.p)
            self.shuffle_maps = util.add_shuffle_map(self.shuffle_maps, 100, self.p)

        self.all_alpha_primes = nn.ParameterList()  # List of our trainable alpha prime values
        self.alpha_dist = alpha_dist  # Reference to chosen alpha distribution
        if self.actfun == "combinact":
            self.num_combinact_actfuns = len(actfuns.get_combinact_actfuns())  # Number of actfuns used by combinact
            # self.shuffle_maps = []  # List of shuffle maps used for shuffle permutations
            if alpha_dist == "per_cluster":
                self.all_alpha_primes.append(nn.Parameter(torch.zeros(int(250 * self.p / self.k), self.num_combinact_actfuns)))
                self.all_alpha_primes.append(nn.Parameter(torch.zeros(int(200 * self.p / self.k), self.num_combinact_actfuns)))
                if num_layers == 3:
                    self.all_alpha_primes.append(
                        nn.Parameter(torch.zeros(int(100 * self.p / self.k), self.num_combinact_actfuns)))
            if alpha_dist == "per_perm":
                for layer in range(num_layers):
                    self.all_alpha_primes.append(nn.Parameter(torch.zeros(self.p, self.num_combinact_actfuns)))

    def forward(self, x):

        x = x.reshape(x.size(0), self.input_dim)

        for i, fc in enumerate(self.linear_layers):
            x = fc(x)
            if i < len(self.linear_layers) - 1:
                x = self.batch_norms[i](x)
                x = actfuns.activate(x, actfun=self.actfun,
                                     k=self.k, p=self.p, M=x.shape[1],
                                     layer_type='linear',
                                     permute_type=self.permute_type,
                                     shuffle_maps=self.shuffle_maps[i],
                                     alpha_primes=self.all_alpha_primes[i],
                                     alpha_dist=self.alpha_dist)

        return x


class CombinactCNN(nn.Module):
    def __init__(self, actfun,
                 num_input_channels=3,
                 input_dim=32,
                 num_outputs=10,
                 k=2, p=1,
                 alpha_dist="per_cluster",
                 permute_type="shuffle"):
        super(CombinactCNN, self).__init__()

        # Validate input
        # error = util.test_net_inputs(actfun, net_struct)
        # if error is not None:
        #     raise ValueError(error)

        self.actfun = actfun
        self.k, self.p = k, p
        if actfun == 'relu' or actfun == 'abs':
            self.k = 1
        self.permute_type = permute_type
        self.alpha_dist = alpha_dist
        self.shuffle_maps = []

        pk_ratio = self.p / self.k
        self.conv_layers = nn.ModuleList([
            nn.ModuleList([nn.Conv2d(in_channels=num_input_channels, out_channels=32, kernel_size=3, padding=1),
                           nn.Conv2d(in_channels=int(32 * pk_ratio), out_channels=64, kernel_size=3, padding=1)]),
            nn.ModuleList([nn.Conv2d(in_channels=int(64 * pk_ratio), out_channels=128, kernel_size=3, padding=1),
                           nn.Conv2d(in_channels=int(128 * pk_ratio), out_channels=128, kernel_size=3, padding=1)]),
            nn.ModuleList([nn.Conv2d(in_channels=int(128 * pk_ratio), out_channels=256, kernel_size=3, padding=1),
                           nn.Conv2d(in_channels=int(256 * pk_ratio), out_channels=256, kernel_size=3, padding=1)])
        ])

        self.shuffle_maps = util.add_shuffle_map(self.shuffle_maps, 32, self.p)
        self.shuffle_maps = util.add_shuffle_map(self.shuffle_maps, 64, self.p)
        self.shuffle_maps = util.add_shuffle_map(self.shuffle_maps, 128, self.p)
        self.shuffle_maps = util.add_shuffle_map(self.shuffle_maps, 128, self.p)
        self.shuffle_maps = util.add_shuffle_map(self.shuffle_maps, 256, self.p)
        self.shuffle_maps = util.add_shuffle_map(self.shuffle_maps, 256, self.p)

        self.batch_norms = nn.ModuleList([
            nn.BatchNorm2d(32),
            nn.BatchNorm2d(128),
            nn.BatchNorm2d(256),
        ])

        self.pooling = nn.ModuleList([
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.MaxPool2d(kernel_size=2, stride=2)
        ])

        self.linear_layers = nn.ModuleList([
            nn.Linear(int(256 * pk_ratio) * (int(input_dim / 8) ** 2), 1024),
            nn.Linear(int(1024 * pk_ratio), 512),
            nn.Linear(int(512 * pk_ratio), num_outputs)
        ])

        self.shuffle_maps = util.add_shuffle_map(self.shuffle_maps, 1024, self.p)
        self.shuffle_maps = util.add_shuffle_map(self.shuffle_maps, 512, self.p)

        self.all_alpha_primes = nn.ParameterList()  # List of our trainable alpha prime values
        self.alpha_dist = alpha_dist  # Reference to chosen alpha distribution
        if self.actfun == "combinact":
            self.num_combinact_actfuns = len(actfuns.get_combinact_actfuns())  # Number of actfuns used by combinact
            if alpha_dist == "per_cluster":
                self.all_alpha_primes.append(nn.Parameter(torch.zeros(int(32 * self.p / self.k), self.num_combinact_actfuns)))
                self.all_alpha_primes.append(nn.Parameter(torch.zeros(int(64 * self.p / self.k), self.num_combinact_actfuns)))
                self.all_alpha_primes.append(nn.Parameter(torch.zeros(int(128 * self.p / self.k), self.num_combinact_actfuns)))
                self.all_alpha_primes.append(nn.Parameter(torch.zeros(int(128 * self.p / self.k), self.num_combinact_actfuns)))
                self.all_alpha_primes.append(nn.Parameter(torch.zeros(int(256 * self.p / self.k), self.num_combinact_actfuns)))
                self.all_alpha_primes.append(nn.Parameter(torch.zeros(int(256 * self.p / self.k), self.num_combinact_actfuns)))
                self.all_alpha_primes.append(nn.Parameter(torch.zeros(int(1024 * self.p / self.k), self.num_combinact_actfuns)))
                self.all_alpha_primes.append(nn.Parameter(torch.zeros(int(512 * self.p / self.k), self.num_combinact_actfuns)))
            if alpha_dist == "per_perm":
                for layer in range(8):
                    self.all_alpha_primes.append(nn.Parameter(torch.zeros(self.p, self.num_combinact_actfuns)))

    def forward(self, x):
        for block in range(3):
            x = self.conv_layers[block][0](x)
            x = self.batch_norms[block](x)
            x = actfuns.activate(x, actfun=self.actfun,
                                 k=self.k, p=self.p, M=x.shape[1],
                                 layer_type='conv',
                                 permute_type=self.permute_type,
                                 shuffle_maps=self.shuffle_maps[block * 2],
                                 alpha_primes=self.all_alpha_primes[block * 2],
                                 alpha_dist=self.alpha_dist)
            x = self.conv_layers[block][1](x)
            x = actfuns.activate(x, actfun=self.actfun,
                                 k=self.k, p=self.p, M=x.shape[1],
                                 layer_type='conv',
                                 permute_type=self.permute_type,
                                 shuffle_maps=self.shuffle_maps[(block * 2) + 1],
                                 alpha_primes=self.all_alpha_primes[(block * 2) + 1],
                                 alpha_dist=self.alpha_dist)
            x = self.pooling[block](x)

        x = x.view(x.size(0), -1)

        for i, fc in enumerate(self.linear_layers):
            x = fc(x)
            if i < len(self.linear_layers) - 1:
                x = actfuns.activate(x, actfun=self.actfun,
                                     k=self.k, p=self.p, M=x.shape[1],
                                     layer_type='linear',
                                     permute_type=self.permute_type,
                                     alpha_primes=self.all_alpha_primes[6 + i],
                                     alpha_dist=self.alpha_dist)

        return x
