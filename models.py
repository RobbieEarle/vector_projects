import torch
import torch.utils.data
import torch.nn as nn
import activation_functions as actfuns
import util
import numpy as np
import time


class CombinactNN(nn.Module):

    hyper_params = {
        "relu": {"adam_beta_1": np.exp(-2.375018573261741),
                 "adam_beta_2": np.exp(-6.565065478550015),
                 "adam_eps": np.exp(-19.607731090387627),
                 "adam_wd": np.exp(-11.86635747404571),
                 "max_lr": np.exp(-5.7662952418075175),
                 "cycle_peak": 0.2935155263985412
                 },
        "cf_relu": {"adam_beta_1": np.exp(-4.44857338551192),
                    "adam_beta_2": np.exp(-4.669825410890087),
                    "adam_eps": np.exp(-17.69933166220988),
                    "adam_wd": np.exp(-12.283288733512373),
                    "max_lr": np.exp(-8.563504990329884),
                    "cycle_peak": 0.10393251332079881
                    },
        "multi_relu": {"adam_beta_1": np.exp(-2.859441513546877),
                       "adam_beta_2": np.exp(-5.617992566623951),
                       "adam_eps": np.exp(-20.559015044774018),
                       "adam_wd": np.exp(-12.693844976989661),
                       "max_lr": np.exp(-5.802816398828524),
                       "cycle_peak": 0.28499869111025217
                       },
        "combinact": {"adam_beta_1": np.exp(-2.6436039683427253),
                      "adam_beta_2": np.exp(-7.371516988658699),
                      "adam_eps": np.exp(-16.989022147994522),
                      "adam_wd": np.exp(-12.113778466374383),
                      "max_lr": np.exp(-8),
                      "cycle_peak": 0.4661308739740898
                      },
        "l2": {"adam_beta_1": np.exp(-2.244614412525641),
               "adam_beta_2": np.exp(-5.502197648895974),
               "adam_eps": np.exp(-16.919215725249092),
               "adam_wd": np.exp(-13.99956243808541),
               "max_lr": np.exp(-5.383090612225605),
               "cycle_peak": 0.35037784343793205
               },
        "abs": {"adam_beta_1": np.exp(-3.1576858739457845),
                "adam_beta_2": np.exp(-4.165206705873042),
                "adam_eps": np.exp(-20.430988799955056),
                "adam_wd": np.exp(-13.049933891070697),
                "max_lr": np.exp(-5.809683797646132),
                "cycle_peak": 0.34244342851740034
                },
        "cf_abs": {"adam_beta_1": np.exp(-5.453380890632929),
                   "adam_beta_2": np.exp(-5.879222236954101),
                   "adam_eps": np.exp(-18.303333640483068),
                   "adam_wd": np.exp(-15.152599023560422),
                   "max_lr": np.exp(-6.604045812173043),
                   "cycle_peak": 0.11189158130301018
                   },
        "l2_lae": {"adam_beta_1": np.exp(-2.4561852034212),
                   "adam_beta_2": np.exp(-5.176943480470942),
                   "adam_eps": np.exp(-16.032458209235187),
                   "adam_wd": np.exp(-12.860274699438266),
                   "max_lr": np.exp(-5.540947578537945),
                   "cycle_peak": 0.40750994546983904
                   },
        "max": {"adam_beta_1": np.exp(-2.2169207045481505),
                "adam_beta_2": np.exp(-7.793567052557596),
                "adam_eps": np.exp(-18.23187258333265),
                "adam_wd": np.exp(-12.867866026516422),
                "max_lr": np.exp(-5.416840501318637),
                "cycle_peak": 0.28254869607601146
                }

    }

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

    hyper_params = {
        "relu": {"adam_beta_1": np.exp(-2.5713946178339486),
                 "adam_beta_2": np.exp(-8.088852495066451),
                 "adam_eps": np.exp(-18.24053115491185),
                 "adam_wd": np.exp(-12.007877998144522),
                 "max_lr": np.exp(-7.277101799190481),
                 "cycle_peak": 0.364970594416471
                 },
        "multi_relu": {"adam_beta_1": np.exp(-2.3543718655934547),
                       "adam_beta_2": np.exp(-7.063659565937045),
                       "adam_eps": np.exp(-19.22492182089545),
                       "adam_wd": np.exp(-10.269718286116909),
                       "max_lr": np.exp(-6.8770611857136075),
                       "cycle_peak": 0.4253002234015769
                       },
        "combinact": {"adam_beta_1": np.exp(-2.016683834468364),
                      "adam_beta_2": np.exp(-7.820800773443709),
                      "adam_eps": np.exp(-18.01936303461807),
                      "adam_wd": np.exp(-14.443234599437305),
                      "max_lr": np.exp(-6.7810979033379875),
                      "cycle_peak": 0.5439417885046983
                      },
        "l2": {"adam_beta_1": np.exp(-1.5749000540594622),
               "adam_beta_2": np.exp(-3.6702433473885767),
               "adam_eps": np.exp(-17.788820155080888),
               "adam_wd": np.exp(-14.297423169143356),
               "max_lr": np.exp(-7.246379919517856),
               "cycle_peak": 0.4721781379107825
               },
        "abs": {"adam_beta_1": np.exp(-1.6509199183692422),
                "adam_beta_2": np.exp(-5.449016919866456),
                "adam_eps": np.exp(-18.97360098070963),
                "adam_wd": np.exp(-11.927993917764805),
                "max_lr": np.exp(-7.591007314708498),
                "cycle_peak": 0.48552168878517715
                },
        "l2_lae": {"adam_beta_1": np.exp(-1.511652530521991),
                   "adam_beta_2": np.exp(-5.10036591613782),
                   "adam_eps": np.exp(-20.158860548398614),
                   "adam_wd": np.exp(-11.630968574087534),
                   "max_lr": np.exp(-6.992522933149952),
                   "cycle_peak": 0.41503241211381126
                   },
        "max": {"adam_beta_1": np.exp(-2.3151753028565794),
                "adam_beta_2": np.exp(-4.660984944761118),
                "adam_eps": np.exp(-19.231174065933367),
                "adam_wd": np.exp(-8.028370292260313),
                "max_lr": np.exp(-6.720521846837062),
                "cycle_peak": 0.4677382752348381
                }

    }

    def __init__(self, actfun,
                 num_input_channels=3,
                 input_dim=32,
                 num_outputs=10,
                 k=2, p=1,
                 alpha_dist="per_cluster",
                 permute_type="shuffle",
                 pfact=1):
        super(CombinactCNN, self).__init__()

        self.actfun = actfun
        self.k, self.p = k, p
        if actfun == 'relu' or actfun == 'abs':
            self.k = 1
        self.permute_type = permute_type
        self.alpha_dist = alpha_dist
        self.shuffle_maps = []

        pk_ratio = self.p / self.k

        sizes = [int(32 * pfact),
                 int(64 * pfact),
                 int(128 * pfact),
                 int(256 * pfact),
                 int(512 * pfact),
                 int(1024 * pfact)]

        for i, size in enumerate(sizes):
            sizes[i] = k * int(size / self.k)

        self.conv_layers = nn.ModuleList([
            nn.ModuleList([nn.Conv2d(in_channels=num_input_channels, out_channels=sizes[0], kernel_size=3, padding=1),
                           nn.Conv2d(in_channels=int(sizes[0] * pk_ratio), out_channels=sizes[1], kernel_size=3, padding=1)]),
            nn.ModuleList([nn.Conv2d(in_channels=int(sizes[1] * pk_ratio), out_channels=sizes[2], kernel_size=3, padding=1),
                           nn.Conv2d(in_channels=int(sizes[2] * pk_ratio), out_channels=sizes[2], kernel_size=3, padding=1)]),
            nn.ModuleList([nn.Conv2d(in_channels=int(sizes[2] * pk_ratio), out_channels=sizes[3], kernel_size=3, padding=1),
                           nn.Conv2d(in_channels=int(sizes[3] * pk_ratio), out_channels=sizes[3], kernel_size=3, padding=1)])
        ])

        self.shuffle_maps = util.add_shuffle_map(self.shuffle_maps, sizes[0], self.p)
        self.shuffle_maps = util.add_shuffle_map(self.shuffle_maps, sizes[1], self.p)
        self.shuffle_maps = util.add_shuffle_map(self.shuffle_maps, sizes[2], self.p)
        self.shuffle_maps = util.add_shuffle_map(self.shuffle_maps, sizes[2], self.p)
        self.shuffle_maps = util.add_shuffle_map(self.shuffle_maps, sizes[3], self.p)
        self.shuffle_maps = util.add_shuffle_map(self.shuffle_maps, sizes[3], self.p)

        self.batch_norms = nn.ModuleList([
            nn.BatchNorm2d(sizes[0]),
            nn.BatchNorm2d(sizes[2]),
            nn.BatchNorm2d(sizes[3]),
        ])

        self.pooling = nn.ModuleList([
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.MaxPool2d(kernel_size=2, stride=2)
        ])

        self.linear_layers = nn.ModuleList([
            nn.Linear(int(sizes[3] * pk_ratio) * (int(input_dim / 8) ** 2), sizes[5]),
            nn.Linear(int(sizes[5] * pk_ratio), sizes[4]),
            nn.Linear(int(sizes[4] * pk_ratio), num_outputs)
        ])

        self.shuffle_maps = util.add_shuffle_map(self.shuffle_maps, sizes[5], self.p)
        self.shuffle_maps = util.add_shuffle_map(self.shuffle_maps, sizes[4], self.p)

        self.all_alpha_primes = nn.ParameterList()  # List of our trainable alpha prime values
        self.alpha_dist = alpha_dist  # Reference to chosen alpha distribution
        if self.actfun == "combinact":
            self.num_combinact_actfuns = len(actfuns.get_combinact_actfuns())  # Number of actfuns used by combinact
            if alpha_dist == "per_cluster":
                self.all_alpha_primes.append(nn.Parameter(torch.zeros(int(sizes[0] * self.p / self.k), self.num_combinact_actfuns)))
                self.all_alpha_primes.append(nn.Parameter(torch.zeros(int(sizes[1] * self.p / self.k), self.num_combinact_actfuns)))
                self.all_alpha_primes.append(nn.Parameter(torch.zeros(int(sizes[2] * self.p / self.k), self.num_combinact_actfuns)))
                self.all_alpha_primes.append(nn.Parameter(torch.zeros(int(sizes[2] * self.p / self.k), self.num_combinact_actfuns)))
                self.all_alpha_primes.append(nn.Parameter(torch.zeros(int(sizes[3] * self.p / self.k), self.num_combinact_actfuns)))
                self.all_alpha_primes.append(nn.Parameter(torch.zeros(int(sizes[3] * self.p / self.k), self.num_combinact_actfuns)))
                self.all_alpha_primes.append(nn.Parameter(torch.zeros(int(sizes[5] * self.p / self.k), self.num_combinact_actfuns)))
                self.all_alpha_primes.append(nn.Parameter(torch.zeros(int(sizes[4] * self.p / self.k), self.num_combinact_actfuns)))
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
                                 alpha_dist=self.alpha_dist)
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
                                 alpha_dist=self.alpha_dist)
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
                                     alpha_primes=alpha_primes,
                                     alpha_dist=self.alpha_dist)

        return x
