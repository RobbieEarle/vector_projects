import torch
import torch.utils.data
import torch.nn as nn
import activation_functions as actfuns
import util
import math
import numpy as np


class MLP(nn.Module):

    def __init__(self, actfun,
                 input_dim=784,
                 output_dim=10,
                 k=2, p=1, g=1,
                 alpha_dist="per_cluster",
                 permute_type="shuffle",
                 reduce_actfuns=False,
                 num_params=600000):
        super(MLP, self).__init__()

        if permute_type == 'invert' and p % k != 0:
            p = k

        self.input_dim = input_dim
        self.actfun = actfun
        self.p, self.k, self.g = p, k, g
        self.permute_type = permute_type
        self.alpha_dist = alpha_dist
        self.shuffle_maps = []
        self.reduce_actfuns = reduce_actfuns
        self.iris = True if input_dim == 4 else False

        pk_ratio = util.get_pk_ratio(self.actfun, self.p, self.k, self.g)

        a = 1.25 * (pk_ratio / self.g)
        b = (1.25 * input_dim) + ((pk_ratio / self.g) * output_dim) + 2.25
        c = output_dim - num_params

        n2a = (-b + np.sqrt((b ** 2) - (4 * a * c))) / (2 * a)
        n2b = (-b - np.sqrt((b ** 2) - (4 * a * c))) / (2 * a)
        n2 = np.max([n2a, n2b])
        n1 = (num_params - (n2 * (((pk_ratio / self.g) * output_dim) + 1)) + output_dim) / (input_dim + 1 + (n2 * (
                pk_ratio / self.g)))

        pre_acts = [int(n1), int(n2)]

        post_acts = []
        for i, pre_act in enumerate(pre_acts):
            pre_acts[i] = self.k * self.g * int(pre_act / (self.k * self.g))
            if actfun == 'bin_partition_full':
                post_acts.append(int((pre_acts[i] * self.p) + (2 * math.floor((pre_acts[i] * self.p) / (3 * self.k)) * (
                        1 - self.k))))
            else:
                post_acts.append(int(pre_acts[i] * pk_ratio))

        if self.iris:
            pre_acts = [4, 4]
            post_acts = [int(4 * pk_ratio), int(4 * pk_ratio)]

        self.linear_layers = nn.ModuleDict()
        self.linear_layers['l1'] = nn.Linear(input_dim, pre_acts[0])
        self.linear_layers['l2'] = nn.ModuleList()
        for group in range(g):
            self.linear_layers['l2'].append(nn.Linear(int(post_acts[0] / g), int(pre_acts[1] / g)))
        self.linear_layers['l3'] = nn.Linear(post_acts[1], output_dim)

        if not self.iris:
            self.batch_norms = nn.ModuleDict({
                'l1': nn.BatchNorm1d(pre_acts[0]),
                'l2': nn.BatchNorm1d(pre_acts[1])
            })

        self.shuffle_maps = util.add_shuffle_map(self.shuffle_maps, pre_acts[0], self.p)
        self.shuffle_maps = util.add_shuffle_map(self.shuffle_maps, pre_acts[1], self.p)

        self.all_alpha_primes = nn.ParameterList()
        self.alpha_dist = alpha_dist
        if self.actfun == "combinact":
            self.num_combinact_actfuns = len(actfuns.get_combinact_actfuns(reduce_actfuns))
            if alpha_dist == "per_cluster":
                self.all_alpha_primes.append(nn.Parameter(torch.zeros(post_acts[0], self.num_combinact_actfuns)))
                self.all_alpha_primes.append(nn.Parameter(torch.zeros(post_acts[1], self.num_combinact_actfuns)))
            if alpha_dist == "per_perm":
                for layer in range(2):
                    self.all_alpha_primes.append(nn.Parameter(torch.zeros(self.p, self.num_combinact_actfuns)))


    def forward(self, x):

        x = x.reshape(x.size(0), self.input_dim)

        x = self.linear_layers['l1'](x)
        x = self.batch_norms['l1'](x) if not self.iris else x
        x = self.activate(x, 0)
        x = x.unsqueeze(0) if len(x.shape) == 1 else x

        all_outputs = None
        for group_idx, group_fc in enumerate(self.linear_layers['l2']):
            group_idx_start = group_idx * int(x.shape[1] / self.g)
            group_idx_end = (group_idx + 1) * int(x.shape[1] / self.g)
            curr_inputs = x[:, group_idx_start:group_idx_end]
            curr_outputs = group_fc(curr_inputs)
            if group_idx == 0:
                all_outputs = curr_outputs
            else:
                all_outputs = torch.cat((all_outputs, curr_outputs), dim=1)
        x = all_outputs
        x = self.batch_norms['l2'](x) if not self.iris else x
        x = self.activate(x, 1)
        x = x.unsqueeze(0) if len(x.shape) == 1 else x

        x = self.linear_layers['l3'](x)

        return x

    def activate(self, x, layer):
        if self.actfun == 'combinact':
            alpha_primes = self.all_alpha_primes[layer]
        else:
            alpha_primes = None
        return actfuns.activate(x, actfun=self.actfun,
                                k=self.k, p=self.p, M=x.shape[1],
                                layer_type='linear',
                                permute_type=self.permute_type,
                                shuffle_maps=self.shuffle_maps[layer],
                                alpha_primes=alpha_primes,
                                alpha_dist=self.alpha_dist,
                                reduce_actfuns=self.reduce_actfuns)