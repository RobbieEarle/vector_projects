import torch
import torch.utils.data
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.optim.lr_scheduler import CyclicLR

import os
import sys
import numpy as np
import math
import random
import datetime
import csv
import time


def signed_l3(z):
    x3 = z[:, :, 0].pow(3)
    y3 = z[:, :, 1].pow(3)
    out_val = (x3 + y3).tanh() * (x3 + y3).abs().pow(1 / 3)
    return out_val


_ln2 = 0.6931471805599453
_ACTFUNS2D = {
    'max':
        lambda z: torch.max(z, dim=2).values,
    'min':
        lambda z: torch.min(z, dim=2).values,
    'signed_geomean':
        lambda z: z[:, :, 0].sign() * z[:, :, 1].sign() * (z[:, :, 0] * z[:, :, 1]).abs_().sqrt_(),
    'swish2':
        lambda z: z[:, :, 0] * torch.sigmoid(z[:, :, 1]),
    'l2':
        lambda z: (z[:, :, 0].pow(2) + z[:, :, 1].pow(2)).sqrt_(),
    'l3-signed':
        lambda z: signed_l3(z),
    'linf':
        lambda z: torch.max(z[:, :, 0].abs(), z[:, :, 1].abs()),
    'lse-approx':
        lambda z: torch.max(z[:, :, 0], z[:, :, 1]) + torch.max(torch.tensor(0., device=z.device), _ln2 - 0.305 * (z[:, :, 0] - z[:, :, 1]).abs_()),
    'zclse-approx':
        lambda z: torch.max(z[:, :, 0], z[:, :, 1]) + torch.max(torch.tensor(-_ln2, device=z.device), -0.305 * (z[:, :, 0] - z[:, :, 1]).abs_()),
    'nlsen-approx':
        lambda z: -torch.max(-z[:, :, 0], -z[:, :, 1]) - torch.max(torch.tensor(0., device=z.device), _ln2 - 0.305 * (z[:, :, 0] - z[:, :, 1]).abs_()),
    'zcnlsen-approx':
        lambda z: -torch.max(-z[:, :, 0], -z[:, :, 1]) - torch.max(torch.tensor(-_ln2, device=z.device), -0.305 * (z[:, :, 0] - z[:, :, 1]).abs_())
}


def test_net_inputs(actfun, in_size, M1, M2, out_size, k1, k2, p1, p2, g2, g_out):
    if M1 * p1 % k1 != 0:
        return 'k1 must divide M1*p1.  k1 = {}, M1*p1 = {}'.format(k1, M1 * p1)
    if M2 % g2 != 0:
        return 'g2 must divide M2.  M2 = {}, g2 = {}'.format(M2, g2)
    if (M1 * p1 / k1) % g2 != 0:
        return 'g2 must divide M1*p1/k1.  M1*p1/k1 = {}, g2 = {}'.format(M1 * p1 / k1, g2)
    return None


class Net(nn.Module):

    def __init__(self,
                 actfun='max',
                 batch_size=100,
                 in_size=784,
                 M1=240,
                 M2=240,
                 out_size=10,
                 k1=2,
                 k2=2,
                 p1=2,
                 p2=2,
                 g2=2,
                 g_out=1
                 ):
        super(Net, self).__init__()

        error = test_net_inputs(actfun, in_size, M1, M2, out_size, k1, k2, p1, p2, g2, g_out)
        if error is not None:
            raise ValueError(error)

        if actfun == 'relu':
            k1 = 1
            k2 = 1
            p1 = 1
            p2 = 1
            g2 = 1
            g_out = 1

        self.actfun = actfun
        self.batch_size = batch_size
        self.M1 = M1
        self.M2 = M2
        self.k1 = k1
        self.k2 = k2
        self.p1 = p1
        self.p2 = p2
        self.g2 = g2
        self.g_out = g_out

        # Round 1 Params
        self.fc1 = nn.Linear(in_size, M1)

        # Round 2 Params
        self.r2_fc_groups = []
        for i in range(g2):
            self.r2_fc_groups.append(nn.Linear(int((M1*p1/k1)/g2), int(M2/g2)))

        # Round 3 Params
        self.r3_fc_groups = []
        for i in range(g_out):
            self.r3_fc_groups.append(nn.Linear(int((M2*p2/k2)/g_out), int(out_size/g_out)))

        # Batchnorm for the pre-activations of the two hidden layers
        self.bn1 = nn.BatchNorm1d(M1)
        self.bn2 = nn.BatchNorm1d(int(M2 / g2))

    def forward(self, x):

        # x is initially torch.Size([100, 1, 28, 28]), this step converts to torch.Size([100, 784])
        x = x.view(self.batch_size, -1)

        # Handles relu activation functions (used as control in experiment)
        if self.actfun == 'relu':
            x = F.relu(self.bn1(self.fc1(x)))
            x = F.relu(self.bn2(self.r2_fc_groups[0](x)))
            x = self.r3_fc_groups[0](x)

        # Handling all other activation functions
        else:
            print(x.shape)
            x = self.activate(self.bn1(self.fc1(x)), self.M1, self.k1, self.p1)
            print(x.shape)

            x = x.view((self.batch_size, int((self.M1*self.p1/self.k1)/self.g2), self.g2))
            print(x.shape)
            print(x[0, :10, :])

            for i, fc2 in enumerate(self.r2_fc_groups):
                print()
                print(x[:, :, i].shape)
                print(x[0, :10, i])
                x_i = self.activate(self.bn2(fc2(x[:, :, i])), int(self.M2 / self.g2), self.k2, self.p2)
                print(x_i.shape)

            print()
            print("sdfds" + 234)

            x = self.activate(self.bn2(self.fc2(x)), self.M2, self.k2, self.p2)
            x = self.r3_params[0](x)

        return x

    def activate(self, x, M, k, p):
        clusters = math.floor(M / k)
        remainder = M % k

        x = x.view(self.batch_size, M, 1)

        for i in range(1, p):
            x = torch.cat((x[:,:,:i], torch.cat((x[:, i:, 0], x[:, :i, 0]), dim=1).view(self.batch_size, M, 1)), dim=2)
            x = x.view(self.batch_size, M, -1)

        if remainder != 0:
            y = x[:, M - remainder:, :]
            x = x[:, :M - remainder, :]
            y = y.view(self.batch_size, 1, remainder, p)
            y = _ACTFUNS2D[self.actfun](y)
            y = y.view(y.shape[0], 1, p)

        x = x.view(self.batch_size, clusters, k, p)
        x = _ACTFUNS2D[self.actfun](x)

        if remainder != 0:
            x = torch.cat((x, y), dim=1)

        # Note that at the moment if we flatten x the outputs from the permutations will be interleaved. ie. if p = 4
        # the first 4 flattened elements will be the first element from each of the 4 permutations, NOT the first 4
        # elements from the first permutation. Might need to fix later (not permutation invariant?)
        return x


def weights_init(m):
    irange = 0.005
    if type(m) == nn.Linear:
        m.weight.data.uniform_(-1 * irange, irange)
        m.bias.data.fill_(0)


def train_model(model, train_loader, validation_loader, hyper_params):

    # ---- Initialization
    model.apply(weights_init)
    optimizer = optim.Adam(model.parameters(),
                           lr=10**-8,
                           betas=(hyper_params['adam_beta_1'], hyper_params['adam_beta_2']),
                           eps=hyper_params['adam_eps'],
                           weight_decay=hyper_params['adam_wd']
                           )
    criterion = nn.CrossEntropyLoss()
    scheduler = CyclicLR(optimizer,
                         base_lr=10**-8,
                         max_lr=hyper_params['max_lr'],
                         step_size_up=int(hyper_params['cycle_peak'] * 5000),
                         step_size_down=int((1-hyper_params['cycle_peak']) * 5000),
                         cycle_momentum=False
                         )

    # ---- Start Training
    epoch = 1
    while epoch <= 10:

        final_train_loss = 0
        # ---- Training
        for batch_idx, (x, targetx) in enumerate(train_loader):
            model.train()
            if torch.cuda.is_available():
                x, targetx = x.cuda(non_blocking=True), targetx.cuda(non_blocking=True)
            optimizer.zero_grad()
            out = model(x)
            train_loss = criterion(out, targetx)
            train_loss.backward()
            optimizer.step()
            scheduler.step()
            final_train_loss = train_loss

        # ---- Testing
        num_correct = 0
        num_total = 0
        final_val_loss = 0
        model.eval()
        with torch.no_grad():
            for batch_idx2, (y, targety) in enumerate(validation_loader):
                if torch.cuda.is_available():
                    y, targety = y.cuda(non_blocking=True), targety.cuda(non_blocking=True)
                out = model(y)
                val_loss = criterion(out, targety)
                final_val_loss = val_loss
                _, prediction = torch.max(out.data, 1)
                num_correct += torch.sum(prediction == targety.data)
                num_total += len(prediction)
        accuracy = num_correct * 1.0 / num_total

        # Logging test results
        print(
            "    ({}) Epoch {}: train_loss = {:1.6f}  |  val_loss = {:1.6f}  |  accuracy = {:1.6f}"
                .format(model.actfun, epoch, final_train_loss, final_val_loss, accuracy), flush=True
        )

        epoch += 1


def run_experiment():

    trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (1.0,))])
    mnist_train_full = datasets.MNIST(root='./data', train=True, download=True, transform=trans)
    train_set_indices = np.arange(0, 50000)
    validation_set_indices = np.arange(50000, 60000)

    mnist_train = torch.utils.data.Subset(mnist_train_full, train_set_indices)
    mnist_validation = torch.utils.data.Subset(mnist_train_full, validation_set_indices)
    batch_size = 100
    train_loader = torch.utils.data.DataLoader(dataset=mnist_train, batch_size=batch_size, shuffle=True, pin_memory=True)
    validation_loader = torch.utils.data.DataLoader(dataset=mnist_validation, batch_size=batch_size, shuffle=True, pin_memory=True)

    model = Net(batch_size=batch_size)

    hyper_params = {"l2": {"adam_beta_1": 0.760516,
                           "adam_beta_2": 0.999983,
                           "adam_eps": 1.7936 * 10 **-8,
                           "adam_wd": 1.33755 * 10 **-5,
                           "max_lr": 0.0122491,
                           "cycle_peak": 0.234177
                           },
                    "linf": {"adam_beta_1": 0.193453,
                             "adam_beta_2": 0.999734,
                             "adam_eps": 5.78422 * 10 ** -9,
                             "adam_wd": 1.43599 * 10 ** -5,
                             "max_lr": 0.0122594,
                             "cycle_peak": 0.428421
                             },
                    "max": {"adam_beta_1": 0.942421,
                            "adam_beta_2": 0.999505,
                            "adam_eps": 2.81391 * 10 ** -9,
                            "adam_wd": 8.57978 * 10 ** -6,
                            "max_lr": 0.0245333,
                            "cycle_peak": 0.431589
                            },
                    "relu": {"adam_beta_1": 0.892104,
                             "adam_beta_2": 0.999429,
                             "adam_eps": 5.89807 * 10 ** -8,
                             "adam_wd": 3.62133 * 10 ** -6,
                             "max_lr": 0.0134033,
                             "cycle_peak": 0.310841
                             },
                    "signed_geomean": {"adam_beta_1": 0.965103,
                                       "adam_beta_2": 0.99997,
                                       "adam_eps": 9.15089 * 10 ** -8,
                                       "adam_wd": 6.99736 * 10 ** -5,
                                       "max_lr": 0.0077076,
                                       "cycle_peak": 0.36065
                                       },
                    "swish2": {"adam_beta_1": 0.766942,
                               "adam_beta_2": 0.999799,
                               "adam_eps": 3.43514 * 10 ** -9,
                               "adam_wd": 2.46361 * 10 ** -6,
                               "max_lr": 0.0155614,
                               "cycle_peak": 0.417112
                               },
                    "zclse-approx": {"adam_beta_1": 0.929379,
                                     "adam_beta_2": 0.999822,
                                     "adam_eps": 6.87644 * 10 ** -9,
                                     "adam_wd": 1.11525 * 10 ** -5,
                                     "max_lr": 0.0209105,
                                     "cycle_peak": 0.425568
                                     },
                    }

    print("Running...")
    train_model(model, train_loader, validation_loader, hyper_params['l2'])
    print()


if __name__ == '__main__':

    run_experiment()
