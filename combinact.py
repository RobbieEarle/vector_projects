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


def get_n_params(model):
    pp=0
    for p in list(model.parameters()):
        nn=1
        for s in list(p.size()):
            nn = nn*s
        pp += nn
    return pp


def seed_all(seed=None, only_current_gpu=False, mirror_gpus=False):
    r"""
    Initialises the random number generators for random, numpy, and both CPU and GPU(s)
    for torch.
    Arguments:
        seed (int, optional): seed value to use for the random number generators.
            If :attr:`seed` is ``None`` (default), seeds are picked at random using
            the methods built in to each RNG.
        only_current_gpu (bool, optional): indicates whether to only re-seed the current
            cuda device, or to seed all of them. Default is ``False``.
        mirror_gpus (bool, optional): indicates whether all cuda devices should receive
            the same seed, or different seeds. If :attr:`mirror_gpus` is ``False`` and
            :attr:`seed` is not ``None``, each device receives a different but
            deterministically determined seed. Default is ``False``.
    Note that we override the settings for the cudnn backend whenever this function is
    called. If :attr:`seed` is not ``None``, we set::
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    in order to ensure experimental results behave deterministically and are repeatible.
    However, enabling deterministic mode may result in an impact on performance. See
    `link`_ for more details. If :attr:`seed` is ``None``, we return the cudnn backend
    to its performance-optimised default settings of::
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True
    .. _link:
        https://pytorch.org/docs/stable/notes/randomness.html
    """
    # Note that random, np.random and torch's RNG all have different
    # implementations so they will produce different numbers even with
    # when they are seeded the same.

    # Seed Python's built-in random number generator
    random.seed(seed)
    # Seed numpy's random number generator
    np.random.seed(seed)

    def get_seed():
        '''
        On Python 3.2 and above, and when system sources of randomness are
        available, use `os.urandom` to make a new seed. Otherwise, use the
        current time.
        '''
        try:
            import os
            # Use system's source of entropy (on Linux, syscall `getrandom()`)
            s = int.from_bytes(os.urandom(4), byteorder="little")
        except AttributeError:
            from datetime import datetime
            # Get the current time in mircoseconds, and map to an integer
            # in the range [0, 2**32)
            s = int((datetime.utcnow() - datetime(1970, 1, 1)).total_seconds()
                    * 1000000) % 4294967296
        return s

    # Seed pytorch's random number generator on the CPU
    # torch doesn't support a None argument, so we have to source our own seed
    # with high entropy if none is given.
    s = seed if seed is not None else get_seed()
    torch.manual_seed(s)

    if seed is None:
        # Since seeds are random, we don't care about determinism and
        # will set the backend up for optimal performance
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True
    else:
        # Ensure cudNN is deterministic, so the results are consistent
        # for this seed
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # Seed pytorch's random number generator on the GPU(s)
    if only_current_gpu:
        # Only re-seed the current GPU
        if mirror_gpus:
            # ... re-seed with the same as the CPU seed
            torch.cuda.manual_seed(s)
        elif seed is None:
            # ... re-seed at random, however pytorch deems fit
            torch.cuda.seed()
        else:
            # ... re-seed with a deterministic seed based on, but
            # not equal to, the CPU seed
            torch.cuda.manual_seed((seed + 1) % 4294967296)
    elif mirror_gpus:
        # Seed multiple GPUs, each with the same seed
        torch.cuda.manual_seed_all(s)
    elif seed is None:
        # Seed multiple GPUs, all with unique seeds
        # ... a random seed for each GPU, however pytorch deems fit
        torch.cuda.seed_all()
    else:
        # Seed multiple GPUs, all with unique seeds
        # ... different deterministic seeds for each GPU
        # We assign the seeds in ascending order, and can't exceed the
        # random state's maximum value of 2**32 == 4294967296
        for device in range(torch.cuda.device_count()):
            with torch.cuda.device(device):
                torch.cuda.manual_seed((seed + 1 + device) % 4294967296)


def test_net_inputs(actfun, in_size, M1, M2, out_size, k1, k2, p1, p2, g2, g_out):
    if M2 % g2 != 0:
        return 'g2 must divide M2.  M2 = {}, g2 = {}'.format(M2, g2)
    if out_size % g_out != 0:
        return 'g_out must divide out_size.  out_size = {}, g_out = {}'.format(out_size, g_out)
    if (M1 * p1) % k1 != 0:
        return 'k1 must divide M1*p1.  k1 = {}, M1*p1 = {}'.format(k1, M1 * p1)
    if (M2 * p2) % k2 != 0:
        return 'k2 must divide M2*p2.  k2 = {}, M2*p2 = {}'.format(k2, M2 * p2)
    if (M2 / g2) % k2 != 0:
        return 'k2 must divide M2/g2.  k2 = {}, M2/g2 = {}'.format(k2, M2 / g2)
    if (M1 * p1 / k1) % g2 != 0:
        return 'g2 must divide M1*p1/k1.  M1*p1/k1 = {}, g2 = {}'.format(M1 * p1 / k1, g2)
    if (M2 * p2 / k2) % g_out != 0:
        return 'g_out must divide M2*p2/k2.  M2*p2/k2 = {}, g_out = {}'.format(M2 * p2 / k2, g_out)
    return None


class ScottNet(nn.Module):

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
                 g_out=2
                 ):
        super(ScottNet, self).__init__()

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
        self.out_size = out_size
        self.k1 = k1
        self.k2 = k2
        self.p1 = p1
        self.p2 = p2
        self.g2 = g2
        self.g_out = g_out

        # Round 1 Params
        self.fc1 = nn.Linear(in_size, M1)

        # Round 2 Params
        self.r2_fc_groups = nn.ModuleList([nn.Linear(int((M1*p1/k1)/g2), int(M2/g2)) for i in range(g2)])

        # Round 3 Params
        self.r3_fc_groups = nn.ModuleList([nn.Linear(int((M2*p2/k2)/g_out), int(out_size/g_out)) for i in range(g_out)])

        # Batchnorm for the pre-activations of the two hidden layers
        self.bn1 = nn.BatchNorm1d(M1)
        self.bn2 = nn.BatchNorm1d(int(M2 / g2))

    def permute(self, x, method, seed):
        if method == "roll":
            return torch.cat((x[:, seed:, 0], x[:, :seed, 0]), dim=1)

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

            # ------- First Round
            #   In: [batch_size, in_size]
            #   Out: [batch_size, M1 / k1, p1]
            x = self.activate(self.bn1(self.fc1(x)), self.M1, self.k1, self.p1)

            # Grouping: [batch_size, M1 / k1, p1] -> [batch_size, (M1 * p1 / k1) / g2, g2]
            x = torch.transpose(x, dim0=1, dim1=2).reshape(self.batch_size,
                                                           int((self.M1*self.p1/self.k1) / self.g2), self.g2)

            # ------- Second Round
            #   In: [batch_size, (M1 * p1 / k1) / g2, g2]
            #   Out: [batch_size, (M2 / k2) / g2, p2 * g2]
            outputs = torch.zeros((self.batch_size, int((self.M2 / self.g2) / self.k2), self.p2 * self.g2),
                                  device=x.device)
            for i, fc2 in enumerate(self.r2_fc_groups):
                outputs[:, :, i*self.p2:(i+1)*self.p2] = self.activate(
                    self.bn2(fc2(x[:, :, i])), int(self.M2 / self.g2), self.k2, self.p2)
            x = outputs

            # Grouping: [batch_size, (M2 / g2) / k2, p2 * g2] -> [batch_size, (M2 * p2 / k2) / g_out, g_out]
            x = torch.transpose(x, dim0=1, dim1=2).reshape(self.batch_size,
                                                           int((self.M2 * self.p2 / self.k2) / self.g_out), self.g_out)

            # ------- Third Round
            #   In: [batch_size, (M2 * p2 / k2) / g_out, g_out]
            #   Out: [batch_size, out_size / g_out, g_out]
            outputs = torch.zeros((self.batch_size, int(self.out_size / self.g_out), self.g_out),
                                  device=x.device)
            for i, fc3 in enumerate(self.r3_fc_groups):
                outputs[:, :, i] = fc3(x[:, :, i])

            # Grouping: [batch_size, out_size / g_out, g_out] -> [batch_size, out_size]
            x = torch.transpose(outputs, dim0=1, dim1=2).reshape((self.batch_size, self.out_size))

        return x

    def activate(self, x, M, k, p):
        clusters = math.floor(M / k)
        remainder = M % k
        x = x.view(self.batch_size, M, 1)

        # Duplicate and permute x
        for i in range(1, p):
            x = torch.cat((x[:,:,:i], self.permute(x, "roll", i).view(self.batch_size, M, 1)), dim=2)

        # Handle if k doesn't divide M evenly
        if remainder != 0:
            # Separate reminder elements from complete clusters
            y = x[:, M - remainder:, :]
            x = x[:, :M - remainder, :]
            # Apply activation function to remainder elements
            y = y.view(self.batch_size, 1, remainder, p)
            y = _ACTFUNS2D[self.actfun](y)
            y = y.view(y.shape[0], 1, p)

        # Apply activation function to complete clusters
        x = x.view(self.batch_size, clusters, k, p)
        x = _ACTFUNS2D[self.actfun](x)

        # Combine results from complete clusters with result from remainder elements if necessary
        if remainder != 0:
            x = torch.cat((x, y), dim=1)

        return x


def weights_init(m):
    irange = 0.005
    if type(m) == nn.Linear:
        m.weight.data.uniform_(-1 * irange, irange)
        m.bias.data.fill_(0)


def train_model(model, outfile_path, fieldnames, seed, train_loader, validation_loader, hyper_params):

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
    while epoch <= 2:

        start_time = time.time()
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

        # Outputting data to CSV at end of epoch
        with open(outfile_path, mode='a') as out_file:
            writer = csv.DictWriter(out_file, fieldnames=fieldnames, lineterminator='\n')
            writer.writerow({'seed': seed,
                             'epoch': epoch,
                             'actfun': model.actfun,
                             'train_loss': float(final_train_loss),
                             'val_loss': float(final_val_loss),
                             'acc': float(accuracy),
                             'time': (time.time() - start_time),
                             'M1': model.M1,
                             'M2': model.M2,
                             'k1': model.k1,
                             'k2': model.k2,
                             'p1': model.p1,
                             'p2': model.p2,
                             'g2': model.g2,
                             'g_out': model.g_out,
                             'n_params': get_n_params(model)
                             })

        epoch += 1


def run_experiment(actfun, seed, outfile_path):

    trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (1.0,))])
    mnist_train_full = datasets.MNIST(root='./data', train=True, download=True, transform=trans)
    train_set_indices = np.arange(0, 50000)
    validation_set_indices = np.arange(50000, 60000)

    mnist_train = torch.utils.data.Subset(mnist_train_full, train_set_indices)
    mnist_validation = torch.utils.data.Subset(mnist_train_full, validation_set_indices)
    batch_size = 100
    train_loader = torch.utils.data.DataLoader(dataset=mnist_train, batch_size=batch_size, shuffle=True, pin_memory=True)
    validation_loader = torch.utils.data.DataLoader(dataset=mnist_validation, batch_size=batch_size, shuffle=True, pin_memory=True)

    rng = np.random.RandomState(seed)
    while True:
        M1 = rng.randint(60, 120) * 2
        M2 = rng.randint(60, 120) * 2
        k1 = rng.randint(2, 11)
        k2 = rng.randint(2, 11)
        p1 = rng.randint(1, 11)
        p2 = rng.randint(1, 11)
        g2 = rng.randint(1, 5)
        g_out = rng.randint(1, 5)
        if test_net_inputs(actfun, 784, M1, M2, 10, k1, k2, p1, p2, g2, g_out) is None:
            break
    model = ScottNet(batch_size=100, actfun=actfun, in_size=784, out_size=10,
                     M1=M1, M2=M2, k1=k1, k2=k2, p1=p1,
                     p2=p2, g2=g2, g_out=g_out)
    if torch.cuda.is_available():
        model = model.cuda()

    print(
        "Model Params: M1 = {] | M2 = {} | k1 = {} | k2 = {} | p1 = {} | p2 = {} | g2 = {} | g_out = {} | params = {}"
            .format(M1, M2, k1, k2, p1, p2, g2, g_out, get_n_params(model)), flush=True
    )

    fieldnames = ['seed', 'epoch', 'actfun', 'train_loss', 'val_loss', 'acc', 'time',
                  'M1', 'M2', 'k1', 'k2', 'p1', 'p2', 'g2', 'g_out', 'n_params']
    with open(outfile_path, mode='w') as out_file:
        writer = csv.DictWriter(out_file, fieldnames=fieldnames, lineterminator='\n')
        writer.writeheader()

    hyper_params = {"l2": {"adam_beta_1": 0.760516,
                           "adam_beta_2": 0.999983,
                           "adam_eps": 1.7936 * 10 ** -8,
                           "adam_wd": 1.33755 * 10 ** -5,
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
    train_model(model, outfile_path, fieldnames, seed, train_loader, validation_loader, hyper_params[actfun])
    print()


if __name__ == '__main__':

    if len(sys.argv) == 1:
        actfun = "max"
        seed = 0
        outfile_path = str(datetime.date.today()) + "-combinact-" + str(actfun) + "-" + str(seed) + ".csv"

    else:
        seed_all(0)
        actfun = sys.argv[1]
        seed = int(sys.argv[2])
        outfile_path = sys.argv[3] + "/" + str(datetime.date.today()) + "-combinact-"\
                       + str(actfun) + "-" + str(seed) + ".csv"

    print("Activation Function: " + str(actfun))
    print("Save Path: " + str(outfile_path))

    run_experiment(actfun, seed, outfile_path)
