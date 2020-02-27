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
        lambda z: torch.max(z[:, :, 0], z[:, :, 1]) + torch.max(torch.tensor(0.).cuda(), _ln2 - 0.305 * (z[:, :, 0] - z[:, :, 1]).abs_()) if torch.cuda.is_available()
        else torch.max(z[:, :, 0], z[:, :, 1]) + torch.max(torch.tensor(0.), _ln2 - 0.305 * (z[:, :, 0] - z[:, :, 1]).abs_()),
    'zclse-approx':
        lambda z: torch.max(z[:, :, 0], z[:, :, 1]) + torch.max(torch.tensor(-_ln2).cuda(), -0.305 * (z[:, :, 0] - z[:, :, 1]).abs_()) if torch.cuda.is_available()
        else torch.max(z[:, :, 0], z[:, :, 1]) + torch.max(torch.tensor(-_ln2), -0.305 * (z[:, :, 0] - z[:, :, 1]).abs_()),
    'nlsen-approx':
        lambda z: -torch.max(-z[:, :, 0], -z[:, :, 1]) - torch.max(torch.tensor(0.).cuda(), _ln2 - 0.305 * (z[:, :, 0] - z[:, :, 1]).abs_()) if torch.cuda.is_available()
        else -torch.max(-z[:, :, 0], -z[:, :, 1]) - torch.max(torch.tensor(0.), _ln2 - 0.305 * (z[:, :, 0] - z[:, :, 1]).abs_()),
    'zcnlsen-approx':
        lambda z: -torch.max(-z[:, :, 0], -z[:, :, 1]) - torch.max(torch.tensor(-_ln2).cuda(), -0.305 * (z[:, :, 0] - z[:, :, 1]).abs_()) if torch.cuda.is_available()
        else -torch.max(-z[:, :, 0], -z[:, :, 1]) - torch.max(torch.tensor(-_ln2), -0.305 * (z[:, :, 0] - z[:, :, 1]).abs_())
}


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


class Net(nn.Module):

    def __init__(self,
                 hidden_u1=240,
                 hidden_u2=240,
                 k=2,
                 actfun=''
                 ):
        super(Net, self).__init__()

        self.k = k
        self.actfun = actfun
        postact_hidden_u1 = hidden_u1
        postact_hidden_u2 = hidden_u2
        if self.actfun != 'relu':
            postact_hidden_u1 = int(math.ceil(hidden_u1 / k))
            postact_hidden_u2 = int(math.ceil(hidden_u2 / k))

        self.fc1 = nn.Linear(784, hidden_u1)
        self.fc2 = nn.Linear(postact_hidden_u1, hidden_u2)
        self.fc3 = nn.Linear(postact_hidden_u2, 10)
        self.bn1 = nn.BatchNorm1d(hidden_u1)
        self.bn2 = nn.BatchNorm1d(hidden_u2)

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        if self.actfun == 'relu':
            x = F.relu(self.bn1(self.fc1(x)))
            x = F.relu(self.bn2(self.fc2(x)))
        else:
            x = self.activate(self.bn1(self.fc1(x)))
            x = self.activate(self.bn2(self.fc2(x)))
        x = self.fc3(x)

        return x

    def activate(self, x):
        batch_size = x.shape[0]
        num_hidden_nodes = x.shape[1]
        num_groups = math.floor(num_hidden_nodes / self.k)
        remainder = num_hidden_nodes % self.k

        if remainder != 0:
            y = x[:, num_hidden_nodes - remainder:]
            x = x[:, :num_hidden_nodes - remainder]
            y = y.view(batch_size, 1, remainder)
            y = _ACTFUNS2D[self.actfun](y)
            y = y.view(y.shape[0], 1)

        x = x.view(batch_size, num_groups, self.k)
        x = _ACTFUNS2D[self.actfun](x)

        if remainder != 0:
            x = torch.cat((x, y), dim=1)

        return x


def weights_init(m):
    irange = 0.005
    if type(m) == nn.Linear:
        m.weight.data.uniform_(-1 * irange, irange)
        m.bias.data.fill_(0)


def train_model(model,
                outfile_path,
                fieldnames,
                train_loader,
                validation_loader,
                seed=0,
                adam_beta_1=0.9,
                adam_beta_2=0.99,
                adam_eps=10 ** (-8),
                adam_wd=10 ** (-3),
                base_lr=0.1,
                max_lr=1,
                cycle_size=500):

    # ---- Initialization
    model.apply(weights_init)
    optimizer = optim.Adam(model.parameters(),
                           lr=base_lr,
                           betas=(adam_beta_1, adam_beta_2),
                           eps=adam_eps,
                           weight_decay=adam_wd
                           )
    criterion = nn.CrossEntropyLoss()
    scheduler = CyclicLR(optimizer,
                         base_lr=base_lr,
                         max_lr=max_lr,
                         step_size_up=int(cycle_size / 2),
                         cycle_momentum=False
                         )

    # ---- Start Training
    epoch = 1
    while epoch <= 10:

        epoch_best_train_loss = -1
        epoch_best_val_loss = -1
        epoch_best_acc = 0

        start_time = time.time()
        # ---- Training
        for batch_idx, (x, targetx) in enumerate(train_loader):
            model.train()
            if torch.cuda.is_available():
                x, targetx = x.cuda(), targetx.cuda()
            optimizer.zero_grad()
            out = model(x)
            train_loss = criterion(out, targetx)
            if train_loss < epoch_best_train_loss or epoch_best_train_loss == -1:
                epoch_best_train_loss = train_loss
            train_loss.backward()
            optimizer.step()
            scheduler.step()

            # ---- Testing
            if batch_idx % 100 == 0 or (epoch == 10 and batch_idx == 499):
                num_correct = 0
                num_total = 0
                total_val_loss = 0
                total_val_batch_size = 0
                model.eval()
                with torch.no_grad():
                    for batch_idx2, (y, targety) in enumerate(validation_loader):
                        if torch.cuda.is_available():
                            y, targety = y.cuda(), targety.cuda()
                        out = model(y)
                        val_loss = criterion(out, targety)
                        if val_loss < epoch_best_val_loss or epoch_best_val_loss == -1:
                            epoch_best_val_loss = val_loss
                        total_val_loss += val_loss
                        total_val_batch_size += 1
                        _, prediction = torch.max(out.data, 1)
                        num_correct += torch.sum(prediction == targety.data)
                        num_total += len(prediction)
                accuracy = num_correct * 1.0 / num_total
                if accuracy > epoch_best_acc:
                    epoch_best_acc = accuracy
                avg_val_loss = total_val_loss / total_val_batch_size

                print(
                    "    ({}) Epoch {}, Batch {}  :   train_loss = {:1.6f}  |  test_loss = {:1.6f}  |  accuracy = {:1.6f}"
                        .format(model.actfun, epoch, batch_idx, train_loss, avg_val_loss, accuracy)
                )
        with open(outfile_path, mode='a') as out_file:
            writer = csv.DictWriter(out_file, fieldnames=fieldnames, lineterminator='\n')
            writer.writerow({'seed':seed,
                             'epoch': epoch,
                             'actfun': model.actfun,
                             'train_loss': float(epoch_best_train_loss),
                             'val_loss': float(epoch_best_val_loss),
                             'top_accuracy': float(epoch_best_acc),
                             'time': (time.time() - start_time),
                             'adam_beta_1': adam_beta_1,
                             'adam_beta_2': adam_beta_2,
                             'adam_eps': adam_eps,
                             'adam_wd': adam_wd,
                             'base_lr': base_lr,
                             'max_lr': max_lr
                             })

        epoch += 1


def run_experiment(iterations, outfile_path):

    trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (1.0,))])
    mnist_train_full = datasets.MNIST(root='./data', train=True, download=True, transform=trans)
    train_set_indices = np.arange(0, 50000)
    validation_set_indices = np.arange(50000, 60000)

    fieldnames = ['seed', 'epoch', 'actfun', 'train_loss', 'val_loss', 'top_accuracy', 'time',
                  'adam_beta_1', 'adam_beta_2', 'adam_eps', 'adam_wd', 'base_lr', 'max_lr']
    if not os.path.exists(outfile_path):
        with open(outfile_path, mode='w') as out_file:
            writer = csv.DictWriter(out_file, fieldnames=fieldnames, lineterminator='\n')
            writer.writeheader()

    for i in range(iterations):

        seed = i
        seed_all(seed)

        mnist_train = torch.utils.data.Subset(mnist_train_full, train_set_indices)
        mnist_validation = torch.utils.data.Subset(mnist_train_full, validation_set_indices)
        batch_size = 100
        train_loader = torch.utils.data.DataLoader(dataset=mnist_train, batch_size=batch_size, shuffle=True)
        validation_loader = torch.utils.data.DataLoader(dataset=mnist_validation, batch_size=batch_size, shuffle=True)

        models = [
            Net(actfun='relu'),
            Net(actfun='max'),
            Net(actfun='min'),
            Net(actfun='signed_geomean'),
            Net(actfun='swish2'),
            Net(actfun='l2'),
            # Net(actfun='l3-signed')
            Net(actfun='linf'),
            Net(actfun='lse-approx'),
            Net(actfun='zclse-approx'),
            Net(actfun='nlsen-approx'),
            Net(actfun='zcnlsen-approx')
        ]

        adam_beta_1 = 1 - 10**np.random.uniform(-2, 0)
        adam_beta_2 = 1 - 10**np.random.uniform(-3, -2)
        adam_eps = 10**np.random.uniform(-8, -7)
        adam_wd = 10**np.random.uniform(-6, -3)
        base_lr = 10**(-8)
        max_lr = 10**np.random.uniform(-4, 0)

        print("-----> Iteration " + str(i))
        print("seed=" + str(seed))
        print("adam_beta_1=" + str(adam_beta_1))
        print("adam_beta_2=" + str(adam_beta_2))
        print("adam_eps=" + str(adam_eps))
        print("adam_wd=" + str(adam_wd))
        print("base_lr=" + str(base_lr))
        print("max_lr=" + str(max_lr))
        print()

        for model in models:
            print("  --> Iteration : " + str(i) + ", Activation " + str(model.actfun))
            if torch.cuda.is_available():
                model.cuda()
            train_model(model,
                        outfile_path,
                        fieldnames,
                        train_loader,
                        validation_loader,
                        seed=seed,
                        adam_beta_1=adam_beta_1,
                        adam_beta_2=adam_beta_2,
                        adam_eps=adam_eps,
                        adam_wd=adam_wd,
                        base_lr=base_lr,
                        max_lr=max_lr,
                        cycle_size=500)
            print()


if __name__ == '__main__':
    print(sys.argv[1])
    outfile_path = sys.argv[1] + "/" + str(datetime.date.today()) + "combinact_rand_search.csv"
    run_experiment(1000, outfile_path)
