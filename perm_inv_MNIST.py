import torch
import torch.utils.data
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms

import argparse
import os
import numpy as np
import math
import random
import datetime
import csv
import time


# -------------------- Helper Functions

def get_n_params(model):
    """
    :param model: Pytorch network model
    :return: Number of parameters in the model
    """
    pp=0
    for p in list(model.parameters()):
        nn=1
        for s in list(p.size()):
            nn = nn*s
        pp += nn
    return pp


def weights_init(m):
    """
    Randomly initialize weights for model. Always uses seed 0 so weights initialize to same value across experiments
    :param m: model
    :return:
    """
    irange = 0.005
    if type(m) == nn.Linear:
        m.weight.data.uniform_(-1 * irange, irange)
        m.bias.data.fill_(0)


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


def print_exp_settings(seed, outfile_path, net_struct, curr_model):
    print(
        "\n===================================================================\n\n"
        "Seed: {} \n"
        "Outfile Path: {} \n"
        "Number of Layers: {} \n"
        "Network Structure: \n"
        "{} \n"
        "Model Type: {} \n\n"
            .format(seed, outfile_path, net_struct['num_layers'], net_struct, curr_model), flush=True
    )


# -------------------- Network Module

class PermInvMnistNet(nn.Module):
    def __init__(self, net_struct, actfun):
        super(PermInvMnistNet, self).__init__()

        self.net_struct = net_struct
        self.actfun = actfun
        self.linear_layers = []
        self.num_inputs = 784
        self.num_outputs = 10
        self.num_hidden_layers = net_struct['num_layers']

        num_next_layer_inputs = self.num_inputs
        for layer in range(self.num_hidden_layers):
            self.linear_layers.append(nn.Linear(num_next_layer_inputs, net_struct['M'][layer]))
            num_next_layer_inputs = net_struct['M'][layer]
        self.linear_layers.append(nn.Linear(num_next_layer_inputs, self.num_outputs))

        self.linear_layers = nn.ModuleList(self.linear_layers)

    def activation(self, x, actfun):
        if actfun == 'relu':
            return F.relu(x)
        elif actfun == 'abs':
            return torch.abs_(x)
        else:
            raise ValueError("Activation function \"" + str(actfun) + "\" not recognized")

    def forward(self, x):

        # x is initially torch.Size([100, 1, 28, 28]), this step converts to torch.Size([100, 784])
        batch_size = x.shape[0]
        x = x.view(batch_size, -1)

        for layer in range(self.num_hidden_layers):
            x = self.linear_layers[layer](x)
            x = self.activation(x, self.actfun)

        x = self.linear_layers[-1](x)

        return x


# -------------------- Setting Up & Running Training Function

def train_model(actfun,
                net_struct,
                outfile_path,
                fieldnames,
                seed,
                train_loader,
                validation_loader,
                sample_size,
                device):
    """
    Runs training session for a given randomized model
    :param actfun: what activation type is being used by model
    :param net_struct: structure of our neural network
    :param outfile_path: path to save outputs from training session
    :param fieldnames: column names for output file
    :param seed: seed for randomization
    :param train_loader: training data loader
    :param validation_loader: validation data loader
    :param sample_size: number of training samples used in this experiment
    :param device: reference to CUDA device for GPU support
    :return:
    """

    # ---- Initialization
    model = PermInvMnistNet(net_struct=net_struct, actfun=actfun).to(device)
    model.apply(weights_init)

    for param in model.parameters():
        if len(param.data.shape) > 1:
            print(param.data.shape, flush=True)
            print(param.data[0, :10], flush=True)
    print()

    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)
    criterion = nn.CrossEntropyLoss()

    # ---- Start Training
    epoch = 1
    while epoch <= 10:

        start_time = time.time()
        final_train_loss = 0
        # ---- Training
        model.train()
        for batch_idx, (x, targetx) in enumerate(train_loader):
            x, targetx = x.to(device), targetx.to(device)
            optimizer.zero_grad()
            output = model(x)
            train_loss = criterion(output, targetx)
            train_loss.backward()
            optimizer.step()
            final_train_loss = train_loss

        # ---- Testing
        num_correct = 0
        num_total = 0
        final_val_loss = 0
        model.eval()
        with torch.no_grad():
            for batch_idx2, (y, targety) in enumerate(validation_loader):
                y, targety = y.to(device), targety.to(device)
                output = model(y)
                val_loss = criterion(output, targety)
                final_val_loss = val_loss
                _, prediction = torch.max(output.data, 1)
                num_correct += torch.sum(prediction == targety.data)
                num_total += len(prediction)
        accuracy = num_correct * 1.0 / num_total

        # Logging test results
        print(
            "    Epoch {}: train_loss = {:1.6f}  |  val_loss = {:1.6f}  |  accuracy = {:1.6f}  |  time = {}"
                .format(epoch, final_train_loss, final_val_loss, accuracy, (time.time() - start_time)), flush=True
        )

        # Outputting data to CSV at end of epoch
        with open(outfile_path, mode='a') as out_file:
            writer = csv.DictWriter(out_file, fieldnames=fieldnames, lineterminator='\n')
            writer.writerow({'seed': seed,
                             'epoch': epoch,
                             'train_loss': float(final_train_loss),
                             'val_loss': float(final_val_loss),
                             'acc': float(accuracy),
                             'time': (time.time() - start_time),
                             'net_struct': model.net_struct,
                             'model_type': model.actfun,
                             'num_layers': net_struct['num_layers'],
                             'sample_size': sample_size
                             })

        epoch += 1


def setup_experiment(seed, outfile_path, actfun):
    """
    Retrieves training / validation data, randomizes network structure and activation functions, creates model,
    creates new output file, sets hyperparameters for optimizer and scheduler during training, initializes training
    :param seed: seed for parameter randomization
    :param outfile_path: path to save outputs from experiment
    :param curr_model: model architecture
    :return:
    """

    # ---- Create new output file
    fieldnames = ['seed', 'epoch', 'train_loss', 'val_loss', 'acc', 'time', 'net_struct', 'model_type',
                  'num_layers', 'sample_size']
    with open(outfile_path, mode='w') as out_file:
        writer = csv.DictWriter(out_file, fieldnames=fieldnames, lineterminator='\n')
        writer.writeheader()

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

    net_struct = {
        'num_layers': 1,
        'M': [32],
        'k': [2],
        'p': [1],
        'g': [1]
    }

    print_exp_settings(seed, outfile_path, net_struct, actfun)

    for sample_size in range(5000, 50001, 5000):

        # ---- Loading MNIST
        seed_all(seed+sample_size)
        trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        mnist_train_full = datasets.MNIST(root='./data', train=True, download=True, transform=trans)
        train_set_indices = np.random.choice(50000, sample_size, replace=False)
        validation_set_indices = np.arange(50000, 60000)

        mnist_train = torch.utils.data.Subset(mnist_train_full, train_set_indices)
        mnist_validation = torch.utils.data.Subset(mnist_train_full, validation_set_indices)
        batch_size = 100
        train_loader = torch.utils.data.DataLoader(dataset=mnist_train, batch_size=batch_size,
                                                   shuffle=True, **kwargs)
        validation_loader = torch.utils.data.DataLoader(dataset=mnist_validation, batch_size=batch_size,
                                                        shuffle=True, **kwargs)

        # ---- Begin training model
        print("------------ Sample Size " + str(sample_size) + "...", flush=True)
        print()
        print("Sample of randomized indices and weights:")
        print(train_set_indices)
        train_model(actfun, net_struct, outfile_path, fieldnames, seed,
                    train_loader, validation_loader, sample_size, device)
        print()


# --------------------  Entry Point
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--seed', type=int, default=0, help='Job seed')
    parser.add_argument('--actfun', type=str, default='relu',
                        help='relu, multi_relu, cf_relu, combinact, l1, l2, l2_lae, abs, max'
                        )
    parser.add_argument('--save_path', type=str, default='', help='Where to save results')
    args = parser.parse_args()

    out = os.path.join(
        args.save_path,
        '{}-{}-{}.csv'.format(datetime.date.today(),
                              args.actfun,
                              args.seed))

    setup_experiment(args.seed,
                     out,
                     args.actfun
                     )

