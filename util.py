import torch
import torch.utils.data
import torch.nn as nn
import torchvision.datasets as datasets
import torchvision.transforms as transforms

import numpy as np
import random
import activation_functions as actfuns


# -------------------- Training Utils

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


def get_n_params(model):
    """
    :param model: Pytorch network model
    :return: Number of parameters in the model
    """
    pp = 0
    for p in list(model.parameters()):
        nn = 1
        for s in list(p.size()):
            nn = nn * s
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


def print_exp_settings(seed, dataset, outfile_path, net_struct, curr_model):
    print(
        "\n===================================================================\n\n"
        "Seed: {} \n"
        "Data Set: {} \n"
        "Outfile Path: {} \n"
        "Number of Layers: {} \n"
        "Network Structure: \n"
        "{} \n"
        "Model Type: {} \n\n"
            .format(seed, dataset, outfile_path, net_struct['num_layers'], net_struct, curr_model), flush=True
    )


# -------------------- Model Utils

def test_net_inputs(actfun, net_struct, in_size):
    """
    Tests network structure and activation hyperparameters to make sure they are valid
    :param actfun: activation function used by network
    :param net_struct: given network structure
    :param in_size: number of inputs
    :return:
    """

    if actfun not in actfuns.get_actfuns() and actfun != 'combinact':
        return 'Invalid activation function: {}'.format(actfun)

    layer_inputs = in_size
    for layer in range(net_struct['num_layers']):
        M = net_struct['M'][layer]
        k = net_struct['k'][layer]
        p = net_struct['p'][layer]
        g = net_struct['g'][layer]

        if layer_inputs % g != 0:
            return 'g must divide the number of output (post activation) nodes from the previous layer.  ' \
                   'Layer = {}, Previous layer outputs = {}, g = {}'.format(layer, layer_inputs, g)
        if M % g != 0:
            return 'g must divide the number of pre-activation nodes in this layer.  ' \
                   'Layer = {}, M = {}, g = {}'.format(layer, M, g)
        if M % k != 0:
            return 'k must divide the number of nodes M in this layer. ' \
                   'Layer = {}, M = {}, k = {}'.format(layer, M, k)

        layer_inputs = M * p / k

    return None


def permute(x, method, offset, num_groups=2, shuffle_map=None):
    if method == "roll":
        return torch.cat((x[:, offset:, 0], x[:, :offset, 0]), dim=1)
    elif method == "roll_grouped":
        group_size = int(x.shape[1] / num_groups)
        output = None
        for i, group in enumerate(range(num_groups)):
            start = offset + group_size * group
            if i == num_groups - 1:
                group_size += x.shape[1] % num_groups
            end = start + group_size - offset
            curr_roll = torch.cat((x[:, start:end, 0], x[:, start - offset:start, 0]),
                                  dim=1)
            if group == 0:
                output = curr_roll
            else:
                output = torch.cat((output, curr_roll), dim=1)
        return output
    elif method == "shuffle":
        return x[:, shuffle_map, [0]]


def load_dataset(dataset,
                 seed=0,
                 batch_size=None,
                 kwargs=None,
                 sample_size=None):

    seed_all(seed)

    if dataset == 'mnist':
        trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        train_set_full = datasets.MNIST(root='./data', train=True, download=True, transform=trans)
        test_set_full = datasets.MNIST(root='./data', train=False, download=True, transform=trans)

        if sample_size is None:
            sample_size = 60000
        if batch_size is None:
            batch_size = 100

        train_set_indices = np.random.choice(60000, sample_size, replace=False)
        test_set_indices = np.random.choice(10000, 10000, replace=False)

    elif dataset == 'cifar10':
        trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        train_set_full = datasets.CIFAR10(root='./data', train=True, download=True, transform=trans)
        test_set_full = datasets.CIFAR10(root='./data', train=False, download=True, transform=trans)

        if sample_size is None:
            sample_size = 50000
        if batch_size is None:
            batch_size = 4

        train_set_indices = np.random.choice(50000, sample_size, replace=False)
        test_set_indices = np.random.choice(10000, 10000, replace=False)

    print("------------ Sample Size " + str(sample_size) + "...", flush=True)
    print()

    train_set = torch.utils.data.Subset(train_set_full, train_set_indices)
    test_set = torch.utils.data.Subset(test_set_full, test_set_indices)

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, **kwargs)
    validation_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False, **kwargs)

    return train_loader, validation_loader, sample_size
