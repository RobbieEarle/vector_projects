import torch
import torch.utils.data
import torch.nn as nn
import torchvision.datasets as datasets
import torchvision.transforms as transforms

import numpy as np
import random
import activation_functions as actfuns


# -------------------- Training Utils

def get_extras(args):
    extras = ""
    if args.var_n_params:
        extras += '-var_n_params'
    if args.var_n_samples:
        extras += '-var_n_samples'
    if args.reduce_actfuns:
        extras += '-reduce_actfuns'
    if args.var_k:
        extras += '-var_k'
    if args.var_p:
        extras += '-var_p'
    if args.var_perm_method:
        extras += '-var_perm'
    if args.p_param_eff:
        extras += '-p_param_eff'
    return extras


def get_actfuns(actfun):
    if actfun == 'all':
        all_actfuns = ['combinact', 'relu', 'abs', 'max', 'min', 'lse', 'lae', 'l2', 'linf', 'prod', 'signed_geomean',
                       'swishk', 'binary_ops_partition', 'binary_ops_all']
    elif actfun == '1d':
        all_actfuns = ['relu', 'abs']
    elif actfun == 'old_all':
        all_actfuns = ['relu', 'abs', 'l2', 'combinact', 'max', 'multi_relu', 'l2_lae']
    elif actfun == 'old_high_ord':
        all_actfuns = ['l2', 'combinact', 'max']
    elif actfun == 'new_all':
        all_actfuns = ['min', 'lse', 'lae', 'linf', 'prod', 'signed_geomean', 'swishk', 'binary_ops_partition',
                       'binary_ops_all']
    elif actfun == 'pk_test':
        all_actfuns = ['l2', 'max']
    elif actfun == 'pk_non_opt':
        all_actfuns = ['lae', 'signed_geomean', 'linf', 'swishk', 'prod']
    elif actfun == 'pk_non_opt_all':
        all_actfuns = ['lae', 'signed_geomean', 'linf', 'swishk', 'prod', 'relu']
    else:
        all_actfuns = [actfun]

    return all_actfuns


def get_num_params(args):
    if args.model == 'nn':
        if args.var_n_params:
            num_params = [1000000, 800000, 600000, 400000, 200000]
        else:
            num_params = [1000000]
    elif args.model == 'cnn':
        if args.var_n_params:
            num_params = [3000000, 2500000, 2000000, 1500000, 1000000, 500000]
        else:
            num_params = [3000000]
    return num_params


def get_train_samples(args):
    if args.var_n_samples:
        train_samples = [50000, 45000, 40000, 35000, 30000, 25000, 20000, 15000, 10000, 5000]
    elif args.overfit:
        train_samples = [15000]
    else:
        train_samples = [args.sample_size]

    return train_samples


def get_perm_methods(perm_method):
    if perm_method:
        perm_methods = ['shuffle', 'roll', 'roll_grouped']
    else:
        perm_methods = ['shuffle']
    return perm_methods


def get_pk_vals(args):
    if args.var_k:
        k_vals = [2, 3, 4, 5, 6]
    else:
        k_vals = [2]
    if args.var_p:
        p_vals = [1, 2, 3, 4, 5]
    elif args.p_param_eff:
        p_vals = [2, 3]
    elif args.var_perm_method:
        p_vals = [2]
    else:
        p_vals = [1]

    return p_vals, k_vals


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


def get_model_params(model):
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


def print_exp_settings(seed, dataset, outfile_path, curr_model, curr_actfun,
                       hyper_params, num_params, sample_size, curr_k, curr_p,
                       perm_method):
    print(
        "\n===================================================================\n\n"
        "Seed: {} \n"
        "Data Set: {} \n"
        "Outfile Path: {} \n"
        "Model Type: {} \n"
        "Activation Function: {} \n"
        "Hyper-params: {} \n"
        "k: {}, p: {}\n"
        "Permutation Type: {}\n"
        "Num Params: {}\n"
        "Sample Size: {}\n\n"
            .format(seed, dataset, outfile_path, curr_model, curr_actfun, hyper_params,
                    curr_k, curr_p, perm_method, num_params, sample_size), flush=True
    )


def hook_f(module, input, output):
    print("FORWARD")
    print(module)
    print(input[0].shape)
    if len(input[0].shape) == 4:
        print(input[0][0, 0, :4, :4])
    elif len(input[0].shape) == 2:
        print(input[0][0, :16])
    # print("IN: {} {}".format(type(input), len(input)))
    # for curr_in in input:
    #     print("     {}".format(curr_in.shape))
    # print("OUT: {} {}".format(type(output), len(output)))
    # for curr_out in output:
    #     print("     {}".format(curr_out.shape))
    print()


def hook_b(module, input, output):
    print("BACKWARD")
    print(module)
    print(input[0].shape)
    if len(input[0].shape) == 4:
        print(input[0][0, 0, :4, :4])
    elif len(input[0].shape) == 2:
        print(input[0][0, :10])
    # print("IN: {} {}".format(type(input), len(input)))
    # for curr_in in input:
    #     print("     {}".format(curr_in.shape))
    # print("OUT: {} {}".format(type(output), len(output)))
    # for curr_out in output:
    #     print("     {}".format(curr_out.shape))
    print()


# -------------------- Model Utils


def get_cnn_num_params(n, in_dim, out_dim, p, k):
    pk_ratio = p / k
    constraint_func1 = (9 * (in_dim * n[0])) + (
                9 * pk_ratio * ((n[0] * n[1]) + (n[1] * n[2]) + (n[2] * n[2]) + (n[2] * n[3]) + (n[3] * n[3])))
    constraint_func2 = (pk_ratio * (int(in_dim / 8) ** 2) * n[3] * n[5]) + (pk_ratio * n[4] * n[5]) + (
                pk_ratio * n[4] * out_dim)
    constraint_func3 = in_dim + np.sum(n) + n[2] + n[3] + out_dim
    return constraint_func1 + constraint_func2 + constraint_func3


def calc_cnn_preacts(required_num_params, in_dim, out_dim, p, k):
    n = np.array([2.0, 4.0, 8.0, 16.0, 32.0, 64.0])
    curr_num_params = 0
    while curr_num_params < required_num_params:
        curr_num_params = get_cnn_num_params(n, in_dim, out_dim, p, k)
        if curr_num_params < required_num_params:
            n *= 2

    fac = 2
    dist = curr_num_params - required_num_params
    me = required_num_params * 0.001
    while np.abs(dist) > me:
        prev_dist = dist
        if curr_num_params > required_num_params:
            n *= 1/fac
        elif curr_num_params < required_num_params:
            n *= fac
        curr_num_params = get_cnn_num_params(n, in_dim, out_dim, p, k)
        dist = curr_num_params - required_num_params
        if prev_dist * dist < 0:
            fac = fac ** 0.75

    return n.astype(int)


def test_nn_inputs(actfun, net_struct, in_size):
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


def add_shuffle_map(shuffle_maps, num_nodes, p):
    new_maps = []
    for perm in range(p):
        new_maps.append(torch.randperm(num_nodes))
    shuffle_maps.append(new_maps)
    return shuffle_maps


def permute(x, method, offset, num_groups=2, shuffle_map=None):
    if method == "roll":
        return torch.cat((x[:, offset:, ...], x[:, :offset, ...]), dim=1)
    elif method == "roll_grouped":
        group_size = int(x.shape[1] / num_groups)
        output = None
        for i, group in enumerate(range(num_groups)):
            start = offset + group_size * group
            if i == num_groups - 1:
                group_size += x.shape[1] % num_groups
            end = start + group_size - offset
            curr_roll = torch.cat((x[:, start:end, ...], x[:, start - offset:start, ...]),
                                  dim=1)
            if group == 0:
                output = curr_roll
            else:
                output = torch.cat((output, curr_roll), dim=1)
        return output
    elif method == "shuffle":
        return x[:, shuffle_map, ...]


def load_dataset(dataset,
                 seed=0,
                 batch_size=None,
                 sample_size=60000,
                 kwargs=None):

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

    elif dataset == 'fashion_mnist':
        trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        train_set_full = datasets.FashionMNIST(root='./data', train=True, download=True, transform=trans)
        test_set_full = datasets.FashionMNIST(root='./data', train=False, download=True, transform=trans)

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
            batch_size = 64

        train_set_indices = np.random.choice(50000, sample_size, replace=False)
        test_set_indices = np.random.choice(10000, 10000, replace=False)

    elif dataset == 'cifar100':
        trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        train_set_full = datasets.CIFAR100(root='./data', train=True, download=True, transform=trans)
        test_set_full = datasets.CIFAR100(root='./data', train=False, download=True, transform=trans)

        if sample_size is None:
            sample_size = 50000
        if batch_size is None:
            batch_size = 64

        train_set_indices = np.random.choice(50000, sample_size, replace=False)
        test_set_indices = np.random.choice(10000, 10000, replace=False)

    elif dataset == 'svhn':
        trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        dataset_full = datasets.SVHN(root='./data', download=True, transform=trans)
        train_set_full = torch.utils.data.Subset(dataset_full, torch.arange(60000))
        test_set_full = torch.utils.data.Subset(dataset_full, torch.arange(10000) + 60000)

        if sample_size is None:
            sample_size = 60000
        if batch_size is None:
            batch_size = 64

        train_set_indices = np.random.choice(60000, sample_size, replace=False)
        test_set_indices = np.random.choice(10000, 10000, replace=False)

    print("------------ Sample Size " + str(sample_size) + "...", flush=True)
    print()

    train_set = torch.utils.data.Subset(train_set_full, train_set_indices)
    test_set = torch.utils.data.Subset(test_set_full, test_set_indices)

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, **kwargs)
    validation_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False, **kwargs)

    return train_loader, validation_loader, sample_size, batch_size
