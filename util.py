import torch
import torch.utils.data
import torch.nn as nn
import torchvision.datasets as datasets
import torchvision.transforms as transforms

import numpy as np
import random
import activation_functions as actfuns
from auto_augment import CIFAR10Policy
from collections import namedtuple
from sklearn import model_selection
from sklearn.datasets import load_iris
import os
try:
    from torch_lr_finder import LRFinder
    import matplotlib.pyplot as plt
    from apex import amp
except ImportError:
    print("Unable to load amp, LR finder, and pyplot")


# -------------------- Training Utils

def get_actfuns(actfun):
    if actfun == 'all':
        all_actfuns = ['combinact', 'relu', 'abs', 'max', 'min', 'lse', 'lae', 'l2', 'linf', 'prod', 'signed_geomean',
                       'swishk', 'bin_partition_full', 'bin_all_full']
    elif actfun == 'old_high_ord':
        all_actfuns = ['l2', 'combinact', 'max']
    elif actfun == 'all_pk':
        all_actfuns = ['l2', 'max', 'lae', 'signed_geomean', 'linf', 'swishk', 'prod']
    elif actfun == 'all_pk_relu':
        all_actfuns = ['l2', 'max', 'lae', 'signed_geomean', 'linf', 'swishk', 'prod', 'relu']
    elif actfun == 'all_pk_comb_relu':
        all_actfuns = ['l2', 'max', 'lae', 'signed_geomean', 'linf', 'swishk', 'prod', 'combinact', 'relu']
    elif actfun == 'e4_peff_intermediate_res':
        all_actfuns = ['prod', 'combinact', 'relu']
    elif actfun == 'pk_test':
        all_actfuns = ['l2', 'max', 'relu', 'combinact']
    elif actfun == 'pk_opt':
        all_actfuns = ['lae', 'signed_geomean', 'linf', 'swishk', 'prod']
    elif actfun == 'e3_peff3_log':
        all_actfuns = ['lae', 'signed_geomean', 'linf', 'swishk', 'prod', 'relu']
    elif actfun == 'bin':
        all_actfuns = ['bin_partition_full', 'bin_partition_nopass', 'bin_all_full', 'bin_all_nopass', 'bin_all_nopass_sgm']
    elif actfun == 'bin_duplicate':
        all_actfuns = ['bin_all_full', 'bin_all_nopass', 'bin_all_nopass_sgm']
    elif actfun == 'pg_redo':
        all_actfuns = ['linf', 'swishk', 'prod']
    elif actfun == 'rs_nparam':
        all_actfuns = ['max', 'relu', 'swishk', 'l2']
    elif actfun == 'max_relu':
        all_actfuns = ['max', 'relu']
    elif actfun == 'pg4':
        all_actfuns = ['linf', 'swishk', 'prod']
    else:
        all_actfuns = [actfun]

    return all_actfuns


def get_num_params(args):
    if args.var_n_params == 'new':
        num_params = [3e4, 1e5, 1e6, 1e7, 1e8]
    elif args.num_params == 0:
        num_params = [1e7]
    else:
        num_params = [args.num_params]

    return num_params


def get_train_samples(args):
    if args.var_n_samples:
        train_samples = [50000, 45000, 40000, 35000, 30000, 25000, 20000, 15000, 10000, 5000]
    elif args.overfit:
        train_samples = [2500]
    else:
        train_samples = [args.sample_size]

    return train_samples


def get_perm_methods(args):
    if args.var_perm_method:
        perm_methods = ['shuffle', 'roll', 'roll_grouped']
    else:
        perm_methods = [args.perm_method]
    return perm_methods


def get_pkg_vals(args):
    if args.var_k:
        k_vals = [2, 3, 4, 5, 6]
    else:
        k_vals = [args.k]

    if args.var_p:
        p_vals = [1, 2, 3, 4, 5]
    elif args.p_param_eff:
        p_vals = [2, 3]
    elif args.var_perm_method:
        p_vals = [2]
    elif args.var_pg:
        p_vals = [2, 4, 6, 8]
    else:
        p_vals = [args.p]

    if args.var_g:
        g_vals = [1, 2, 3, 4, 5]
    else:
        g_vals = [args.g]

    return p_vals, k_vals, g_vals


def weights_init(m):
    """
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
                       num_params, sample_size, batch_size, curr_k, curr_p,
                       curr_g, perm_method, resnet_ver, resnet_width, optim,
                       validation):

    print(
        "\n===================================================================\n\n"
        "Seed: {} \n"
        "Data Set: {} \n"
        "Outfile Path: {} \n"
        "Model Type: {} \n"
        "ResNet Version: {}\n"
        "ResNet Width: {}\n"
        "Activation Function: {} \n"
        "k: {}, p: {}, g: {}\n"
        "Permutation Type: {}\n"
        "Num Params: {}\n"
        "Num Training Samples: {}\n"
        "Batch Size: {}\n"
        "Optimizer: {}\n"
        "Validation: {}\n\n"
            .format(seed, dataset, outfile_path, curr_model, resnet_ver, resnet_width, curr_actfun,
                    curr_k, curr_p, curr_g, perm_method, num_params, sample_size, batch_size,
                    optim, validation,), flush=True
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

def conv_layer_params(kernel, in_dim, out_dim):
    return (kernel * in_dim + 1) * out_dim


def linear_layer_params(in_dim, out_dim):
    return (in_dim + 1) * out_dim


def get_cnn_num_params(n, in_channels, out_channels, in_dim, pk_ratio, g):
    total_params = conv_layer_params(9, in_channels, n[0])
    total_params += conv_layer_params(9, (pk_ratio / g) * n[0], n[1])
    total_params += conv_layer_params(9, (pk_ratio / g) * n[1], n[2])
    total_params += conv_layer_params(9, (pk_ratio / g) * n[2], n[2])
    total_params += conv_layer_params(9, (pk_ratio / g) * n[2], n[3])
    total_params += conv_layer_params(9, (pk_ratio / g) * n[3], n[3])

    for group in range(g):
        total_params += linear_layer_params(int((n[3] * pk_ratio) * (int(in_dim / 8) ** 2) / g), int(n[5] / g))
        total_params += linear_layer_params(int(n[5] * (pk_ratio / g)), int(n[4] / g))
    total_params += linear_layer_params(int(n[4] * (pk_ratio / g)), int(out_channels))

    return total_params


def calc_cnn_preacts(required_num_params, in_channels, out_channels, in_dim, pk_ratio, p, k, g):
    if required_num_params > 100000:
        n = np.array([2.0, 4.0, 8.0, 16.0, 32.0, 64.0])
    else:
        n = np.array([4.0, 4.0, 4.0, 6.0, 6.0, 8.0])
    fac = 2
    curr_num_params = get_cnn_num_params(n, in_channels, out_channels, in_dim, pk_ratio, g)
    dist = curr_num_params - required_num_params
    me = required_num_params * 0.01
    while np.abs(dist) > (200 + me):
        prev_dist = dist
        if curr_num_params > required_num_params:
            n *= 1/fac
        elif curr_num_params < required_num_params:
            n *= fac
        curr_num_params = get_cnn_num_params(n, in_channels, out_channels, in_dim, pk_ratio, g)
        dist = curr_num_params - required_num_params
        if prev_dist * dist < 0:
            fac = fac ** 0.75

    return n.astype(int)


def get_pk_ratio(actfun, p, k, g):
    if actfun == 'groupsort':
        pk_ratio = p
    elif actfun == 'bin_part_full' or actfun == 'ail_part_full':
        pk_ratio = (p * (2 + k)) / (3 * k)
    elif actfun == 'bin_all_full' or actfun == 'ail_all_full':
        pk_ratio = p * ((3 / k) + 1)
    elif actfun == 'bin_all_max_min_sgm' or actfun == 'ail_all_or_and_xnor':
        pk_ratio = p * ((3 / k))
    elif actfun == 'bin_all_max_min' or actfun == 'ail_all_or_and':
        pk_ratio = p * ((2 / k))
    elif actfun == 'bin_all_max_sgm' or actfun == 'ail_all_or_xnor':
        pk_ratio = p * ((2 / k))
    else:
        pk_ratio = (p / k)
    return pk_ratio


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


def permute(x, method, layer_type, k, offset, num_groups=2, shuffle_map=None):
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
    elif method == 'invert':
        if layer_type == 'linear':
            x = x.reshape(x.shape[0], int(x.shape[1] / k), k)
            x = x[:, :, shuffle_map]
            x = x.reshape(x.shape[0], x.shape[1] * x.shape[2])
        elif layer_type == 'conv':
            x = x.reshape(x.shape[0], int(x.shape[1] / k), k, x.shape[2], x.shape[3])
            x = x[:, :, shuffle_map, ...]
            x = x.reshape(x.shape[0], x.shape[1] * x.shape[2], x.shape[3], x.shape[4])

        return x


def load_dataset(
        args,
        model,
        dataset,
        seed=0,
        validation=False,
        batch_size=None,
        train_sample_size=60000,
        kwargs=None):

    seed_all(seed)

    if dataset == 'iris':
        features, labels = load_iris(return_X_y=True)
        features_train, features_test, labels_train, labels_test = model_selection.train_test_split(features, labels,
                                                                                                    random_state=0,
                                                                                                    train_size=0.8,
                                                                                                    stratify=labels,
                                                                                                    shuffle=True)

        train_dataset = torch.utils.data.TensorDataset(torch.Tensor(features_train),
                                                       torch.tensor(labels_train, dtype=torch.long))
        eval_dataset = torch.utils.data.TensorDataset(torch.Tensor(features_test),
                                                      torch.tensor(labels_test, dtype=torch.long))
        aug_train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=features_train.shape[0],
                                                       drop_last=True,
                                                       shuffle=True, **kwargs)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=features_train.shape[0], drop_last=True,
                                                   shuffle=True,
                                                   **kwargs)
        aug_eval_loader = torch.utils.data.DataLoader(eval_dataset, batch_size=features_train.shape[0],
                                                      drop_last=False,
                                                      shuffle=False, **kwargs)
        eval_loader = torch.utils.data.DataLoader(eval_dataset, batch_size=features_train.shape[0],
                                                  drop_last=False,
                                                  shuffle=False,
                                                  **kwargs)

        return aug_train_loader, train_loader, aug_eval_loader, eval_loader, features_train.shape[0], 1

    if dataset == 'mnist':
        aug_trans, trans = [], []
        if args.aug:
            aug_trans.append(transforms.RandomAffine(degrees=10, scale=(0.8, 1.2), translate=(0.08, 0.08), shear=0.3))
        aug_trans.append(transforms.ToTensor())
        aug_trans.append(transforms.Normalize((0.1307,), (0.3081,)))
        trans.append(transforms.ToTensor())
        trans.append(transforms.Normalize((0.1307,), (0.3081,)))

        aug_trans_all = transforms.Compose(aug_trans)
        trans_all = transforms.Compose(trans)
        aug_train_set = datasets.MNIST(root='./data', train=True, download=True, transform=aug_trans_all)
        train_set = datasets.MNIST(root='./data', train=True, download=True, transform=trans_all)
        aug_test_set = datasets.MNIST(root='./data', train=False, download=True, transform=aug_trans_all)
        test_set = datasets.MNIST(root='./data', train=False, download=True, transform=trans_all)

        if batch_size is None:
            batch_size = 256

    elif dataset == 'cifar10' or dataset == 'cifar100':
        aug_trans, trans = [], []
        if args.aug:
            aug_trans.append(transforms.RandomHorizontalFlip())
            aug_trans.append(CIFAR10Policy())
        aug_trans.append(transforms.ToTensor())
        aug_trans.append(transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))
        trans.append(transforms.ToTensor())
        trans.append(transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))

        aug_trans_all = transforms.Compose(aug_trans)
        trans_all = transforms.Compose(trans)
        if dataset == 'cifar10':
            aug_train_set = datasets.CIFAR10(root='./data', train=True, download=True, transform=aug_trans_all)
            train_set = datasets.CIFAR10(root='./data', train=True, download=True, transform=trans_all)
            aug_test_set = datasets.CIFAR10(root='./data', train=False, download=True, transform=aug_trans_all)
            test_set = datasets.CIFAR10(root='./data', train=False, download=True, transform=trans_all)
        elif dataset == 'cifar100':
            aug_train_set = datasets.CIFAR100(root='./data', train=True, download=True, transform=aug_trans_all)
            train_set = datasets.CIFAR100(root='./data', train=True, download=True, transform=trans_all)
            aug_test_set = datasets.CIFAR100(root='./data', train=False, download=True, transform=aug_trans_all)
            test_set = datasets.CIFAR100(root='./data', train=False, download=True, transform=trans_all)

        if batch_size is None:
            batch_size = 256

    train_sample_size = len(train_set) if train_sample_size is None else train_sample_size
    if validation:
        if dataset == 'mnist':
            train_idx = np.arange(50000)
            train_idx = train_idx[:train_sample_size]
            val_idx = np.arange(50000, 60000)
        else:
            train_sample_indices = np.arange(train_sample_size)
            train_sample_lbls = train_set.targets[:train_sample_size]
            train_sample_lbls = train_sample_lbls.numpy() if isinstance(train_set.data, torch.Tensor) else train_sample_lbls
            train_idx, val_idx, _, _ = model_selection.train_test_split(train_sample_indices,
                                                                        train_sample_lbls,
                                                                        test_size=0.1,
                                                                        stratify=train_sample_lbls,
                                                                        random_state=0)
        aug_eval_set = aug_train_set
        eval_set = train_set
    else:
        train_idx = np.random.choice(len(train_set), train_sample_size, replace=False)
        val_idx = np.arange(len(test_set))

        aug_eval_set = aug_test_set
        eval_set = test_set

    train_sample_size = train_idx.shape[0]

    aug_train_set = torch.utils.data.Subset(aug_train_set, train_idx)
    train_set = torch.utils.data.Subset(train_set, train_idx)
    aug_eval_set = torch.utils.data.Subset(aug_eval_set, val_idx)
    eval_set = torch.utils.data.Subset(eval_set, val_idx)

    aug_train_loader = torch.utils.data.DataLoader(aug_train_set, batch_size=batch_size, drop_last=True, shuffle=True, **kwargs)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, drop_last=True, shuffle=True, **kwargs)
    aug_eval_loader = torch.utils.data.DataLoader(aug_eval_set, batch_size=batch_size, drop_last=False, shuffle=False, **kwargs)
    eval_loader = torch.utils.data.DataLoader(eval_set, batch_size=batch_size, drop_last=False, shuffle=False, **kwargs)

    return aug_train_loader, train_loader, aug_eval_loader, eval_loader, train_sample_size, batch_size


def get_rms_hyperparams(args):
    if args.actfun == 'swish':
        grid_settings = [(lr_init, lr_gamma, alpha, momentum)
                         for lr_init in [1e-3, 1e-4, 1e-5]
                         for lr_gamma in [0.98, 0.99]
                         for alpha in [0.9, 0.99]
                         for momentum in [0, 0.9]]
    else:
        grid_settings = [(lr_init, lr_gamma, alpha, momentum)
                         for lr_init in [1e-4]
                         for lr_gamma in [0.98, 0.99]
                         for alpha in [0.9, 0.99]
                         for momentum in [0]]

    return grid_settings[args.grid_id]


def get_grid_id(actfun, args):
    if actfun == 'max':
        return 6
    elif actfun == 'relu' or actfun == 'swish':
        return 16
    elif actfun == 'swishk':
        if args.perm_method == 'invert':
            return 16
        else:
            return 10
    elif actfun == 'swishy':
        if args.perm_method == 'invert':
            return 16
        else:
            return 13
    elif actfun == 'bin_all_max_min':
        return 9
    elif actfun == 'ail_or' or actfun == 'ail_xnor':
        return 0
    elif actfun == 'ail_all_or_and':
        return 3
    elif actfun == 'ail_all_or_xnor':
        return 19
    elif actfun == 'ail_all_or_and_xnor':
        return 2
    elif actfun == 'ail_part_or_xnor':
        return 14
    elif actfun == 'ail_part_or_and_xnor':
        return 5

    else:
        return args.grid_id


def print_model_params(model):
    print("=============================== Hyper params:")
    i = 0
    for name, param in model.named_parameters():
        # print(name, param.shape)
        if len(param.shape) == 4:
            print(param[:2, :2, :4, :4])
            break
        elif len(param.shape) == 3:
            print(param[:2, :2, :2])
        elif len(param.shape) == 2:
            print(param[:4, :4])
            break
        elif len(param.shape) == 1:
            print(param[:3])
        print()
        i += 1
        if i == 4:
            break
    print("===================================================================")


def run_lr_finder(
        args,
        model,
        train_loader,
        optimizer,
        criterion,
        val_loader=None,
        verbose=True,
        show=True,
        figpth=None,
        device=None,
        recommender="logmean14",
):
    if verbose:
        print("Running learning rate finder")
    if args.mix_pre_apex:
        model, optimizer = amp.initialize(model, optimizer, opt_level="O2")
    lr_finder = LRFinder(model, optimizer, criterion, device=device)
    min_lr = 1e-7 if args.model == 'mlp' else 1e-10
    lr_finder.range_test(
        train_loader,
        val_loader=val_loader,
        start_lr=min_lr,
        end_lr=10,
        num_iter=200,
        diverge_th=3,
    )
    # print(lr_finder.history["lr"])
    # print(lr_finder.history["loss"])
    min_index = np.argmin(lr_finder.history["loss"])
    lr_at_min = lr_finder.history["lr"][min_index]
    min_loss = lr_finder.history["loss"][min_index]
    max_index = np.argmax(lr_finder.history["loss"][:min_index])
    lr_at_max = lr_finder.history["lr"][max_index]
    max_loss = lr_finder.history["loss"][max_index]
    if not show and not figpth:
        lr_steepest = None
    else:
        if verbose:
            print("Plotting learning rate finder results")
        hf = plt.figure(figsize=(15, 9))
        ax = plt.axes()
        _, lr_steepest = lr_finder.plot(skip_start=0, skip_end=3, log_lr=True, ax=ax)
        ylim = np.array([min_loss, max_loss])
        ylim += 0.1 * np.diff(ylim) * np.array([-1, 1])
        plt.ylim(ylim)
        plt.tick_params(reset=True, color=(0.2, 0.2, 0.2))
        plt.tick_params(labelsize=14)
        ax.minorticks_on()
        ax.tick_params(direction="out")
    init_loss = lr_finder.history["loss"][0]
    loss_12 = min_loss + 0.5 * (max_loss - min_loss)
    index_12 = max_index + np.argmin(
        np.abs(np.array(lr_finder.history["loss"][max_index:min_index]) - loss_12)
    )
    lr_12 = lr_finder.history["lr"][index_12]
    loss_13 = min_loss + 1 / 3 * (max_loss - min_loss)
    index_13 = max_index + np.argmin(
        np.abs(np.array(lr_finder.history["loss"][max_index:min_index]) - loss_13)
    )
    lr_13 = lr_finder.history["lr"][index_13]
    loss_23 = min_loss + 2 / 3 * (max_loss - min_loss)
    index_23 = max_index + np.argmin(
        np.abs(np.array(lr_finder.history["loss"][max_index:min_index]) - loss_23)
    )
    lr_23 = lr_finder.history["lr"][index_23]
    loss_14 = min_loss + 1 / 4 * (max_loss - min_loss)
    index_14 = max_index + np.argmin(
        np.abs(np.array(lr_finder.history["loss"][max_index:min_index]) - loss_14)
    )
    lr_14 = lr_finder.history["lr"][index_14]
    if recommender == "div10":
        lr_recomend = np.exp(np.mean([np.log(lr_at_min / 10), np.log(lr_12)]))
    elif recommender == "min12":
        lr_recomend = np.min([lr_at_min / 10, lr_12])
    elif recommender == "min13":
        lr_recomend = np.min([lr_at_min / 10, lr_13])
    elif recommender == "min14":
        lr_recomend = np.min([lr_at_min / 10, lr_14])
    elif recommender == "logmean12":
        lr_recomend = np.exp(np.mean([np.log(lr_at_min / 10), np.log(lr_12)]))
    elif recommender == "logmean13":
        lr_recomend = np.exp(np.mean([np.log(lr_at_min / 10), np.log(lr_13)]))
    elif recommender == "logmean14":
        lr_recomend = np.exp(np.mean([np.log(lr_at_min / 10), np.log(lr_14)]))
    if verbose:
        if lr_steepest is not None:
            print("LR at steepest grad: {:.3e}  (red)".format(lr_steepest))
        print("LR at minimum loss : {:.3e}".format(lr_at_min))
        print("LR a tenth of min  : {:.3e}  (orange)".format(lr_at_min / 10))
        print("LR when 1/4 up     : {:.3e}  (yellow)".format(lr_14))
        print("LR when 1/3 up     : {:.3e}  (blue)".format(lr_13))
        print("LR when 1/2 up     : {:.3e}  (cyan)".format(lr_12))
        print("LR when 2/3 up     : {:.3e}  (green)".format(lr_23))
        print("LR recommended     : {:.3e}  (black)".format(lr_recomend))
    if show or figpth:
        ax.axvline(x=lr_steepest, color="red")
        ax.axvline(x=lr_at_min / 10, color="orange")
        ax.axvline(x=lr_14, color="yellow")
        ax.axvline(x=lr_13, color="blue")
        ax.axvline(x=lr_12, color="cyan")
        ax.axvline(x=lr_23, color="green")
        ax.axvline(x=lr_recomend, color="black", ls=":")
    if figpth:
        # Save figure
        os.makedirs(os.path.dirname(figpth), exist_ok=True)
        plt.savefig(figpth)
        if verbose:
            print("LR Finder results saved to {}".format(figpth))
    if show:
        plt.show()
    return lr_recomend


class PiecewiseLinear(namedtuple('PiecewiseLinear', ('knots', 'vals'))):

    def __call__(self, t):
        return np.interp([t], self.knots, self.vals)[0]
