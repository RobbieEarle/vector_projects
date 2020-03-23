import torch
import torch.utils.data
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.optim.lr_scheduler import CyclicLR
from torch import logsumexp

import sys
import numpy as np
import math
import numbers
import random
import datetime
import csv
import time


def actfun_signed_geomean(z):
    prod = torch.prod(z, dim=2)
    signs = prod.sign()
    return signs * prod.abs().sqrt()


def actfun_signed_l3(z):
    x3 = z[:, :, 0].pow(3)
    y3 = z[:, :, 1].pow(3)
    out_val = (x3 + y3).tanh() * (x3 + y3).abs().pow(1 / 3)
    return out_val


def actfun_logavgexp(input, dim, keepdim=False, temperature=None, dtype=torch.float32):
    if isinstance(temperature, numbers.Number) and temperature == 1:
        temperature = None
    input_dtype = input.dtype
    if dtype is not None:
        input = input.to(dtype)
    if isinstance(temperature, torch.Tensor):
        temperature = temperature.to(input.dtype)
    if temperature is not None:
        input = input.div(temperature)
    log_n = math.log(input.shape[dim])
    lae = logsumexp(input, dim=dim, keepdim=True).sub(log_n)
    if temperature is not None:
        lae = lae.mul(temperature)
    if not keepdim:
        lae = lae.squeeze(dim)
    return lae.to(input_dtype)


_ln2 = 0.6931471805599453
_ACTFUNS2D = {
    'max':
        lambda z: torch.max(z, dim=2).values,
    'min':
        lambda z: torch.min(z, dim=2).values,
    'signed_geomean':
        lambda z: actfun_signed_geomean(z),
    'swishk':
        lambda z: z[:, :, 0] * torch.prod((torch.sigmoid(z)), dim=2),
    'l2':
        lambda z: (torch.sum(z.pow(2), dim=2)).sqrt_(),
    'l3-signed':
        lambda z: actfun_signed_l3(z),
    'linf':
        lambda z: torch.max(z.abs(), dim=2).values,
    'lse':
        lambda z: torch.logsumexp(z, dim=2),
    'lae':
        lambda z: actfun_logavgexp(z, dim=2),
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


def test_net_inputs(net_struct, actfuns, in_size, out_size):
    """
    Tests network structure and activation hyperparameters to make sure they are valid
    :param net_struct: given network structure
    :param actfuns: given activation functions
    :param in_size: number of inputs
    :param out_size: number of outputs
    :return:
    """

    if len(actfuns) != net_struct.size()[0]:
        return 'actfuns must have one set of entries for each layer. Layers: {}, Sets: {}'.format(net_struct.size()[0], len(actfuns))

    layer_inputs = in_size
    for layer in range(net_struct.size()[0]):
        M = int(net_struct[layer, 0])
        k = int(net_struct[layer, 1])
        p = int(net_struct[layer, 2])
        g = int(net_struct[layer, 3])

        if layer_inputs % g != 0:
            return 'g must divide the number of output (post activation) nodes from the previous layer.  ' \
                   'Layer = {}, Previous layer outputs = {}, g = {}'.format(layer, layer_inputs, g)
        if M % g != 0:
            return 'g must divide the number of pre-activation nodes in this layer.  ' \
                   'Layer = {}, M = {}, g = {}'.format(layer, M, g)
        if (M / g) % k != 0:
            return 'k must divide the number of nodes M divided by the number of groups in this layer. ' \
                   'Layer = {}, M / g_prev = {}, k = {}'.format(layer, M / g, k)
        if len(actfuns[layer]) != p:
            return 'Each layer\'s set of actfun entries must equal that layer\'s p value. ' \
                   'Layer: {}, num entries: {}, p: {}'.format(layer, len(actfuns[layer]), p)

        layer_inputs = M * p / k

    return None


class CombinactNet(nn.Module):

    def __init__(self,
                 net_struct,
                 actfuns,
                 in_size,
                 out_size,
                 batch_size=100
                 ):
        """
        :param net_struct: Structure of network. L x 4 array, L = number of hidden layers
            M = Number of pre-activation nodes in current hidden layer
            k = Size of activation inputs for current layer (cluster size)
            p = Number of permutations of pre-activation nodes in current hidden layer
            g = Number of groups to apply to previous layer outputs and current layer pre-activation nodes
        :param actfuns: L x p array of activation functions to be applied in each permutation of each layer
        :param in_size: Number of input nodes
        :param out_size: Number of outputs nodes
        :param batch_size: Batchsize for minibatch optimization
        """

        super(CombinactNet, self).__init__()

        # ---- Error checking given network structure and activation functions
        error = test_net_inputs(net_struct, actfuns, in_size, out_size)
        if error is not None:
            raise ValueError(error)

        # self.actfun = 'lae'

        self.num_hidden_layers = net_struct.size()[0]
        self.net_struct = net_struct
        self.actfuns = actfuns
        self.in_size = in_size
        self.out_size = out_size
        self.batch_size = batch_size
        self.all_weights = nn.ModuleList()
        self.all_batch_norms = nn.ModuleList()
        self.hyper_params = {'M': [], 'k': [], 'p': [], 'g': []}

        # Creating nn.Linear transformations for each layer. Also stores hyperparams in easier to reference dict
        layer_inputs = int(in_size)
        for layer in range(self.num_hidden_layers + 1):
            # Last layer is output layer
            if layer == self.num_hidden_layers:
                M = int(out_size)
                k = 1
                p = 1
                g = 1
            else:
                M = int(net_struct[layer, 0])
                k = int(net_struct[layer, 1])
                p = int(net_struct[layer, 2])
                g = int(net_struct[layer, 3])
                self.hyper_params['M'].append(M)
                self.hyper_params['k'].append(k)
                self.hyper_params['p'].append(p)
                self.hyper_params['g'].append(g)
                self.all_batch_norms.append(nn.ModuleList([nn.BatchNorm1d(int(M / g)) for i in range(g)]))

            num_pre_act_nodes = M
            self.all_weights.append(nn.ModuleList([nn.Linear(int(layer_inputs / g), int(num_pre_act_nodes / g)) for i in range(g)]))

            layer_inputs = M * p / k

    def permute(self, x, method, seed):
        if method == "roll":
            return torch.cat((x[:, seed:, 0], x[:, :seed, 0]), dim=1)

    def forward(self, x):

        # x is initially torch.Size([100, 1, 28, 28]), this step converts to torch.Size([100, 784])
        x = x.view(self.batch_size, -1)

        # TODO: Handle ReLU
        # # Handles relu activation functions (used as control in experiment)
        # if self.actfun == 'relu':
        #     x = F.relu(self.bn1(self.fc1(x)))
        #     x = F.relu(self.bn2(self.r2_fc_groups[0](x)))
        #     x = self.r3_fc_groups[0](x)

        layer_inputs = self.in_size
        for layer in range(self.num_hidden_layers + 1):

            # Last layer is output layer, so hyperparameters are all 1
            if layer == self.num_hidden_layers:
                M = self.out_size
                k = 1
                p = 1
                g = 1
            # Otherwise retrieves hyperparameters for this layer
            else:
                M = self.hyper_params['M'][layer]
                k = self.hyper_params['k'][layer]
                p = self.hyper_params['p'][layer]
                g = self.hyper_params['g'][layer]
                layer_actfuns = self.actfuns[layer]

            # Group inputs
            x = x.reshape(self.batch_size, int(layer_inputs / g), g)

            # Creates placeholder for outputs (ie. post activation nodes)
            outputs = torch.zeros((self.batch_size, int((M / k) / g), p * g), device=x.device)

            # For each group in this layer
            for i, fc in enumerate(self.all_weights[layer]):

                # If this layer is the last layer, we simply output the pre-activation nodes in this layer
                if layer == self.num_hidden_layers:
                    post_act_nodes = fc(x[:, :, i]).unsqueeze(dim=2)

                # Otherwise we apply batchnorm to pre-activation nodes, and then apply our activation functions
                else:
                    pre_act_nodes = self.all_batch_norms[layer][i](fc(x[:, :, i]))
                    post_act_nodes = self.activate(pre_act_nodes, layer_actfuns, int(M / g), k, p)
                outputs[:, :, i * p:(i + 1) * p] = post_act_nodes

            # We transpose so that when we reshape our outputs, the results from the permutations merge correctly
            x = torch.transpose(outputs, dim0=1, dim1=2)

            # Records the number of outputs from this layer as the number of inputs for the next layer
            layer_inputs = M * p / k

        x = x.reshape((self.batch_size, self.out_size))

        return x

    def activate(self, x, actfuns, M, k, p):
        clusters = math.floor(M / k)
        x = x.view(self.batch_size, M, 1)

        # Duplicate and permute x
        for i in range(1, p):
            x = torch.cat((x[:, :, :i], self.permute(x, "roll", i).view(self.batch_size, M, 1)), dim=2)

        x = x.view(self.batch_size, clusters, k, p)

        # Apply activations functions to each permutation
        outputs = torch.zeros((self.batch_size, clusters, p), device=x.device)
        for perm in range(p):
            outputs[:, :, perm] = _ACTFUNS2D[actfuns[perm]](x[:, :, :, perm])
        x = outputs

        return x


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


def train_model(model, outfile_path, fieldnames, seed, train_loader, validation_loader, hyper_params):
    """
    Runs training session for a given randomized model
    :param model: model to train
    :param outfile_path: path to save outputs from training session
    :param fieldnames: column names for output file
    :param seed: seed for randomization
    :param train_loader: training data loader
    :param validation_loader: validation data loader
    :param hyper_params: optimizer and scheduler hyperparameters
    :return:
    """

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
                         step_size_up=int(hyper_params['cycle_peak'] * 5000),  # 5000 = tot number of batches: 500 * 10
                         step_size_down=int((1-hyper_params['cycle_peak']) * 5000),
                         cycle_momentum=False
                         )

    # ---- Start Training
    epoch = 1
    while epoch <= 10:

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
            # print(batch_idx, train_loss)
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
                             'net_struct': model.net_struct.tolist(),
                             'actfuns': model.actfuns,
                             'n_params': get_n_params(model)
                             })

        epoch += 1


def setup_experiment(seed, outfile_path):
    """
    Retrieves training / validation data, randomizes network structure and activation functions, creates model,
    creates new output file, sets hyperparameters for optimizer and scheduler during training, initializes training
    :param seed: seed for parameter randomization
    :param outfile_path: path to save outputs from experiment
    :return:
    """

    # ---- Loading MNIST
    trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (1.0,))])
    mnist_train_full = datasets.MNIST(root='./data', train=True, download=True, transform=trans)
    train_set_indices = np.arange(0, 50000)
    validation_set_indices = np.arange(50000, 60000)

    mnist_train = torch.utils.data.Subset(mnist_train_full, train_set_indices)
    mnist_validation = torch.utils.data.Subset(mnist_train_full, validation_set_indices)
    batch_size = 100
    train_loader = torch.utils.data.DataLoader(dataset=mnist_train, batch_size=batch_size, shuffle=True, pin_memory=True)
    validation_loader = torch.utils.data.DataLoader(dataset=mnist_validation, batch_size=batch_size, shuffle=True, pin_memory=True)

    # ---- Randomizing network structure & activation functions
    rng = np.random.RandomState(seed)
    num_hidden_layers = rng.randint(1, 4)
    net_struct = torch.zeros(num_hidden_layers, 4)
    all_actfuns = ['max', 'signed_geomean', 'swishk', 'l2', 'linf', 'lse', 'lae']
    actfuns = []
    while True:

        # For each layer, randomizes M, k, p, and g within given ranges
        for layer in range(num_hidden_layers):
            net_struct[layer, 0] = rng.randint(75, 120) * 2  # M
            net_struct[layer, 1] = rng.randint(2, 11)  # k
            net_struct[layer, 2] = rng.randint(1, 11)  # p
            # First layer doesn't get grouped
            if layer == 0:
                net_struct[layer, 3] = 1  # g
            else:
                net_struct[layer, 3] = rng.randint(1, 5)  # g
            # Adjust M so that it is divisible by g and k
            net_struct[layer, 0] = int(net_struct[layer, 0] / (net_struct[layer, 1] * net_struct[layer, 3])
                                       ) * net_struct[layer, 1] * net_struct[layer, 3]

        # For each layer, designate one activation function for each of its p permutations
        for layer in range(num_hidden_layers):
            layer_actfuns = []
            for perm in range(int(net_struct[layer, 2])):
                layer_actfuns.append(all_actfuns[rng.randint(0, len(all_actfuns))])
            actfuns.append(layer_actfuns)

        # Test to ensure the network structure is valid
        test = test_net_inputs(net_struct, actfuns, in_size=784, out_size=10)
        if test is None:
            break
        print(
            "\nInvalid network structure: \n{}\nError: {}\nTrying again..."
                .format(net_struct, test), flush=True
        )
        actfuns = []
        net_struct = torch.zeros(num_hidden_layers, 4)

    # ---- Create new model using randomized structure
    model = CombinactNet(net_struct=net_struct, actfuns=actfuns, in_size=784, out_size=10, batch_size=batch_size)
    if torch.cuda.is_available():
        model = model.cuda()

    print(
        "\nOutfile Path: {} \nNetwork Structure: \n{} \nActivation Functions: \n{} \nNumber of Parameters: {}\n\n"
            .format(outfile_path, net_struct, actfuns, get_n_params(model)), flush=True
    )

    # ---- Create new output file
    fieldnames = ['seed', 'epoch', 'train_loss', 'val_loss', 'acc', 'time', 'net_struct', 'actfuns', 'n_params']
    with open(outfile_path, mode='w') as out_file:
        writer = csv.DictWriter(out_file, fieldnames=fieldnames, lineterminator='\n')
        writer.writeheader()

    # ---- Use optimized hyperparams for l2 from previous random search
    # TODO: Optimize these parameters using another random search. Right now params are optimal for only l2
    hyper_params = {"adam_beta_1": 0.760516,
                    "adam_beta_2": 0.999983,
                    "adam_eps": 1.7936 * 10 ** -8,
                    "adam_wd": 1.33755 * 10 ** -5,
                    "max_lr": 0.0122491,
                    "cycle_peak": 0.234177
                    }

    # ---- Begin training model
    print("Running...")
    train_model(model, outfile_path, fieldnames, seed, train_loader, validation_loader, hyper_params)
    print()


if __name__ == '__main__':

    # ---- Handle running locally
    if len(sys.argv) == 1:
        seed_all(0)
        argv_seed = 3
        argv_outfile_path = str(datetime.date.today()) + "-combinact-" + str(argv_seed) + ".csv"

    # ---- Handle running on Vector
    else:
        seed_all(0)
        argv_index = int(sys.argv[1])
        argv_seed = int(sys.argv[2]) + (500 * argv_index)
        argv_outfile_path = sys.argv[3] + "/" + str(datetime.date.today()) + "-combinact-" + str(argv_seed) + ".csv"

    setup_experiment(argv_seed, argv_outfile_path)
