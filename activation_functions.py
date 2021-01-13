import torch
import torch.nn.functional as F
from torch import logsumexp
import util

import math
import numbers
import time


def activate(x, actfun, p=1, k=1, M=None,
             layer_type='conv',
             permute_type='shuffle',
             shuffle_maps=None,
             alpha_primes=None,
             alpha_dist=None,
             reduce_actfuns=False
             ):

    if permute_type == 'invert':
        assert p % k == 0, 'k must divide p if you use the invert shuffle type ya big dummy.'

    # Unsqueeze a dimension and populate it with permutations of our inputs
    x = x.unsqueeze(2)
    curr_permute = permute_type
    permute_base = 0
    for i in range(1, p):
        curr_shuffle = shuffle_maps[i]
        if permute_type == 'invert':
            if i % k == 0:
                curr_permute = 'shuffle'
                permute_base = 0
            else:
                curr_permute = 'invert'
                permute_base = x.shape[2] - 1
                curr_shuffle = torch.arange(k)
                curr_shuffle[0] = i % k
                curr_shuffle[i % k] = 0
        permutation = util.permute(x[:, :, permute_base, ...], curr_permute, layer_type, k,
                                   offset=i, shuffle_map=curr_shuffle).unsqueeze(2)
        x = torch.cat((x[:, :, :i, ...], permutation), dim=2)

    # This transpose makes it so that during the next reshape (when we combine our p permutations
    # into a single dimension), the order goes one full permutation after another (instead of
    # interleaving the permutations)
    x = torch.transpose(x, dim0=1, dim1=2)

    # Combine p permutations into a single dimension, then cluster into groups of size k
    batch_size = x.shape[0]
    if layer_type == 'conv':
        num_channels = x.shape[2]
        height = x.shape[3]
        width = x.shape[4]
        x = x.reshape(batch_size, int(num_channels * p / k), k, height, width)
    elif layer_type == 'linear':
        num_channels = M
        x = x.reshape(batch_size, int(num_channels * p / k), k)

    bin_partition_actfuns = ['bin_part_full', 'bin_part_max_min_sgm', 'bin_part_max_sgm',
                             'ail_part_full', 'ail_part_or_and_xnor', 'ail_part_or_xnor',
                             'nail_part_full', 'nail_part_or_and_xnor', 'nail_part_or_xnor']
    bin_all_actfuns = ['bin_all_full', 'bin_all_max_min', 'bin_all_max_sgm', 'bin_all_max_min_sgm',
                       'ail_all_full', 'ail_all_or_and', 'ail_all_or_xnor', 'ail_all_or_and_xnor',
                       'nail_all_full', 'nail_all_or_and', 'nail_all_or_xnor', 'nail_all_or_and_xnor']

    if actfun == 'combinact':
        x = combinact(x,
                      p=p,
                      layer_type=layer_type,
                      alpha_primes=alpha_primes,
                      alpha_dist=alpha_dist,
                      reduce_actfuns=reduce_actfuns)
    elif actfun == 'cf_relu' or actfun == 'cf_abs':
        x = coin_flip(x, actfun, M=num_channels * p, k=k)
    elif actfun in bin_partition_actfuns or actfun in bin_all_actfuns:
        x = binary_ops(x, actfun, layer_type, bin_partition_actfuns, bin_all_actfuns)
    elif actfun == 'groupsort':
        x = groupsort(x, layer_type)
    else:
        x = x.squeeze()
        x = _ACTFUNS[actfun](x)

    return x


# -------------------- Activation Functions

_COMBINACT_ACTFUNS = ['max', 'swishk', 'l1', 'l2', 'linf', 'lse', 'lae', 'min', 'nlsen', 'nlaen', 'signed_geomean']
_COMBINACT_ACTFUNS_REDUCED = ['max', 'swishk', 'l2', 'lae', 'signed_geomean']


def get_combinact_actfuns(reduce_actfuns=False):
    if reduce_actfuns:
        return _COMBINACT_ACTFUNS_REDUCED
    else:
        return _COMBINACT_ACTFUNS


def logistic_and_approx(z):
    return torch.where(
        (z < 0).all(dim=2),
        z.sum(dim=2),
        torch.min(z, dim=2).values,
    )


def logistic_or_approx(z):
    return torch.where(
        (z > 0).all(dim=2),
        z.sum(dim=2),
        torch.max(z, dim=2).values,
    )


def logistic_xnor_approx(z):
    return torch.sign(torch.prod(z, dim=2)) * torch.min(z.abs(), dim=2).values


def logistic_and_approx_normalized(z):
    # divide by math.sqrt((1 + 2.5 * np.pi) / (2 * math.pi))
    return logistic_and_approx(z).mul_(0.8424043984960415)


def logistic_or_approx_normalized(z):
    # divide by math.sqrt((1 + 2.5 * np.pi) / (2 * math.pi))
    return logistic_or_approx(z).mul_(0.8424043984960415)


def logistic_xnor_approx_normalized(z):
    # divide by math.sqrt(1 - 2 / math.pi)
    return logistic_xnor_approx(z).mul_(1.658896739970306)

def combinact(x, p, layer_type='linear', alpha_primes=None, alpha_dist=None, reduce_actfuns=False):

    if reduce_actfuns:
        all_actfuns = _COMBINACT_ACTFUNS_REDUCED
    else:
        all_actfuns = _COMBINACT_ACTFUNS

    # Recording current input shape
    batch_size = x.shape[0]
    num_clusters = x.shape[1]
    img_size = x.shape[-1]

    # print(alpha_primes.shape)
    layer_alphas = F.softmax(alpha_primes, dim=1)  # Convert alpha prime to alpha

    # Computing all activation functions
    outputs = None
    for i, actfun in enumerate(all_actfuns):
        if i == 0:
            outputs = _ACTFUNS[actfun](x).to(x.device)
            outputs = outputs.unsqueeze(dim=2)
        else:
            outputs = torch.cat((outputs, _ACTFUNS[actfun](x).unsqueeze(dim=2)), dim=2)

    # Handling per-permutation alpha vector
    if alpha_dist == "per_perm":
        if layer_type == 'conv':
            layer_alphas = layer_alphas.reshape([1, 1,
                                                 layer_alphas.shape[0], layer_alphas.shape[1],
                                                 1, 1])
            outputs = outputs.reshape([batch_size, int(num_clusters / p), p,
                                       len(all_actfuns), img_size, img_size])

        elif layer_type == 'linear':
            layer_alphas = layer_alphas.reshape([1, 1,
                                                 layer_alphas.shape[0], layer_alphas.shape[1]])
            outputs = outputs.reshape([batch_size, int(num_clusters / p), p,
                                       len(all_actfuns)])
        outputs = outputs * layer_alphas  # Multiply elements in last 2 dims of outputs by layer_alphas
        outputs = torch.sum(outputs, dim=3)  # Sum across all actfuns

        if layer_type == 'conv':
            outputs = outputs.reshape([batch_size, num_clusters, img_size, img_size])
        elif layer_type == 'linear':
            outputs = outputs.reshape([batch_size, num_clusters])

    # Handling per-cluster alpha vector
    elif alpha_dist == "per_cluster":
        if layer_type == 'conv':
            layer_alphas = layer_alphas.reshape([1,
                                                 layer_alphas.shape[0], layer_alphas.shape[1],
                                                 1, 1])
        elif layer_type == 'linear':
            layer_alphas = layer_alphas.reshape([1,
                                                 layer_alphas.shape[0], layer_alphas.shape[1]])

        outputs = outputs * layer_alphas  # Multiply elements in last 2 dims of outputs by layer_alphas
        outputs = torch.sum(outputs, dim=2)  # Sum across all actfuns

    return outputs


def coin_flip(z, actfun, M, k):
    shuffle_map = torch.empty(int(M / k), dtype=torch.long).random_(k)
    z = z[:, torch.arange(z.size(1)), shuffle_map, ...]
    if actfun == 'cf_relu':
        return F.relu_(z)
    elif actfun == 'cf_abs':
        return torch.abs_(z)


def multi_relu(z):
    z = _ACTFUNS['max'](z)
    return F.relu(z)


def groupsort(z, layer_type):
    z = z.sort(dim=2, descending=True).values
    if layer_type == 'conv':
        z = z.reshape(z.shape[0], z.shape[1] * z.shape[2],
                             z.shape[3], z.shape[4])
    elif layer_type == 'linear':
        z = z.reshape(z.shape[0], z.shape[1] * z.shape[2])
    return z


class SignedGeomean(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        prod = torch.prod(input, dim=2)
        signs = prod.sign()
        return signs * prod.abs().sqrt()

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors

        prod = input.prod(dim=2).unsqueeze(2)
        signs = prod.sign()

        A = (prod.expand(input.shape) / input).abs().sqrt() * input
        B = 2 * input.abs().pow(3 / 2)

        grad_input = signs * A / B * grad_output.unsqueeze(2).expand(input.shape)
        grad_input[input.abs() == 0] = 0

        return grad_input


def signed_l3(z):
    x3 = z[:, :, 0].pow(3)
    y3 = z[:, :, 1].pow(3)
    out_val = (x3 + y3).tanh() * (x3 + y3).abs().pow(1 / 3)
    return out_val


def logavgexp(input, dim, keepdim=False, temperature=None, dtype=torch.float32):
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


def binary_ops(z, actfun, layer_type, bin_partition_actfuns, bin_all_actfuns):

    bin_pass = None

    if actfun in bin_partition_actfuns:
        if actfun == 'bin_part_full':
            partition = math.floor(z.shape[1] / 4)
            zs = [
                torch.max(z[:, :partition, ...], dim=2).values,
                torch.min(z[:, partition:2 * partition, ...], dim=2).values,
                sgm(z[:, 2 * partition: 3 * partition, ...]),
            ]
            bin_pass = z[:, 3 * partition:, ...]
        elif actfun == 'bin_part_max_min_sgm':
            partition = math.floor(z.shape[1] / 3)
            zs = [
                torch.max(z[:, :partition, ...], dim=2).values,
                torch.min(z[:, partition:2 * partition, ...], dim=2).values,
                sgm(z[:, 2 * partition:, ...]),
            ]
        elif actfun == 'bin_part_max_sgm':
            partition = math.floor(z.shape[1] / 2)
            zs = [
                torch.max(z[:, :partition, ...], dim=2).values,
                sgm(z[:, partition:, ...]),
            ]
        elif actfun == 'ail_part_full':
            partition = math.floor(z.shape[1] / 4)
            zs = [
                logistic_or_approx(z[:, :partition, ...]),
                logistic_and_approx(z[:, partition:2 * partition, ...]),
                logistic_xnor_approx(z[:, 2 * partition: 3 * partition, ...]),
            ]
            bin_pass = z[:, 3 * partition:, ...]
        elif actfun == 'nail_part_full':
            partition = math.floor(z.shape[1] / 4)
            zs = [
                logistic_or_approx_normalized(z[:, :partition, ...]),
                logistic_and_approx_normalized(z[:, partition:2 * partition, ...]),
                logistic_xnor_approx_normalized(z[:, 2 * partition: 3 * partition, ...]),
            ]
            bin_pass = z[:, 3 * partition:, ...]
        elif actfun == 'ail_part_or_and_xnor':
            partition = math.floor(z.shape[1] / 3)
            zs = [
                logistic_or_approx(z[:, :partition, ...]),
                logistic_and_approx(z[:, partition:2 * partition, ...]),
                logistic_xnor_approx(z[:, 2 * partition:, ...]),
            ]
        elif actfun == 'nail_part_or_and_xnor':
            partition = math.floor(z.shape[1] / 3)
            zs = [
                logistic_or_approx_normalized(z[:, :partition, ...]),
                logistic_and_approx_normalized(z[:, partition:2 * partition, ...]),
                logistic_xnor_approx_normalized(z[:, 2 * partition:, ...]),
            ]
        elif actfun == 'ail_part_or_xnor':
            partition = math.floor(z.shape[1] / 2)
            zs = [
                logistic_or_approx(z[:, :partition, ...]),
                logistic_xnor_approx(z[:, partition:, ...]),
            ]
        elif actfun == 'nail_part_or_xnor':
            partition = math.floor(z.shape[1] / 2)
            zs = [
                logistic_or_approx_normalized(z[:, :partition, ...]),
                logistic_xnor_approx_normalized(z[:, partition:, ...]),
            ]
    elif actfun in bin_all_actfuns:
        if actfun == 'bin_all_max_sgm':
            zs = [
                torch.max(z, dim=2).values,
                sgm(z),
            ]
        elif actfun == 'bin_all_max_min':
            zs = [
                torch.max(z, dim=2).values,
                torch.min(z, dim=2).values,
            ]
        elif actfun == 'bin_all_max_min_sgm':
            zs = [
                torch.max(z, dim=2).values,
                torch.min(z, dim=2).values,
                sgm(z),
            ]
        elif actfun == 'bin_all_full':
            zs = [
                torch.max(z, dim=2).values,
                torch.min(z, dim=2).values,
                sgm(z),
            ]
            bin_pass = z
        elif actfun == 'ail_all_or_xnor':
            zs = [
                logistic_or_approx(z),
                logistic_xnor_approx(z),
            ]
        elif actfun == 'nail_all_or_xnor':
            zs = [
                logistic_or_approx_normalized(z),
                logistic_xnor_approx_normalized(z),
            ]
        elif actfun == 'ail_all_or_and':
            zs = [
                logistic_or_approx(z),
                logistic_and_approx(z),
            ]
        elif actfun == 'nail_all_or_and':
            zs = [
                logistic_or_approx_normalized(z),
                logistic_and_approx_normalized(z),
            ]
        elif actfun == 'ail_all_or_and_xnor':
            zs = [
                logistic_or_approx(z),
                logistic_and_approx(z),
                logistic_xnor_approx(z),
            ]
        elif actfun == 'nail_all_or_and_xnor':
            zs = [
                logistic_or_approx_normalized(z),
                logistic_and_approx_normalized(z),
                logistic_xnor_approx_normalized(z),
            ]
        elif actfun == 'ail_all_full':
            zs = [
                logistic_or_approx(z),
                logistic_and_approx(z),
                logistic_xnor_approx(z),
            ]
            bin_pass = z
        elif actfun == 'nail_all_full':
            zs = [
                logistic_or_approx_normalized(z),
                logistic_and_approx_normalized(z),
                logistic_xnor_approx_normalized(z),
            ]
            bin_pass = z

    if bin_pass is not None:
        if layer_type == 'conv':
            bin_pass = bin_pass.reshape(bin_pass.shape[0], bin_pass.shape[1] * bin_pass.shape[2],
                                        bin_pass.shape[3], bin_pass.shape[4])
        elif layer_type == 'linear':
            bin_pass = bin_pass.reshape(bin_pass.shape[0], bin_pass.shape[1] * bin_pass.shape[2])
        zs.append(bin_pass)

    return torch.cat(zs, dim=1)


sgm = SignedGeomean.apply


_ln2 = 0.6931471805599453
_ACTFUNS = {
    'ail_and':
        logistic_and_approx,
    'ail_or':
        logistic_or_approx,
    'ail_xnor':
        logistic_xnor_approx,
    'nail_and':
        logistic_and_approx_normalized,
    'nail_or':
        logistic_or_approx_normalized,
    'nail_xnor':
        logistic_xnor_approx_normalized,
    'combinact':
        combinact,
    'relu':
        F.relu_,
    'nrelu':
        lambda z: F.relu_(z).mul_(1.414213562),
    'tanh':
        F.tanh,
    'leaky_relu':
        F.leaky_relu_,
    'abs':
        torch.abs_,
    'swish':
        lambda z: z * torch.sigmoid(z),
    'nswish':
        lambda z: z.mul_(torch.sigmoid(z)).mul_(1.676531339),
    'prod':
        lambda z: torch.prod(z, dim=2),
    'max':
        lambda z: torch.max(z, dim=2).values,
    'min':
        lambda z: torch.min(z, dim=2).values,
    'signed_geomean':
        sgm,
    'swishk':
        lambda z: z[:, :, 0] * torch.exp(torch.sum(F.logsigmoid(z), dim=2)),
    'swishy':
        lambda z: z[:, :, 0] * torch.exp(torch.sum(F.logsigmoid(z[:, :, 1:]), dim=2)),
    'l1':
        lambda z: (torch.sum(z.abs(), dim=2)),
    'l2':
        lambda z: (torch.sum(z.pow(2), dim=2)).sqrt_(),
    'l3-signed':
        signed_l3,
    'linf':
        lambda z: torch.max(z.abs(), dim=2).values,
    'lse':
        lambda z: torch.logsumexp(z, dim=2),
    'lae':
        lambda z: logavgexp(z, dim=2),
    'nlsen':
        lambda z: -1 * torch.logsumexp(-1 * z, dim=2),
    'nlaen':
        lambda z: -1 * logavgexp(-1 * z, dim=2),
    'lse-approx':
        lambda z: torch.max(z[:, :, 0], z[:, :, 1]) + torch.max(torch.tensor(0., device=z.device), _ln2 - 0.305 * (z[:, :, 0] - z[:, :, 1]).abs_()),
    'lae-approx':
        lambda z: torch.max(z[:, :, 0], z[:, :, 1]) + torch.max(torch.tensor(-_ln2, device=z.device), -0.305 * (z[:, :, 0] - z[:, :, 1]).abs_()),
    'nlsen-approx':
        lambda z: -torch.max(-z[:, :, 0], -z[:, :, 1]) - torch.max(torch.tensor(0., device=z.device), _ln2 - 0.305 * (z[:, :, 0] - z[:, :, 1]).abs_()),
    'nlaen-approx':
        lambda z: -torch.max(-z[:, :, 0], -z[:, :, 1]) - torch.max(torch.tensor(-_ln2, device=z.device), -0.305 * (z[:, :, 0] - z[:, :, 1]).abs_()),
    'multi_relu':
        multi_relu,
}
