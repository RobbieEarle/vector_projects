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
        assert p == 2, 'p must be 2 if you use the invert shuffle type ya big dummy.'
        assert k == 2, 'k must be 2 if you use the invert shuffle type ya big dummy.'
        assert actfun == 'swishk', 'SwishK is the only asymmetric actfun, so using other actfuns doesn\'t make sense!'

    # Unsqueeze a dimension and populate it with permutations of our inputs
    x = x.unsqueeze(2)
    for i in range(1, p):
        permutation = util.permute(x[:, :, 0, ...], permute_type, layer_type,
                                   offset=i, shuffle_map=shuffle_maps[i]).unsqueeze(2)
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

    bin_partition_actfuns = ['bin_partition_full', 'bin_partition_nopass']
    bin_all_actfuns = ['bin_all_full', 'bin_all_nopass_sgm', 'bin_all_nopass']
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


_ACTFUNS = {
    'combinact':
        lambda z: combinact(z),
    'relu':
        lambda z: F.relu_(z),
    'leaky_relu':
        lambda z: F.leaky_relu_(z),
    'abs':
        lambda z: torch.abs_(z),
    'swish':
        lambda z: z * torch.sigmoid(z),
    'prod':
        lambda z: torch.prod(z, dim=2),
    'max':
        lambda z: torch.max(z, dim=2).values,
    'min':
        lambda z: torch.min(z, dim=2).values,
    'signed_geomean':
        lambda z: sgm(z),
    'swishk':
        lambda z: z[:, :, 0] * torch.exp(torch.sum(F.logsigmoid(z), dim=2)),
    'swishk_p':
        lambda z: z[:, :, 0] * torch.exp(torch.sum(F.logsigmoid(z[:, :, 1:]), dim=2)),
    'l1':
        lambda z: (torch.sum(z.abs(), dim=2)),
    'l2':
        lambda z: (torch.sum(z.pow(2), dim=2)).sqrt_(),
    'l3-signed':
        lambda z: signed_l3(z),
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
        lambda z: multi_relu(z),
}
_ln2 = 0.6931471805599453


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
    bin_xor = None
    if actfun in bin_partition_actfuns:
        if actfun == 'bin_partition_full':
            partition = math.floor(z.shape[1] / 3)
            bin_and_or = torch.max(z[:, :partition, ...], dim=2).values
            bin_xor = sgm(z[:, partition: 2 * partition, ...])
            bin_pass = z[:, 2 * partition:, ...]
        elif actfun == 'bin_partition_nopass':
            partition = math.floor(z.shape[1] / 2)
            bin_and_or = torch.max(z[:, :partition, ...], dim=2).values
            bin_xor = sgm(z[:, partition:, ...])
    elif actfun in bin_all_actfuns:
        bin_and = torch.max(z, dim=2).values
        bin_or = torch.min(z, dim=2).values
        bin_and_or = torch.cat((bin_and, bin_or), dim=1)
        if actfun != 'bin_all_nopass_sgm':
            bin_xor = sgm(z)
            if actfun != 'bin_all_nopass':
                bin_pass = z

    if bin_xor is not None:
        if bin_pass is not None:
            if layer_type == 'conv':
                bin_pass = bin_pass.reshape(bin_pass.shape[0], bin_pass.shape[1] * bin_pass.shape[2],
                                            bin_pass.shape[3], bin_pass.shape[4])
            elif layer_type == 'linear':
                bin_pass = bin_pass.reshape(bin_pass.shape[0], bin_pass.shape[1] * bin_pass.shape[2])

            z = torch.cat((bin_and_or, bin_xor, bin_pass), dim=1)
        else:
            z = torch.cat((bin_and_or, bin_xor), dim=1)
    else:
        z = bin_and_or

    return z


sgm = SignedGeomean.apply
