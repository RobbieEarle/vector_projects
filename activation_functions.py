import torch
import torch.nn.functional as F
from torch import logsumexp

import math
import numbers


def calc(actfun, x, layer, M_in, k_in, p_in, g_in, alpha_primes=None, alpha_dist=None):

    global M, k, p, g
    M, k, p, g = M_in, k_in, p_in, g_in

    if actfun == 'combinact':
        batch_size = x.shape[0]
        num_clusters = int(M / k) # Retrieve layer alpha primes
        layer_alphas = F.softmax(alpha_primes, dim=1)  # Convert alpha prime to alpha

        # Outputs collapse our k dimension to give [batch_size, num_clusters, p]
        # We repeat this process once for each activation function, hence the extra dim
        outputs = torch.zeros((batch_size, num_clusters, p, len(_COMBINACT_ACTFUNS)), device=x.device)
        for i, actfun in enumerate(_COMBINACT_ACTFUNS):
            outputs[:, :, :, i] = _ACTFUNS[actfun](x)

        # per_perm -> layer_alphas = [p, num_actfuns]
        if alpha_dist == "per_perm":
            outputs = outputs * layer_alphas  # Multiply elements in last 2 dims of outputs by layer_alphas
            outputs = torch.sum(outputs, dim=3)  # Sum across all actfuns

        # per_cluster -> layer_alphas = [num_clusters * p, num_actfuns]
        elif alpha_dist == "per_cluster":
            outputs = outputs.reshape([batch_size, num_clusters * p, len(_COMBINACT_ACTFUNS)])
            outputs = outputs * layer_alphas  # Multiply elements in last 2 dims of outputs by layer_alphas
            outputs = torch.sum(outputs, dim=2)  # Sum across all actfuns

        outputs = outputs.reshape([batch_size, num_clusters, p])
        return outputs

    elif actfun == "l2_lae":
        if layer == 0 or layer == 1:
            return _ACTFUNS['l2'](x)
        else:
            return _ACTFUNS['lae'](x)

    else:
        return _ACTFUNS[actfun](x)


def get_actfuns():
    return _ACTFUNS


def get_combinact_actfuns():
    return _COMBINACT_ACTFUNS


# -------------------- Activation Functions

_COMBINACT_ACTFUNS = ['max', 'signed_geomean', 'swishk', 'l1', 'l2', 'linf', 'lse', 'lae', 'min', 'nlsen', 'nlaen']
_ACTFUNS = {
    'relu':
        lambda z: F.relu_(z),
    'abs':
        lambda z: torch.abs_(z),
    'max':
        lambda z: torch.max(z, dim=2).values,
    'min':
        lambda z: torch.min(z, dim=2).values,
    'signed_geomean':
        lambda z: signed_geomean(z, dim=2),
    'swishk':
        lambda z: z[:, :, 0] * torch.exp(torch.sum(F.logsigmoid(z), dim=2)),
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
    'cf_relu':
        lambda z: coin_flip(z, 'relu'),
    'cf_abs':
        lambda z: coin_flip(z, 'abs'),
    'multi_relu':
        lambda z: multi_relu(z)
}
_ln2 = 0.6931471805599453
M, k, p, g = None, None, None, None


def coin_flip(z, actfun):
    shuffle_map = torch.empty(int(M / k), dtype=torch.long).random_(k)
    z = z[:, torch.arange(z.size(1)), shuffle_map]
    if actfun == 'relu':
        return F.relu_(z)
    elif actfun == 'abs':
        return torch.abs_(z)


def multi_relu(z):
    z = _ACTFUNS['max'](z)
    return F.relu(z)


def signed_geomean(z, dim=2):
    prod = torch.prod(z, dim=dim)
    signs = prod.sign()
    return signs * prod.abs().sqrt()


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