"""
Binary logic activation functions.
"""

from __future__ import division

import functools

import torch
from torch import nn


def maxd(z, dim):
    return torch.max(z, dim=dim).values


def mind(z, dim):
    return torch.min(z, dim=dim).values


def logistic_and_approx(z, dim):
    return torch.where(
        (z < 0).all(dim=dim), z.sum(dim=dim), torch.min(z, dim=dim).values,
    )


def logistic_or_approx(z, dim):
    return torch.where(
        (z > 0).all(dim=dim), z.sum(dim=dim), torch.max(z, dim=dim).values,
    )


def logistic_xnor_approx(z, dim):
    return torch.sign(torch.prod(z, dim=dim)) * torch.min(z.abs(), dim=dim).values


def logistic_or(z, dim):
    sig_neg_z = torch.sigmoid(-1 * z)
    sig_neg_z_prod = sig_neg_z.prod(dim=dim)
    p = torch.ones_like(sig_neg_z_prod) - sig_neg_z_prod
    return torch.log(p / (1-p))


def logistic_xnor(z, dim):
    sig_z = torch.sigmoid(z)
    sig_neg_z = torch.sigmoid(-1 * z)
    p = sig_z.prod(dim=dim) + sig_neg_z.prod(dim=dim)
    return torch.log(p / (1-p))


def unroll_k(x, k, d):
    if x.shape[d] % k != 0:
        raise ValueError(
            "Argument {} has shape {}. Dimension {} is {}, which is not"
            " divisible by {}.".format(x, x.shape, d, x.shape[d], k)
        )
    shp = list(x.shape)
    d_a = d % len(shp)
    shp = shp[:d_a] + [x.shape[d_a] // k] + [k] + shp[d_a + 1 :]
    x = x.view(*shp)
    d_new = d if d < 0 else d + 1
    return x, d_new


class HOActfun(nn.Module):
    def __init__(self, k=2, dim=1):
        super(HOActfun, self).__init__()
        self.k = k
        self.dim = dim

    @property
    def divisor(self):
        return self.k

    @property
    def feature_factor(self):
        return 1 / self.k


class CReLU(HOActfun):
    def forward(self, x):
        x = torch.cat((x,-x),1)
        return F.relu(x)

    @property
    def divisor(self):
        return 1

    @property
    def feature_factor(self):
        return 2


class MaxOut(HOActfun):
    def forward(self, x):
        x, d_new = unroll_k(x, self.k, self.dim)
        return torch.max(x, dim=d_new).values


class MinOut(HOActfun):
    def forward(self, x):
        x, d_new = unroll_k(x, self.k, self.dim)
        return torch.min(x, dim=d_new).values


class SignedGeomeanFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, dim, keepdim, clamp_grad):
        # Save inputs
        ctx.save_for_backward(input)
        ctx.dim = dim + input.ndim if dim < 0 else dim
        ctx.keepdim = keepdim
        ctx.clamp_grad = clamp_grad
        # Compute forward pass
        prods = input.prod(dim=dim, keepdim=keepdim)
        signs = prods.sign()
        output = signs * prods.abs().sqrt()
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        if not ctx.keepdim:
            grad_output = grad_output.unsqueeze(ctx.dim)

        # Re-compute forward pass
        prods = input.prod(dim=ctx.dim, keepdim=True)
        signs = prods.sign()
        output = signs * prods.abs().sqrt()

        grad_inner = 0.5 * output / input
        # Remove singularities
        grad_inner[input.abs() == 0] = 0
        # Clamp large values
        if ctx.clamp_grad is not None:
            grad_inner = torch.clamp(grad_inner, -ctx.clamp_grad, ctx.clamp_grad)
        # dy/dx = dy/dz * dz/dx
        grad_input = grad_output * grad_inner

        # Need to return None for each non-tensor input to forward
        return grad_input, None, None, None


def signed_geomean(x, dim=1, keepdim=False, clamp_grad=None):
    return SignedGeomeanFunc.apply(x, dim, keepdim, clamp_grad)


class SignedGeomean(HOActfun):
    def __init__(self, clamp_grad=None, *args, **kwargs):
        super(SignedGeomean, self).__init__(*args, **kwargs)
        self.clamp_grad = clamp_grad

    def forward(self, x):
        x, d_new = unroll_k(x, self.k, self.dim)
        return signed_geomean(x, d_new, clamp_grad=self.clamp_grad)


class AIL_AND(HOActfun):
    def forward(self, x):
        x, d_new = unroll_k(x, self.k, self.dim)
        return logistic_and_approx(x, d_new)


class AIL_OR(HOActfun):
    def forward(self, x):
        x, d_new = unroll_k(x, self.k, self.dim)
        return logistic_or_approx(x, d_new)


class AIL_XNOR(HOActfun):
    def forward(self, x):
        x, d_new = unroll_k(x, self.k, self.dim)
        return logistic_xnor_approx(x, d_new)


class IL_OR(HOActfun):
    def forward(self, x):
        x, d_new = unroll_k(x, self.k, self.dim)
        return logistic_or(x, d_new)


class IL_XNOR(HOActfun):
    def forward(self, x):
        x, d_new = unroll_k(x, self.k, self.dim)
        return logistic_xnor(x, d_new)


class MultiActfunDuplicate(HOActfun):
    def __init__(self, actfuns, **kwargs):
        super(MultiActfunDuplicate, self).__init__(**kwargs)
        self.actfuns = actfuns

    def forward(self, x):
        x, d_new = unroll_k(x, self.k, self.dim)
        return torch.cat([f(x, d_new) for f in self.actfuns], dim=self.dim)

    @property
    def feature_factor(self):
        return len(self.actfuns) / self.k


class MultiActfunPartition(HOActfun):
    def __init__(self, actfuns, **kwargs):
        super(MultiActfunPartition, self).__init__(**kwargs)
        self.actfuns = actfuns

    def forward(self, x):
        x, d_new = unroll_k(x, self.k, self.dim)
        xs = torch.split(x, len(self.actfuns), dim=d_new)
        return torch.cat(
            [f(xi, d_new) for f, xi in zip(self.actfuns, xs)], dim=self.dim
        )

    @property
    def divisor(self):
        return len(self.actfuns) * self.k


class max_min_duplicate(MultiActfunDuplicate):
    def __init__(self, **kwargs):
        super(max_min_duplicate, self).__init__([maxd, mind], **kwargs)


class AIL_AND_OR_duplicate(MultiActfunDuplicate):
    def __init__(self, **kwargs):
        super(AIL_AND_OR_duplicate, self).__init__([logistic_and_approx, logistic_or_approx], **kwargs)


class AIL_OR_XNOR_duplicate(MultiActfunDuplicate):
    def __init__(self, **kwargs):
        super(AIL_OR_XNOR_duplicate, self).__init__([logistic_or_approx, logistic_xnor_approx], **kwargs)


class AIL_AND_OR_XNOR_duplicate(MultiActfunDuplicate):
    def __init__(self, **kwargs):
        super(AIL_AND_OR_XNOR_duplicate, self).__init__(
            [logistic_and_approx, logistic_or_approx, logistic_xnor_approx], **kwargs
        )


class max_min_partition(MultiActfunPartition):
    def __init__(self, **kwargs):
        super(max_min_partition, self).__init__([maxd, mind], **kwargs)


class AIL_OR_XNOR_partition(MultiActfunPartition):
    def __init__(self, **kwargs):
        super(AIL_OR_XNOR_partition, self).__init__([logistic_or_approx, logistic_xnor_approx], **kwargs)


class AIL_AND_OR_XNOR_partition(MultiActfunPartition):
    def __init__(self, **kwargs):
        super(AIL_AND_OR_XNOR_partition, self).__init__(
            [logistic_and_approx, logistic_or_approx, logistic_xnor_approx], **kwargs
        )


def actfun_name2factory(name):
    name = name.lower()
    if name == "relu":
        return nn.ReLU
    elif name == "prelu":
        return nn.PReLU
    elif name == "crelu":
        return CReLU
    elif name in ("maxout", "max"):
        return MaxOut
    elif name == "signedgeomean":
        return SignedGeomean
    elif name == "signedgeomean_clamp2":
        return functools.partial(SignedGeomean, clamp_grad=2)
    elif name == "signedgeomean_clamp10":
        return functools.partial(SignedGeomean, clamp_grad=10)
    elif name == "ail_and":
        return AIL_AND
    elif name == "ail_or":
        return AIL_OR
    elif name == "ail_xnor":
        return AIL_XNOR
    elif name == "il_or":
        return IL_OR
    elif name == "il_xnor":
        return IL_XNOR
    elif name == "max_min_dup":
        return max_min_duplicate
    elif name == "ail_and_or_dup":
        return AIL_AND_OR_duplicate
    elif name == "ail_or_xnor_dup":
        return AIL_OR_XNOR_duplicate
    elif name == "ail_and_or_xnor_dup":
        return AIL_AND_OR_XNOR_duplicate
    elif name == "max_min_part":
        return max_min_partition
    elif name == "ail_or_xnor_part":
        return AIL_OR_XNOR_partition
    elif name == "ail_and_or_xnor_part":
        return AIL_AND_OR_XNOR_partition
    else:
        raise ValueError("Unsupported actfun: {}".format(name))
