# """EfficientNets
#
# Adapted under Apache License 2.0 from original by Ross Wightman, available at
# https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/gen_efficientnet.py
# """
#
# import math
# import re
# import logging
# from copy import deepcopy
#
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import torch.utils.model_zoo as model_zoo
#
# import torch_lse_pooling.pooling
# import torch_lse_pooling.adaptive_pooling
# import torch_lse_pooling.mixedpooling
#
# IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
# IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)
#
# __all__ = [
#     'efficientnet_b0',
#     'efficientnet_b1',
#     'efficientnet_b2',
#     'efficientnet_b3',
#     'efficientnet_b4',
#     'efficientnet_b5',
#     'tf_efficientnet_b0',
#     'tf_efficientnet_b1',
#     'tf_efficientnet_b2',
#     'tf_efficientnet_b3',
#     'tf_efficientnet_b4',
#     'tf_efficientnet_b5',
# ]
#
#
# def weight_decay_config(value=1e-4, log=False, is_temperature_decayed=True):
#     filters = {}
#     if is_temperature_decayed:
#         filters['parameter_name'] = lambda n: not n.endswith('bias')
#     else:
#         filters['parameter_name'] = lambda n: (not n.endswith('bias')) and (not n.endswith('temperature'))
#     filters['module'] = lambda m: not isinstance(m, nn.BatchNorm2d)
#
#     return {'name': 'WeightDecay',
#             'value': value,
#             'log': log,
#             'filter': filters,
#             }
#
#
# def ramp_up_lr(lr0, lrT, T):
#     rate = (lrT - lr0) / T
#     return "lambda t: {'lr': %s + t * %s}" % (lr0, rate)
#
#
# def ramp_up_down(peak, peak_step, final_step, initial=0, final=0, key='lr'):
#     return ("lambda t:"
#             " {{'{key}': max({base1}, {init} + {slope1} * t) if t <= {transition}"
#             " else max({base2}, {mid} + {slope2} * (t - {transition}))}}"
#         .format(
#         key=key,
#         init=initial,
#         slope1=(peak - initial) / peak_step,
#         transition=peak_step,
#         mid=peak,
#         slope2=(final - peak) / (final_step - peak_step),
#         base1=min(initial, peak),
#         base2=min(peak, final),
#     )
#     )
#
#
# def _cfg(url='', **kwargs):
#     return {
#         'url': url, 'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': (7, 7),
#         'crop_pct': 0.875, 'interpolation': 'bicubic',
#         'mean': IMAGENET_DEFAULT_MEAN, 'std': IMAGENET_DEFAULT_STD,
#         'first_conv': 'conv_stem', 'classifier': 'classifier',
#         **kwargs
#     }
#
#
# default_cfgs = {
#     'mnasnet_050': _cfg(url=''),
#     'mnasnet_075': _cfg(url=''),
#     'mnasnet_100': _cfg(
#         url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/mnasnet_b1-74cb7081.pth'),
#     'mnasnet_140': _cfg(url=''),
#     'semnasnet_050': _cfg(url=''),
#     'semnasnet_075': _cfg(url=''),
#     'semnasnet_100': _cfg(
#         url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/mnasnet_a1-d9418771.pth'),
#     'semnasnet_140': _cfg(url=''),
#     'mnasnet_small': _cfg(url=''),
#     'mobilenetv1_100': _cfg(url=''),
#     'mobilenetv2_100': _cfg(url=''),
#     'mobilenetv3_050': _cfg(url=''),
#     'mobilenetv3_075': _cfg(url=''),
#     'mobilenetv3_100': _cfg(
#         url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/mobilenetv3_100-35495452.pth'),
#     'chamnetv1_100': _cfg(url=''),
#     'chamnetv2_100': _cfg(url=''),
#     'fbnetc_100': _cfg(
#         url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/fbnetc_100-c345b898.pth',
#         interpolation='bilinear'),
#     'spnasnet_100': _cfg(
#         url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/spnasnet_100-048bc3f4.pth',
#         interpolation='bilinear'),
#     'efficientnet_b0': _cfg(
#         url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/efficientnet_b0-d6904d92.pth'),
#     'efficientnet_b1': _cfg(
#         url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/efficientnet_b1-533bc792.pth',
#         input_size=(3, 240, 240), pool_size=(8, 8), crop_pct=0.882),
#     'efficientnet_b2': _cfg(
#         url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/efficientnet_b2-cf78dc4d.pth',
#         input_size=(3, 260, 260), pool_size=(9, 9), crop_pct=0.890),
#     'efficientnet_b3': _cfg(
#         url='', input_size=(3, 300, 300), pool_size=(10, 10), crop_pct=0.904),
#     'efficientnet_b4': _cfg(
#         url='', input_size=(3, 380, 380), pool_size=(12, 12), crop_pct=0.922),
#     'efficientnet_b5': _cfg(
#         url='', input_size=(3, 456, 456), pool_size=(15, 15), crop_pct=0.934),
#     'tf_efficientnet_b0': _cfg(
#         url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/tf_efficientnet_b0-0af12548.pth',
#         input_size=(3, 224, 224)),
#     'tf_efficientnet_b1': _cfg(
#         url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/tf_efficientnet_b1-5c1377c4.pth',
#         input_size=(3, 240, 240), pool_size=(8, 8), crop_pct=0.882),
#     'tf_efficientnet_b2': _cfg(
#         url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/tf_efficientnet_b2-e393ef04.pth',
#         input_size=(3, 260, 260), pool_size=(9, 9), crop_pct=0.890),
#     'tf_efficientnet_b3': _cfg(
#         url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/tf_efficientnet_b3-e3bd6955.pth',
#         input_size=(3, 300, 300), pool_size=(10, 10), crop_pct=0.904),
#     'tf_efficientnet_b4': _cfg(
#         url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/tf_efficientnet_b4-74ee3bed.pth',
#         input_size=(3, 380, 380), pool_size=(12, 12), crop_pct=0.922),
#     'tf_efficientnet_b5': _cfg(
#         url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/tf_efficientnet_b5-c6949ce9.pth',
#         input_size=(3, 456, 456), pool_size=(15, 15), crop_pct=0.934),
#     'mixnet_s': _cfg(url=''),
#     'mixnet_m': _cfg(
#         url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/mixnet_m-4647fc68.pth'),
#     'mixnet_l': _cfg(url=''),
#     'tf_mixnet_s': _cfg(
#         url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/tf_mixnet_s-89d3354b.pth'),
#     'tf_mixnet_m': _cfg(
#         url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/tf_mixnet_m-0f4d8805.pth'),
#     'tf_mixnet_l': _cfg(
#         url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/tf_mixnet_l-6c92e0c8.pth'),
# }
#
# _DEBUG = False
#
# # Default args for PyTorch BN impl
# _BN_MOMENTUM_PT_DEFAULT = 0.1
# _BN_EPS_PT_DEFAULT = 1e-5
# _BN_ARGS_PT = dict(momentum=_BN_MOMENTUM_PT_DEFAULT, eps=_BN_EPS_PT_DEFAULT)
#
# # Defaults used for Google/Tensorflow training of mobile networks /w RMSprop as per
# # papers and TF reference implementations. PT momentum equiv for TF decay is (1 - TF decay)
# # NOTE: momentum varies btw .99 and .9997 depending on source
# # .99 in official TF TPU impl
# # .9997 (/w .999 in search space) for paper
# _BN_MOMENTUM_TF_DEFAULT = 1 - 0.99
# _BN_EPS_TF_DEFAULT = 1e-3
# _BN_ARGS_TF = dict(momentum=_BN_MOMENTUM_TF_DEFAULT, eps=_BN_EPS_TF_DEFAULT)
#
#
# # https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/helpers.py
# def load_pretrained(model, default_cfg, num_classes=1000, in_chans=3, filter_fn=None):
#     if 'url' not in default_cfg or not default_cfg['url']:
#         logging.warning("Pretrained model URL is invalid, using random initialization.")
#         return
#
#     state_dict = model_zoo.load_url(default_cfg['url'])
#
#     if in_chans == 1:
#         conv1_name = default_cfg['first_conv']
#         logging.info('Converting first conv (%s) from 3 to 1 channel' % conv1_name)
#         conv1_weight = state_dict[conv1_name + '.weight']
#         state_dict[conv1_name + '.weight'] = conv1_weight.sum(dim=1, keepdim=True)
#     elif in_chans != 3:
#         assert False, "Invalid in_chans for pretrained weights"
#
#     # strict = True
#     strict = False
#     classifier_name = default_cfg['classifier']
#     if num_classes == 1000 and default_cfg['num_classes'] == 1001:
#         # special case for imagenet trained models with extra background class in pretrained weights
#         classifier_weight = state_dict[classifier_name + '.weight']
#         state_dict[classifier_name + '.weight'] = classifier_weight[1:]
#         classifier_bias = state_dict[classifier_name + '.bias']
#         state_dict[classifier_name + '.bias'] = classifier_bias[1:]
#     elif num_classes != default_cfg['num_classes']:
#         # completely discard fully connected for all other differences between pretrained and created model
#         del state_dict[classifier_name + '.weight']
#         del state_dict[classifier_name + '.bias']
#         strict = False
#
#     if filter_fn is not None:
#         state_dict = filter_fn(state_dict)
#
#     model.load_state_dict(state_dict, strict=strict)
#
#
# ###
# # https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/adaptive_avgmax_pool.py
# def adaptive_pool_feat_mult(pool_type='avg'):
#     if pool_type == 'catavgmax':
#         return 2
#     else:
#         return 1
#
#
# def adaptive_avgmax_pool2d(x, output_size=1):
#     x_avg = F.adaptive_avg_pool2d(x, output_size)
#     x_max = F.adaptive_max_pool2d(x, output_size)
#     return 0.5 * (x_avg + x_max)
#
#
# def adaptive_catavgmax_pool2d(x, output_size=1):
#     x_avg = F.adaptive_avg_pool2d(x, output_size)
#     x_max = F.adaptive_max_pool2d(x, output_size)
#     return torch.cat((x_avg, x_max), 1)
#
#
# def select_adaptive_pool2d(x, pool_type='avg', output_size=1):
#     """Selectable global pooling function with dynamic input kernel size
#     """
#     if pool_type == 'avg':
#         x = F.adaptive_avg_pool2d(x, output_size)
#     elif pool_type == 'avgmax':
#         x = adaptive_avgmax_pool2d(x, output_size)
#     elif pool_type == 'catavgmax':
#         x = adaptive_catavgmax_pool2d(x, output_size)
#     elif pool_type == 'max':
#         x = F.adaptive_max_pool2d(x, output_size)
#     else:
#         assert False, 'Invalid pool type: %s' % pool_type
#     return x
#
#
# class AdaptiveAvgMaxPool2d(nn.Module):
#     def __init__(self, output_size=1):
#         super(AdaptiveAvgMaxPool2d, self).__init__()
#         self.output_size = output_size
#
#     def forward(self, x):
#         return adaptive_avgmax_pool2d(x, self.output_size)
#
#
# class AdaptiveCatAvgMaxPool2d(nn.Module):
#     def __init__(self, output_size=1):
#         super(AdaptiveCatAvgMaxPool2d, self).__init__()
#         self.output_size = output_size
#
#     def forward(self, x):
#         return adaptive_catavgmax_pool2d(x, self.output_size)
#
#
# class SelectAdaptivePool2d(nn.Module):
#     """Selectable global pooling layer with dynamic input kernel size
#     """
#
#     def __init__(self, output_size=1, pool_type='avg', num_chan=None):
#         super(SelectAdaptivePool2d, self).__init__()
#         self.output_size = output_size
#         self.pool_type = pool_type
#         if num_chan is None and '-chn' in pool_type:
#             raise ValueError('Need to know number of channels for pool type {}'.format(pool_type))
#         if pool_type == 'avgmax':
#             self.pool = AdaptiveAvgMaxPool2d(output_size)
#         elif pool_type == 'catavgmax':
#             self.pool = AdaptiveCatAvgMaxPool2d(output_size)
#         elif pool_type == 'max':
#             self.pool = nn.AdaptiveMaxPool2d(output_size)
#         elif pool_type == 'avg':
#             self.pool = nn.AdaptiveAvgPool2d(output_size)
#         elif pool_type in ('mixed', 'mixed-untrainable'):
#             self.pool = torch_lse_pooling.mixedpooling.MixedAvgMaxPool2d(
#                 spatial_size, trainable=False,
#             )
#         elif pool_type == 'mixed-trainable':
#             self.pool = torch_lse_pooling.mixedpooling.MixedAvgMaxPool2d(
#                 spatial_size, trainable=True,
#             )
#         elif pool_type == 'mixed-trainable-chn':
#             self.pool = torch_lse_pooling.mixedpooling.MixedAvgMaxPool2d(
#                 spatial_size, trainable=True, weight_per_channel=num_chan,
#             )
#         elif pool_type == 'mixed-gated':
#             self.pool = torch_lse_pooling.mixedpooling.GatedAvgMaxPool2d(
#                 spatial_size,
#             )
#         elif pool_type == 'mixed-gated-chn':
#             self.pool = torch_lse_pooling.mixedpooling.GatedAvgMaxPool2d(
#                 spatial_size, weight_per_channel=num_chan,
#             )
#         elif pool_type in ('lae', 'lae-untrainable'):
#             self.pool = torch_lse_pooling.adaptive_pooling.AdaptiveLogAvgExpPool2d(
#                 output_size, temp_trainable=False,
#             )
#         elif pool_type == 'lae-trainable':
#             self.pool = torch_lse_pooling.adaptive_pooling.AdaptiveLogAvgExpPool2d(
#                 output_size, temp_trainable=True,
#             )
#         elif pool_type == 'lae-trainable-chn':
#             self.pool = torch_lse_pooling.adaptive_pooling.AdaptiveLogAvgExpPool2d(
#                 output_size, temp_trainable=True, temp_per_channel=num_chan,
#             )
#         elif pool_type == 'lae-trainable-chn-init4':
#             self.pool = torch_lse_pooling.adaptive_pooling.AdaptiveLogAvgExpPool2d(
#                 output_size, temp_trainable=True, temp_per_channel=num_chan,
#                 temp_initial=4,
#             )
#         elif pool_type == 'lae-trainable-chn-init10':
#             self.pool = torch_lse_pooling.adaptive_pooling.AdaptiveLogAvgExpPool2d(
#                 output_size, temp_trainable=True, temp_per_channel=num_chan,
#                 temp_initial=10,
#             )
#         elif pool_type == 'lae-trainable-chn-init50':
#             self.pool = torch_lse_pooling.adaptive_pooling.AdaptiveLogAvgExpPool2d(
#                 output_size, temp_trainable=True, temp_per_channel=num_chan,
#                 temp_initial=50,
#             )
#         elif pool_type == 'lae-context':
#             self.pool = torch_lse_pooling.pooling.ContextLogAvgExpPool2d(
#                 spatial_size,
#             )
#         elif pool_type == 'lae-context-chn':
#             self.pool = torch_lse_pooling.pooling.ContextLogAvgExpPool2d(
#                 spatial_size, temp_per_channel=num_chan,
#             )
#         else:
#             raise ValueError('Invalid pool type: %s' % pool_type)
#
#     def forward(self, x):
#         return self.pool(x)
#
#     def feat_mult(self):
#         return adaptive_pool_feat_mult(self.pool_type)
#
#     def __repr__(self):
#         return self.__class__.__name__ + ' (' \
#                + 'output_size=' + str(self.output_size) \
#                + ', pool_type=' + self.pool_type + ')'
#
#
# ###
# # https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/conv2d_helpers.py
# def _is_static_pad(kernel_size, stride=1, dilation=1, **_):
#     return stride == 1 and (dilation * (kernel_size - 1)) % 2 == 0
#
#
# def _get_padding(kernel_size, stride=1, dilation=1, **_):
#     padding = ((stride - 1) + dilation * (kernel_size - 1)) // 2
#     return padding
#
#
# def _calc_same_pad(i, k, s, d):
#     return max((math.ceil(i / s) - 1) * s + (k - 1) * d + 1 - i, 0)
#
#
# def _split_channels(num_chan, num_groups):
#     split = [num_chan // num_groups for _ in range(num_groups)]
#     split[0] += num_chan - sum(split)
#     return split
#
#
# class Conv2dSame(nn.Conv2d):
#     """ Tensorflow like 'SAME' convolution wrapper for 2D convolutions
#     """
#
#     def __init__(self, in_channels, out_channels, kernel_size, stride=1,
#                  padding=0, dilation=1, groups=1, bias=True):
#         super(Conv2dSame, self).__init__(
#             in_channels, out_channels, kernel_size, stride, 0, dilation,
#             groups, bias)
#
#     def forward(self, x):
#         ih, iw = x.size()[-2:]
#         kh, kw = self.weight.size()[-2:]
#         pad_h = _calc_same_pad(ih, kh, self.stride[0], self.dilation[0])
#         pad_w = _calc_same_pad(iw, kw, self.stride[1], self.dilation[1])
#         if pad_h > 0 or pad_w > 0:
#             x = F.pad(x, [pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2])
#         return F.conv2d(x, self.weight, self.bias, self.stride,
#                         self.padding, self.dilation, self.groups)
#
#
# def conv2d_pad(in_chs, out_chs, kernel_size, **kwargs):
#     padding = kwargs.pop('padding', '')
#     kwargs.setdefault('bias', False)
#     if isinstance(padding, str):
#         # for any string padding, the padding will be calculated for you, one of three ways
#         padding = padding.lower()
#         if padding == 'same':
#             # TF compatible 'SAME' padding, has a performance and GPU memory allocation impact
#             if _is_static_pad(kernel_size, **kwargs):
#                 # static case, no extra overhead
#                 padding = _get_padding(kernel_size, **kwargs)
#                 return nn.Conv2d(in_chs, out_chs, kernel_size, padding=padding, **kwargs)
#             else:
#                 # dynamic padding
#                 return Conv2dSame(in_chs, out_chs, kernel_size, **kwargs)
#         elif padding == 'valid':
#             # 'VALID' padding, same as padding=0
#             return nn.Conv2d(in_chs, out_chs, kernel_size, padding=0, **kwargs)
#         else:
#             # Default to PyTorch style 'same'-ish symmetric padding
#             padding = _get_padding(kernel_size, **kwargs)
#             return nn.Conv2d(in_chs, out_chs, kernel_size, padding=padding, **kwargs)
#     else:
#         # padding was specified as a number or pair
#         return nn.Conv2d(in_chs, out_chs, kernel_size, padding=padding, **kwargs)
#
#
# class MixedConv2d(nn.Module):
#     """ Mixed Grouped Convolution
#     Based on MDConv and GroupedConv in MixNet impl:
#       https://github.com/tensorflow/tpu/blob/master/models/official/mnasnet/mixnet/custom_layers.py
#     """
#
#     def __init__(self, in_channels, out_channels, kernel_size=3,
#                  stride=1, padding='', dilated=False, depthwise=False, **kwargs):
#         super(MixedConv2d, self).__init__()
#
#         kernel_size = kernel_size if isinstance(kernel_size, list) else [kernel_size]
#         num_groups = len(kernel_size)
#         in_splits = _split_channels(in_channels, num_groups)
#         out_splits = _split_channels(out_channels, num_groups)
#         for idx, (k, in_ch, out_ch) in enumerate(zip(kernel_size, in_splits, out_splits)):
#             d = 1
#             # FIXME make compat with non-square kernel/dilations/strides
#             if stride == 1 and dilated:
#                 d, k = (k - 1) // 2, 3
#             conv_groups = out_ch if depthwise else 1
#             # use add_module to keep key space clean
#             self.add_module(
#                 str(idx),
#                 conv2d_pad(
#                     in_ch, out_ch, k, stride=stride,
#                     padding=padding, dilation=d, groups=conv_groups, **kwargs)
#             )
#         self.splits = in_splits
#
#     def forward(self, x):
#         x_split = torch.split(x, self.splits, 1)
#         x_out = [c(x) for x, c in zip(x_split, self._modules.values())]
#         x = torch.cat(x_out, 1)
#         return x
#
#
# # helper method
# def select_conv2d(in_chs, out_chs, kernel_size, **kwargs):
#     assert 'groups' not in kwargs  # only use 'depthwise' bool arg
#     if isinstance(kernel_size, list):
#         # We're going to use only lists for defining the MixedConv2d kernel groups,
#         # ints, tuples, other iterables will continue to pass to normal conv and specify h, w.
#         return MixedConv2d(in_chs, out_chs, kernel_size, **kwargs)
#     else:
#         depthwise = kwargs.pop('depthwise', False)
#         groups = out_chs if depthwise else 1
#         return conv2d_pad(in_chs, out_chs, kernel_size, groups=groups, **kwargs)
#
#
# # main
# def _resolve_bn_args(kwargs):
#     bn_args = _BN_ARGS_TF.copy() if kwargs.pop('bn_tf', False) else _BN_ARGS_PT.copy()
#     bn_momentum = kwargs.pop('bn_momentum', None)
#     if bn_momentum is not None:
#         bn_args['momentum'] = bn_momentum
#     bn_eps = kwargs.pop('bn_eps', None)
#     if bn_eps is not None:
#         bn_args['eps'] = bn_eps
#     return bn_args
#
#
# def _round_channels(channels, multiplier=1.0, divisor=8, channel_min=None):
#     """Round number of filters based on depth multiplier."""
#     if not multiplier:
#         return channels
#
#     channels *= multiplier
#     channel_min = channel_min or divisor
#     new_channels = max(
#         int(channels + divisor / 2) // divisor * divisor,
#         channel_min)
#     # Make sure that round down does not go down by more than 10%.
#     if new_channels < 0.9 * channels:
#         new_channels += divisor
#     return new_channels
#
#
# def _parse_ksize(ss):
#     if ss.isdigit():
#         return int(ss)
#     else:
#         return [int(k) for k in ss.split('.')]
#
#
# def _decode_block_str(block_str, depth_multiplier=1.0):
#     """ Decode block definition string
#
#     Gets a list of block arg (dicts) through a string notation of arguments.
#     E.g. ir_r2_k3_s2_e1_i32_o16_se0.25_noskip
#
#     All args can exist in any order with the exception of the leading string which
#     is assumed to indicate the block type.
#
#     leading string - block type (
#       ir = InvertedResidual, ds = DepthwiseSep, dsa = DeptwhiseSep with pw act, cn = ConvBnAct)
#     r - number of repeat blocks,
#     k - kernel size,
#     s - strides (1-9),
#     e - expansion ratio,
#     c - output channels,
#     se - squeeze/excitation ratio
#     n - activation fn ('re', 'r6', 'hs', or 'sw')
#     Args:
#         block_str: a string representation of block arguments.
#     Returns:
#         A list of block args (dicts)
#     Raises:
#         ValueError: if the string def not properly specified (TODO)
#     """
#     assert isinstance(block_str, str)
#     ops = block_str.split('_')
#     block_type = ops[0]  # take the block type off the front
#     ops = ops[1:]
#     options = {}
#     noskip = False
#     for op in ops:
#         # string options being checked on individual basis, combine if they grow
#         if op == 'noskip':
#             noskip = True
#         elif op.startswith('n'):
#             # activation fn
#             key = op[0]
#             v = op[1:]
#             if v == 're':
#                 value = F.relu
#             elif v == 'r6':
#                 value = F.relu6
#             elif v == 'hs':
#                 value = hard_swish
#             elif v == 'sw':
#                 value = swish
#             else:
#                 continue
#             options[key] = value
#         else:
#             # all numeric options
#             splits = re.split(r'(\d.*)', op)
#             if len(splits) >= 2:
#                 key, value = splits[:2]
#                 options[key] = value
#
#     # if act_fn is None, the model default (passed to model init) will be used
#     act_fn = options['n'] if 'n' in options else None
#     exp_kernel_size = _parse_ksize(options['a']) if 'a' in options else 1
#     pw_kernel_size = _parse_ksize(options['p']) if 'p' in options else 1
#
#     num_repeat = int(options['r'])
#     # each type of block has different valid arguments, fill accordingly
#     if block_type == 'ir':
#         block_args = dict(
#             block_type=block_type,
#             dw_kernel_size=_parse_ksize(options['k']),
#             exp_kernel_size=exp_kernel_size,
#             pw_kernel_size=pw_kernel_size,
#             out_chs=int(options['c']),
#             exp_ratio=float(options['e']),
#             se_ratio=float(options['se']) if 'se' in options else None,
#             stride=int(options['s']),
#             act_fn=act_fn,
#             noskip=noskip,
#         )
#     elif block_type == 'ds' or block_type == 'dsa':
#         block_args = dict(
#             block_type=block_type,
#             dw_kernel_size=_parse_ksize(options['k']),
#             pw_kernel_size=pw_kernel_size,
#             out_chs=int(options['c']),
#             se_ratio=float(options['se']) if 'se' in options else None,
#             stride=int(options['s']),
#             act_fn=act_fn,
#             pw_act=block_type == 'dsa',
#             noskip=block_type == 'dsa' or noskip,
#         )
#     elif block_type == 'cn':
#         block_args = dict(
#             block_type=block_type,
#             kernel_size=int(options['k']),
#             out_chs=int(options['c']),
#             stride=int(options['s']),
#             act_fn=act_fn,
#         )
#     else:
#         assert False, 'Unknown block type (%s)' % block_type
#
#     # return a list of block args expanded by num_repeat and
#     # scaled by depth_multiplier
#     num_repeat = int(math.ceil(num_repeat * depth_multiplier))
#     return [deepcopy(block_args) for _ in range(num_repeat)]
#
#
# def _decode_arch_def(arch_def, depth_multiplier=1.0):
#     arch_args = []
#     for stack_idx, block_strings in enumerate(arch_def):
#         assert isinstance(block_strings, list)
#         stack_args = []
#         for block_str in block_strings:
#             assert isinstance(block_str, str)
#             stack_args.extend(_decode_block_str(block_str, depth_multiplier))
#         arch_args.append(stack_args)
#     return arch_args
#
#
# def swish(x, inplace=False):
#     if inplace:
#         return x.mul_(x.sigmoid())
#     else:
#         return x * x.sigmoid()
#
#
# def sigmoid(x, inplace=False):
#     return x.sigmoid_() if inplace else x.sigmoid()
#
#
# def hard_swish(x, inplace=False):
#     if inplace:
#         return x.mul_(F.relu6(x + 3.) / 6.)
#     else:
#         return x * F.relu6(x + 3.) / 6.
#
#
# def hard_sigmoid(x, inplace=False):
#     if inplace:
#         return x.add_(3.).clamp_(0., 6.).div_(6.)
#     else:
#         return F.relu6(x + 3.) / 6.
#
#
# class _BlockBuilder:
#     """ Build Trunk Blocks
#
#     This ended up being somewhat of a cross between
#     https://github.com/tensorflow/tpu/blob/master/models/official/mnasnet/mnasnet_models.py
#     and
#     https://github.com/facebookresearch/maskrcnn-benchmark/blob/master/maskrcnn_benchmark/modeling/backbone/fbnet_builder.py
#
#     """
#
#     def __init__(self, channel_multiplier=1.0, channel_divisor=8, channel_min=None,
#                  pad_type='', act_fn=None, se_gate_fn=sigmoid, se_reduce_mid=False,
#                  bn_args=_BN_ARGS_PT, drop_connect_rate=0., verbose=False,
#                  se_pool_type='avg'):
#         self.channel_multiplier = channel_multiplier
#         self.channel_divisor = channel_divisor
#         self.channel_min = channel_min
#         self.pad_type = pad_type
#         self.act_fn = act_fn
#         self.se_gate_fn = se_gate_fn
#         self.se_reduce_mid = se_reduce_mid
#         self.se_pool_type = se_pool_type
#         self.bn_args = bn_args
#         self.drop_connect_rate = drop_connect_rate
#         self.verbose = verbose
#
#         # updated during build
#         self.in_chs = None
#         self.block_idx = 0
#         self.block_count = 0
#
#     def _round_channels(self, chs):
#         return _round_channels(chs, self.channel_multiplier, self.channel_divisor, self.channel_min)
#
#     def _make_block(self, ba):
#         bt = ba.pop('block_type')
#         ba['in_chs'] = self.in_chs
#         ba['out_chs'] = self._round_channels(ba['out_chs'])
#         ba['bn_args'] = self.bn_args
#         ba['pad_type'] = self.pad_type
#         # block act fn overrides the model default
#         ba['act_fn'] = ba['act_fn'] if ba['act_fn'] is not None else self.act_fn
#         ba['se_pool_type'] = self.se_pool_type
#         assert ba['act_fn'] is not None
#         if bt == 'ir':
#             ba['drop_connect_rate'] = self.drop_connect_rate * self.block_idx / self.block_count
#             ba['se_gate_fn'] = self.se_gate_fn
#             ba['se_reduce_mid'] = self.se_reduce_mid
#             if self.verbose:
#                 logging.info('  InvertedResidual {}, Args: {}'.format(self.block_idx, str(ba)))
#             block = InvertedResidual(**ba)
#         elif bt == 'ds' or bt == 'dsa':
#             ba['drop_connect_rate'] = self.drop_connect_rate * self.block_idx / self.block_count
#             if self.verbose:
#                 logging.info('  DepthwiseSeparable {}, Args: {}'.format(self.block_idx, str(ba)))
#             block = DepthwiseSeparableConv(**ba)
#         elif bt == 'cn':
#             if self.verbose:
#                 logging.info('  ConvBnAct {}, Args: {}'.format(self.block_idx, str(ba)))
#             block = ConvBnAct(**ba)
#         else:
#             assert False, 'Uknkown block type (%s) while building model.' % bt
#         self.in_chs = ba['out_chs']  # update in_chs for arg of next block
#
#         return block
#
#     def _make_stack(self, stack_args):
#         blocks = []
#         # each stack (stage) contains a list of block arguments
#         for i, ba in enumerate(stack_args):
#             if self.verbose:
#                 logging.info(' Block: {}'.format(i))
#             if i >= 1:
#                 # only the first block in any stack can have a stride > 1
#                 ba['stride'] = 1
#             block = self._make_block(ba)
#             blocks.append(block)
#             self.block_idx += 1  # incr global idx (across all stacks)
#         return nn.Sequential(*blocks)
#
#     def __call__(self, in_chs, block_args):
#         """ Build the blocks
#         Args:
#             in_chs: Number of input-channels passed to first block
#             block_args: A list of lists, outer list defines stages, inner
#                 list contains strings defining block configuration(s)
#         Return:
#              List of block stacks (each stack wrapped in nn.Sequential)
#         """
#         if self.verbose:
#             logging.info('Building model trunk with %d stages...' % len(block_args))
#         self.in_chs = in_chs
#         self.block_count = sum([len(x) for x in block_args])
#         self.block_idx = 0
#         blocks = []
#         # outer list of block_args defines the stacks ('stages' by some conventions)
#         for stack_idx, stack in enumerate(block_args):
#             if self.verbose:
#                 logging.info('Stack: {}'.format(stack_idx))
#             assert isinstance(stack, list)
#             stack = self._make_stack(stack)
#             blocks.append(stack)
#         return blocks
#
#
# def _initialize_weight_goog(m):
#     # weight init as per Tensorflow Official impl
#     # https://github.com/tensorflow/tpu/blob/master/models/official/mnasnet/mnasnet_model.py
#     if isinstance(m, nn.Conv2d):
#         n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels  # fan-out
#         m.weight.data.normal_(0, math.sqrt(2.0 / n))
#         if m.bias is not None:
#             m.bias.data.zero_()
#     elif isinstance(m, nn.BatchNorm2d):
#         m.weight.data.fill_(1.0)
#         m.bias.data.zero_()
#     elif isinstance(m, nn.Linear):
#         n = m.weight.size(0)  # fan-out
#         init_range = 1.0 / math.sqrt(n)
#         m.weight.data.uniform_(-init_range, init_range)
#         m.bias.data.zero_()
#
#
# def _initialize_weight_default(m):
#     if isinstance(m, nn.Conv2d):
#         nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
#     elif isinstance(m, nn.BatchNorm2d):
#         m.weight.data.fill_(1.0)
#         m.bias.data.zero_()
#     elif isinstance(m, nn.Linear):
#         nn.init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='linear')
#
#
# def drop_connect(inputs, training=False, drop_connect_rate=0.):
#     """Apply drop connect."""
#     if not training:
#         return inputs
#
#     keep_prob = 1 - drop_connect_rate
#     random_tensor = keep_prob + torch.rand(
#         (inputs.size()[0], 1, 1, 1), dtype=inputs.dtype, device=inputs.device)
#     random_tensor.floor_()  # binarize
#     output = inputs.div(keep_prob) * random_tensor
#     return output
#
#
# class ChannelShuffle(nn.Module):
#     # FIXME haven't used yet
#     def __init__(self, groups):
#         super(ChannelShuffle, self).__init__()
#         self.groups = groups
#
#     def forward(self, x):
#         """Channel shuffle: [N,C,H,W] -> [N,g,C/g,H,W] -> [N,C/g,g,H,w] -> [N,C,H,W]"""
#         N, C, H, W = x.size()
#         g = self.groups
#         assert C % g == 0, "Incompatible group size {} for input channel {}".format(
#             g, C
#         )
#         return (
#             x.view(N, g, int(C / g), H, W)
#                 .permute(0, 2, 1, 3, 4)
#                 .contiguous()
#                 .view(N, C, H, W)
#         )
#
#
# class SqueezeExcite(nn.Module):
#     def __init__(self, in_chs, reduce_chs=None, act_fn=F.relu, gate_fn=sigmoid, pool_type='avg'):
#         super(SqueezeExcite, self).__init__()
#         self.act_fn = act_fn
#         self.gate_fn = gate_fn
#         reduced_chs = reduce_chs or in_chs
#         self.conv_reduce = nn.Conv2d(in_chs, reduced_chs, 1, bias=True)
#         self.conv_expand = nn.Conv2d(reduced_chs, in_chs, 1, bias=True)
#
#         if pool_type == 'avg':
#             self.pool = nn.AdaptiveAvgPool2d(1)
#         elif pool_type == 'lae':
#             self.pool = torch_lse_pooling.adaptive_pooling.AdaptiveLogAvgExpPool2d(1)
#         elif pool_type == 'lae-trainable':
#             self.pool = torch_lse_pooling.adaptive_pooling.AdaptiveLogAvgExpPool2d(
#                 1, temp_trainable=True,
#             )
#         elif pool_type == 'lae-trainable-chn':
#             self.pool = torch_lse_pooling.adaptive_pooling.AdaptiveLogAvgExpPool2d(
#                 1, temp_trainable=True, temp_per_channel=in_chs,
#             )
#         elif pool_type == 'lae-trainable-chn-init4':
#             self.pool = torch_lse_pooling.adaptive_pooling.AdaptiveLogAvgExpPool2d(
#                 1, temp_trainable=True, temp_per_channel=in_chs,
#                 temp_initial=4,
#             )
#         elif pool_type == 'lae-trainable-chn-init10':
#             self.pool = torch_lse_pooling.adaptive_pooling.AdaptiveLogAvgExpPool2d(
#                 1, temp_trainable=True, temp_per_channel=in_chs,
#                 temp_initial=10,
#             )
#         elif pool_type == 'lae-trainable-chn-init50':
#             self.pool = torch_lse_pooling.adaptive_pooling.AdaptiveLogAvgExpPool2d(
#                 1, temp_trainable=True, temp_per_channel=in_chs,
#                 temp_initial=50,
#             )
#         else:
#             raise ValueError('Incorrect pool type specified: {}'.format(pool))
#
#     def forward(self, x):
#         # NOTE adaptiveavgpool can be used here, but seems to cause issues with NVIDIA AMP performance
#         # x_se = x.view(x.size(0), x.size(1), -1).mean(-1)
#         x_se = self.pool(x)
#         x_se = x_se.view(x.size(0), x.size(1), 1, 1)
#         x_se = self.conv_reduce(x_se)
#         x_se = self.act_fn(x_se, inplace=True)
#         x_se = self.conv_expand(x_se)
#         x = x * self.gate_fn(x_se)
#         return x
#
#
# class ConvBnAct(nn.Module):
#     def __init__(self, in_chs, out_chs, kernel_size,
#                  stride=1, pad_type='', act_fn=F.relu, bn_args=_BN_ARGS_PT):
#         super(ConvBnAct, self).__init__()
#         assert stride in [1, 2]
#         self.act_fn = act_fn
#         self.conv = select_conv2d(in_chs, out_chs, kernel_size, stride=stride, padding=pad_type)
#         self.bn1 = nn.BatchNorm2d(out_chs, **bn_args)
#
#     def forward(self, x):
#         x = self.conv(x)
#         x = self.bn1(x)
#         x = self.act_fn(x, inplace=True)
#         return x
#
#
# class DepthwiseSeparableConv(nn.Module):
#     """ DepthwiseSeparable block
#     Used for DS convs in MobileNet-V1 and in the place of IR blocks with an expansion
#     factor of 1.0. This is an alternative to having a IR with optional first pw conv.
#     """
#
#     def __init__(self, in_chs, out_chs, dw_kernel_size=3,
#                  stride=1, pad_type='', act_fn=F.relu, noskip=False,
#                  pw_kernel_size=1, pw_act=False,
#                  se_ratio=0., se_gate_fn=sigmoid, se_pool_type='avg',
#                  bn_args=_BN_ARGS_PT, drop_connect_rate=0.):
#         super(DepthwiseSeparableConv, self).__init__()
#         assert stride in [1, 2]
#         self.has_se = se_ratio is not None and se_ratio > 0.
#         self.has_residual = (stride == 1 and in_chs == out_chs) and not noskip
#         self.has_pw_act = pw_act  # activation after point-wise conv
#         self.act_fn = act_fn
#         self.drop_connect_rate = drop_connect_rate
#
#         self.conv_dw = select_conv2d(
#             in_chs, in_chs, dw_kernel_size, stride=stride, padding=pad_type, depthwise=True)
#         self.bn1 = nn.BatchNorm2d(in_chs, **bn_args)
#
#         # Squeeze-and-excitation
#         if self.has_se:
#             self.se = SqueezeExcite(
#                 in_chs, reduce_chs=max(1, int(in_chs * se_ratio)), act_fn=act_fn, gate_fn=se_gate_fn,
#                 pool_type=se_pool_type)
#
#         self.conv_pw = select_conv2d(in_chs, out_chs, pw_kernel_size, padding=pad_type)
#         self.bn2 = nn.BatchNorm2d(out_chs, **bn_args)
#
#     def forward(self, x):
#         residual = x
#
#         x = self.conv_dw(x)
#         x = self.bn1(x)
#         x = self.act_fn(x, inplace=True)
#
#         if self.has_se:
#             x = self.se(x)
#
#         x = self.conv_pw(x)
#         x = self.bn2(x)
#         if self.has_pw_act:
#             x = self.act_fn(x, inplace=True)
#
#         if self.has_residual:
#             if self.drop_connect_rate > 0.:
#                 x = drop_connect(x, self.training, self.drop_connect_rate)
#             x += residual
#         return x
#
#
# class InvertedResidual(nn.Module):
#     """ Inverted residual block w/ optional SE"""
#
#     def __init__(self, in_chs, out_chs, dw_kernel_size=3,
#                  stride=1, pad_type='', act_fn=F.relu, noskip=False,
#                  exp_ratio=1.0, exp_kernel_size=1, pw_kernel_size=1,
#                  se_ratio=0., se_reduce_mid=False, se_gate_fn=sigmoid, se_pool_type='avg',
#                  shuffle_type=None, bn_args=_BN_ARGS_PT, drop_connect_rate=0.):
#         super(InvertedResidual, self).__init__()
#         mid_chs = int(in_chs * exp_ratio)
#         self.has_se = se_ratio is not None and se_ratio > 0.
#         self.has_residual = (in_chs == out_chs and stride == 1) and not noskip
#         self.act_fn = act_fn
#         self.drop_connect_rate = drop_connect_rate
#
#         # Point-wise expansion
#         self.conv_pw = select_conv2d(in_chs, mid_chs, exp_kernel_size, padding=pad_type)
#         self.bn1 = nn.BatchNorm2d(mid_chs, **bn_args)
#
#         self.shuffle_type = shuffle_type
#         if shuffle_type is not None and isinstance(exp_kernel_size, list):
#             self.shuffle = ChannelShuffle(len(exp_kernel_size))
#
#         # Depth-wise convolution
#         self.conv_dw = select_conv2d(
#             mid_chs, mid_chs, dw_kernel_size, stride=stride, padding=pad_type, depthwise=True)
#         self.bn2 = nn.BatchNorm2d(mid_chs, **bn_args)
#
#         # Squeeze-and-excitation
#         if self.has_se:
#             se_base_chs = mid_chs if se_reduce_mid else in_chs
#             self.se = SqueezeExcite(
#                 mid_chs, reduce_chs=max(1, int(se_base_chs * se_ratio)), act_fn=act_fn, gate_fn=se_gate_fn,
#                 pool_type=se_pool_type)
#
#         # Point-wise linear projection
#         self.conv_pwl = select_conv2d(mid_chs, out_chs, pw_kernel_size, padding=pad_type)
#         self.bn3 = nn.BatchNorm2d(out_chs, **bn_args)
#
#     def forward(self, x):
#         residual = x
#
#         # Point-wise expansion
#         x = self.conv_pw(x)
#         x = self.bn1(x)
#         x = self.act_fn(x, inplace=True)
#
#         # FIXME haven't tried this yet
#         # for channel shuffle when using groups with pointwise convs as per FBNet variants
#         if self.shuffle_type == "mid":
#             x = self.shuffle(x)
#
#         # Depth-wise convolution
#         x = self.conv_dw(x)
#         x = self.bn2(x)
#         x = self.act_fn(x, inplace=True)
#
#         # Squeeze-and-excitation
#         if self.has_se:
#             x = self.se(x)
#
#         # Point-wise linear projection
#         x = self.conv_pwl(x)
#         x = self.bn3(x)
#
#         if self.has_residual:
#             if self.drop_connect_rate > 0.:
#                 x = drop_connect(x, self.training, self.drop_connect_rate)
#             x += residual
#
#         # NOTE maskrcnn_benchmark building blocks have an SE module defined here for some variants
#
#         return x
#
#
# class GenEfficientNet(nn.Module):
#     """ Generic EfficientNet
#
#     An implementation of efficient network architectures, in many cases mobile optimized networks:
#       * MobileNet-V1
#       * MobileNet-V2
#       * MobileNet-V3
#       * MnasNet A1, B1, and small
#       * FBNet A, B, and C
#       * ChamNet (arch details are murky)
#       * Single-Path NAS Pixel1
#       * EfficientNet B0-B5
#       * MixNet S, M, L
#     """
#
#     def __init__(self, block_args, num_classes=1000, in_chans=3, stem_size=32, num_features=1280,
#                  channel_multiplier=1.0, channel_divisor=8, channel_min=None,
#                  pad_type='', act_fn=F.relu, drop_rate=0., drop_connect_rate=0.,
#                  se_gate_fn=sigmoid, se_reduce_mid=False, se_pool_type='avg',
#                  bn_args=_BN_ARGS_PT,
#                  global_pool='avg', head_conv='default', weight_init='goog'):
#         super(GenEfficientNet, self).__init__()
#         self.num_classes = num_classes
#         self.drop_rate = drop_rate
#         self.act_fn = act_fn
#         self.num_features = num_features
#
#         stem_size = _round_channels(stem_size, channel_multiplier, channel_divisor, channel_min)
#         self.conv_stem = select_conv2d(in_chans, stem_size, 3, stride=2, padding=pad_type)
#         self.bn1 = nn.BatchNorm2d(stem_size, **bn_args)
#         in_chs = stem_size
#
#         builder = _BlockBuilder(
#             channel_multiplier, channel_divisor, channel_min,
#             pad_type, act_fn, se_gate_fn, se_reduce_mid,
#             bn_args, drop_connect_rate, verbose=_DEBUG, se_pool_type=se_pool_type)
#         self.blocks = nn.Sequential(*builder(in_chs, block_args))
#         in_chs = builder.in_chs
#
#         if not head_conv or head_conv == 'none':
#             self.efficient_head = False
#             self.conv_head = None
#             assert in_chs == self.num_features
#         else:
#             self.efficient_head = head_conv == 'efficient'
#             self.conv_head = select_conv2d(in_chs, self.num_features, 1, padding=pad_type)
#             self.bn2 = None if self.efficient_head else nn.BatchNorm2d(self.num_features, **bn_args)
#
#         self.global_pool = SelectAdaptivePool2d(pool_type=global_pool, num_chan=self.num_features)
#         self.classifier = nn.Linear(self.num_features * self.global_pool.feat_mult(), self.num_classes)
#
#         for m in self.modules():
#             if weight_init == 'goog':
#                 _initialize_weight_goog(m)
#             else:
#                 _initialize_weight_default(m)
#
#     def get_classifier(self):
#         return self.classifier
#
#     def reset_classifier(self, num_classes, global_pool='avg'):
#         self.global_pool = SelectAdaptivePool2d(pool_type=global_pool, num_chan=self.num_features)
#         self.num_classes = num_classes
#         del self.classifier
#         if num_classes:
#             self.classifier = nn.Linear(
#                 self.num_features * self.global_pool.feat_mult(), num_classes)
#         else:
#             self.classifier = None
#
#     def forward_features(self, x, pool=True):
#         x = self.conv_stem(x)
#         x = self.bn1(x)
#         x = self.act_fn(x, inplace=True)
#         x = self.blocks(x)
#         if self.efficient_head:
#             # efficient head, currently only mobilenet-v3 performs pool before last 1x1 conv
#             x = self.global_pool(x)  # always need to pool here regardless of flag
#             x = self.conv_head(x)
#             # no BN
#             x = self.act_fn(x, inplace=True)
#             if pool:
#                 # expect flattened output if pool is true, otherwise keep dim
#                 x = x.view(x.size(0), -1)
#         else:
#             if self.conv_head is not None:
#                 x = self.conv_head(x)
#                 x = self.bn2(x)
#             x = self.act_fn(x, inplace=True)
#             if pool:
#                 x = self.global_pool(x)
#                 x = x.view(x.size(0), -1)
#         return x
#
#     def forward(self, x):
#         x = self.forward_features(x)
#         if self.drop_rate > 0.:
#             x = F.dropout(x, p=self.drop_rate, training=self.training)
#         return self.classifier(x)
#
#
# def _gen_efficientnet(channel_multiplier=1.0, depth_multiplier=1.0,
#                       num_classes=1000, regime='imagenet_normal', scale_lr=1,
#                       is_temperature_decayed=True, dataset='imagenet',
#                       batch_size=None, epochs=None, peak_epoch=5,
#                       custom_regime_lr=0.1, custom_regime_lr_decay_factor=0.1,
#                       custom_regime_momentum=0.9, custom_regime_wd=1e-4,
#                       **kwargs):
#     """Creates an EfficientNet model.
#
#     Ref impl: https://github.com/tensorflow/tpu/blob/master/models/official/efficientnet/efficientnet_model.py
#     Paper: https://arxiv.org/abs/1905.11946
#
#     EfficientNet params
#     name: (channel_multiplier, depth_multiplier, resolution, dropout_rate)
#     'efficientnet-b0': (1.0, 1.0, 224, 0.2),
#     'efficientnet-b1': (1.0, 1.1, 240, 0.2),
#     'efficientnet-b2': (1.1, 1.2, 260, 0.3),
#     'efficientnet-b3': (1.2, 1.4, 300, 0.3),
#     'efficientnet-b4': (1.4, 1.8, 380, 0.4),
#     'efficientnet-b5': (1.6, 2.2, 456, 0.4),
#     'efficientnet-b6': (1.8, 2.6, 528, 0.5),
#     'efficientnet-b7': (2.0, 3.1, 600, 0.5),
#
#     Args:
#       channel_multiplier: multiplier to number of channels per layer
#       depth_multiplier: multiplier to number of repeats per stage
#
#     """
#     arch_def = [
#         ['ds_r1_k3_s1_e1_c16_se0.25'],
#         ['ir_r2_k3_s2_e6_c24_se0.25'],
#         ['ir_r2_k5_s2_e6_c40_se0.25'],
#         ['ir_r3_k3_s2_e6_c80_se0.25'],
#         ['ir_r3_k5_s1_e6_c112_se0.25'],
#         ['ir_r4_k5_s2_e6_c192_se0.25'],
#         ['ir_r1_k3_s1_e6_c320_se0.25'],
#     ]
#     # NOTE: other models in the family didn't scale the feature count
#     num_features = _round_channels(1280, channel_multiplier, 8, None)
#     model = GenEfficientNet(
#         _decode_arch_def(arch_def, depth_multiplier),
#         num_classes=num_classes,
#         stem_size=32,
#         channel_multiplier=channel_multiplier,
#         channel_divisor=8,
#         channel_min=None,
#         num_features=num_features,
#         bn_args=_resolve_bn_args(kwargs),
#         act_fn=swish,
#         **kwargs
#     )
#
#     if regime in {'paper', 'custom-linear'}:
#         if batch_size is None:
#             raise ValueError('Need to know batch_size for {} regime'.format(regime))
#         if 'imagenette' in dataset:
#             n_samples = 12894
#         elif 'imagewoof' in dataset:
#             n_samples = 12454
#         elif 'imagenet' in dataset:
#             n_samples = 1281167
#         else:
#             raise ValueError('Unrecognised dataset: {}'.format(dataset))
#         steps_per_epoch = int(n_samples / batch_size)
#
#     if regime == 'paper':
#         def config_by_epoch(epoch):
#             return {'lr': scale_lr * 0.016 * (0.97 ** round(epoch / 2.4))}
#
#         """RMSProp optimizer with
#         decay 0.9 and momentum 0.9;
#         weight decay 1e-5; initial learning rate 0.256 that decays
#         by 0.97 every 2.4 epochs"""
#         self.regime = [
#             {'optimizer': 'RMSprop',
#              'alpha': 0.9,
#              'momentum': 0.9,
#              'lr': 0.0,
#              'regularizer': weight_decay_config(1e-5, is_temperature_decayed=is_temperature_decayed),
#              'step_lambda': ramp_up_lr(0.1, 0.1 * scale_lr, steps_per_epoch * 5 / scale_lr)},
#         }
#         {'epoch': 5,
#          'epoch_lambda': config_by_epoch,
#          },
#     ]
#     elif regime == 'custom-linear':
#     if epochs is None:
#         raise ValueError('Need to know number of epochs for custom-linear regime')
#     model.regime = [
#         {'epoch': 0, 'optimizer': 'SGD', 'lr': custom_regime_lr, 'momentum': custom_regime_momentum,
#          'regularizer': weight_decay_config(custom_regime_wd, is_temperature_decayed=is_temperature_decayed),
#          'step_lambda': ramp_up_down(
#              custom_regime_lr,
#              steps_per_epoch * peak_epoch,
#              steps_per_epoch * epochs,
#          ),
#          },
#     ]
#
# elif regime == 'cifar_normal':
# model.regime = [
#     {'epoch': 0, 'optimizer': 'SGD', 'lr': 1e-1, 'momentum': 0.9,
#      'regularizer': weight_decay_config(1e-4, is_temperature_decayed=is_temperature_decayed)},
#     {'epoch': 81, 'lr': 1e-2},
#     {'epoch': 122, 'lr': 1e-3},
#     {'epoch': 164, 'lr': 1e-4}
# ]
# elif regime == 'wide-resnet':
# model.regime = [
#     {'epoch': 0, 'optimizer': 'SGD', 'lr': 1e-1, 'momentum': 0.9,
#      'regularizer': weight_decay_config(5e-4, is_temperature_decayed=is_temperature_decayed)},
#     {'epoch': 60, 'lr': 2e-2},
#     {'epoch': 120, 'lr': 4e-3},
#     {'epoch': 160, 'lr': 8e-4}
# ]
# elif regime == 'shakedrop_paper':
# model.regime = [
#     {'epoch': 0, 'optimizer': 'SGD', 'lr': 5e-1, 'momentum': 0.9,
#      'regularizer': weight_decay_config(1e-4, is_temperature_decayed=is_temperature_decayed)},
#     {'epoch': 150, 'lr': 5e-2},
#     {'epoch': 225, 'lr': 5e-3},
# ]
# elif regime == 'pyramidnet_v3':
# model.regime = [
#     {'epoch': 0, 'optimizer': 'SGD', 'lr': 1e-1, 'momentum': 0.9,
#      'regularizer': weight_decay_config(1e-4, is_temperature_decayed=is_temperature_decayed)},
#     {'epoch': 150, 'lr': 1e-2},
#     {'epoch': 225, 'lr': 1e-3},
# ]
# elif regime == 'imagenet_normal':
# model.regime = [
#     {'epoch': 0, 'optimizer': 'SGD', 'momentum': 0.9,
#      'regularizer': weight_decay_config(1e-4, is_temperature_decayed=is_temperature_decayed),
#      'step_lambda': ramp_up_lr(0.1, 0.1 * scale_lr, 5004 * 5 / scale_lr)},
#     {'epoch': 5, 'lr': scale_lr * 1e-1},
#     {'epoch': 30, 'lr': scale_lr * 1e-2},
#     {'epoch': 60, 'lr': scale_lr * 1e-3},
#     {'epoch': 80, 'lr': scale_lr * 1e-4}
# ]
# elif regime == 'imagenet_fast':
# model.regime = [
#     {'epoch': 0, 'optimizer': 'SGD', 'momentum': 0.9,
#      'regularizer': weight_decay_config(1e-4, is_temperature_decayed=is_temperature_decayed),
#      'step_lambda': ramp_up_lr(0.1, 0.1 * 4 * scale_lr, 5004 * 4 / (4 * scale_lr))},
#     {'epoch': 4, 'lr': 4 * scale_lr * 1e-1},
#     {'epoch': 18, 'lr': scale_lr * 1e-1},
#     {'epoch': 21, 'lr': scale_lr * 1e-2},
#     {'epoch': 35, 'lr': scale_lr * 1e-3},
#     {'epoch': 43, 'lr': scale_lr * 1e-4},
# ]
# self.data_regime = [
#     {'epoch': 0, 'input_size': 128, 'batch_size': 256},
#     {'epoch': 18, 'input_size': 224, 'batch_size': 64},
#     {'epoch': 41, 'input_size': 288, 'batch_size': 32},
# ]
# elif 'small' in regime:
# if regime == 'small_half':
#     bs_factor = 2
# else:
#     bs_factor = 1
# scale_lr *= 4 * bs_factor
# model.regime = [
#     {'epoch': 0, 'optimizer': 'SGD',
#      'regularizer': weight_decay_config(1e-4, is_temperature_decayed=is_temperature_decayed),
#      'momentum': 0.9, 'lr': scale_lr * 1e-1},
#     {'epoch': 30, 'lr': scale_lr * 1e-2},
#     {'epoch': 60, 'lr': scale_lr * 1e-3},
#     {'epoch': 80, 'lr': bs_factor * 1e-4}
# ]
# self.data_regime = [
#     {'epoch': 0, 'input_size': 128, 'batch_size': 256 * bs_factor},
#     {'epoch': 80, 'input_size': 224, 'batch_size': 64 * bs_factor},
# ]
# self.data_eval_regime = [
#     {'epoch': 0, 'input_size': 224, 'batch_size': 512 * bs_factor},
# ]
# elif regime == 'finetune':
# model.regime = [
#     {'epoch': 0, 'optimizer': 'RMSprop', 'alpha': 0.9, 'momentum': 0.9,
#      'regularizer': weight_decay_config(1e-5, is_temperature_decayed=is_temperature_decayed),
#      'lr': scale_lr * 1e-7,
#      },
# ]
# elif regime == 'finetune2':
# model.regime = [
#     {'epoch': 0, 'optimizer': 'RMSprop', 'alpha': 0.997, 'momentum': 0.997,
#      'regularizer': weight_decay_config(1e-5, is_temperature_decayed=is_temperature_decayed),
#      'lr': scale_lr * 1e-7,
#      },
# ]
# else:
# raise ValueError('Unsupported regime: {}'.format(regime))
# return model
#
#
# def efficientnet_b0(
#         pretrained=False, num_classes=None, in_chans=3,
#         dataset='imagenet', training=True, **kwargs,
# ):
#     """ EfficientNet-B0 """
#     if num_classes is None:
#         if 'imagenette' in dataset or 'imagewoof' in dataset:
#             num_classes = 10
#         elif 'imagenet' in dataset:
#             num_classes = 1000
#         elif dataset == 'cifar10':
#             num_classes = 10
#         elif dataset == 'cifar100':
#             num_classes = 100
#         else:
#             raise ValueError('Unfamiliar dataset: {}'.format(dataset))
#     kwargs.setdefault('regime', 'imagenet_normal' if dataset == 'imagenet' else 'cifar_normal')
#     if training:
#         kwargs.setdefault('drop_connect_rate', 0.2)
#         kwargs.setdefault('drop_rate', 0.2)
#     default_cfg = default_cfgs['efficientnet_b0']
#     model = _gen_efficientnet(
#         channel_multiplier=1.0, depth_multiplier=1.0,
#         num_classes=num_classes, in_chans=in_chans, dataset=dataset, **kwargs)
#     model.default_cfg = default_cfg
#     if pretrained:
#         load_pretrained(model, default_cfg, num_classes, in_chans)
#     return model
#
#
# def efficientnet_b1(
#         pretrained=False, num_classes=None, in_chans=3,
#         dataset='imagenet', training=True, **kwargs,
# ):
#     """ EfficientNet-B1 """
#     if num_classes is None:
#         if 'imagenette' in dataset or 'imagewoof' in dataset:
#             num_classes = 10
#         elif 'imagenet' in dataset:
#             num_classes = 1000
#         elif dataset == 'cifar10':
#             num_classes = 10
#         elif dataset == 'cifar100':
#             num_classes = 100
#         else:
#             raise ValueError('Unfamiliar dataset: {}'.format(dataset))
#     kwargs.setdefault('regime', 'imagenet_normal' if dataset == 'imagenet' else 'cifar_normal')
#     if training:
#         kwargs.setdefault('drop_connect_rate', 0.2)
#         kwargs.setdefault('drop_rate', 0.2)
#     default_cfg = default_cfgs['efficientnet_b1']
#     model = _gen_efficientnet(
#         channel_multiplier=1.0, depth_multiplier=1.1,
#         num_classes=num_classes, in_chans=in_chans, dataset=dataset, **kwargs)
#     model.default_cfg = default_cfg
#     if pretrained:
#         load_pretrained(model, default_cfg, num_classes, in_chans)
#     return model
#
#
# def efficientnet_b2(
#         pretrained=False, num_classes=None, in_chans=3,
#         dataset='imagenet', training=True, **kwargs,
# ):
#     """ EfficientNet-B2 """
#     if num_classes is None:
#         if 'imagenette' in dataset or 'imagewoof' in dataset:
#             num_classes = 10
#         elif 'imagenet' in dataset:
#             num_classes = 1000
#         elif dataset == 'cifar10':
#             num_classes = 10
#         elif dataset == 'cifar100':
#             num_classes = 100
#         else:
#             raise ValueError('Unfamiliar dataset: {}'.format(dataset))
#     kwargs.setdefault('regime', 'imagenet_normal' if dataset == 'imagenet' else 'cifar_normal')
#     if training:
#         kwargs.setdefault('drop_connect_rate', 0.2)
#         kwargs.setdefault('drop_rate', 0.3)
#     default_cfg = default_cfgs['efficientnet_b2']
#     model = _gen_efficientnet(
#         channel_multiplier=1.1, depth_multiplier=1.2,
#         num_classes=num_classes, in_chans=in_chans, dataset=dataset, **kwargs)
#     model.default_cfg = default_cfg
#     if pretrained:
#         load_pretrained(model, default_cfg, num_classes, in_chans)
#     return model
#
#
# def efficientnet_b3(
#         pretrained=False, num_classes=None, in_chans=3,
#         dataset='imagenet', training=True, **kwargs,
# ):
#     """ EfficientNet-B3 """
#     if num_classes is None:
#         if 'imagenette' in dataset or 'imagewoof' in dataset:
#             num_classes = 10
#         elif 'imagenet' in dataset:
#             num_classes = 1000
#         elif dataset == 'cifar10':
#             num_classes = 10
#         elif dataset == 'cifar100':
#             num_classes = 100
#         else:
#             raise ValueError('Unfamiliar dataset: {}'.format(dataset))
#     kwargs.setdefault('regime', 'imagenet_normal' if dataset == 'imagenet' else 'cifar_normal')
#     if training:
#         kwargs.setdefault('drop_connect_rate', 0.2)
#         kwargs.setdefault('drop_rate', 0.3)
#     default_cfg = default_cfgs['efficientnet_b3']
#     model = _gen_efficientnet(
#         channel_multiplier=1.2, depth_multiplier=1.4,
#         num_classes=num_classes, in_chans=in_chans, dataset=dataset, **kwargs)
#     model.default_cfg = default_cfg
#     if pretrained:
#         load_pretrained(model, default_cfg, num_classes, in_chans)
#     return model
#
#
# def efficientnet_b4(
#         pretrained=False, num_classes=None, in_chans=3,
#         dataset='imagenet', training=True, **kwargs,
# ):
#     """ EfficientNet-B4 """
#     if num_classes is None:
#         if 'imagenette' in dataset or 'imagewoof' in dataset:
#             num_classes = 10
#         elif 'imagenet' in dataset:
#             num_classes = 1000
#         elif dataset == 'cifar10':
#             num_classes = 10
#         elif dataset == 'cifar100':
#             num_classes = 100
#         else:
#             raise ValueError('Unfamiliar dataset: {}'.format(dataset))
#     kwargs.setdefault('regime', 'imagenet_normal' if dataset == 'imagenet' else 'cifar_normal')
#     if training:
#         kwargs.setdefault('drop_connect_rate', 0.2)
#         kwargs.setdefault('drop_rate', 0.4)
#     default_cfg = default_cfgs['efficientnet_b4']
#     model = _gen_efficientnet(
#         channel_multiplier=1.4, depth_multiplier=1.8,
#         num_classes=num_classes, in_chans=in_chans, dataset=dataset, **kwargs)
#     model.default_cfg = default_cfg
#     if pretrained:
#         load_pretrained(model, default_cfg, num_classes, in_chans)
#     return model
#
#
# def efficientnet_b5(
#         pretrained=False, num_classes=None, in_chans=3,
#         dataset='imagenet', training=True, **kwargs,
# ):
#     """ EfficientNet-B5 """
#     if num_classes is None:
#         if 'imagenette' in dataset or 'imagewoof' in dataset:
#             num_classes = 10
#         elif 'imagenet' in dataset:
#             num_classes = 1000
#         elif dataset == 'cifar10':
#             num_classes = 10
#         elif dataset == 'cifar100':
#             num_classes = 100
#         else:
#             raise ValueError('Unfamiliar dataset: {}'.format(dataset))
#     kwargs.setdefault('regime', 'imagenet_normal' if dataset == 'imagenet' else 'cifar_normal')
#     if training:
#         kwargs.setdefault('drop_connect_rate', 0.2)
#         kwargs.setdefault('drop_rate', 0.4)
#     default_cfg = default_cfgs['efficientnet_b5']
#     model = _gen_efficientnet(
#         channel_multiplier=1.6, depth_multiplier=2.2,
#         num_classes=num_classes, in_chans=in_chans, dataset=dataset, **kwargs)
#     model.default_cfg = default_cfg
#     if pretrained:
#         load_pretrained(model, default_cfg, num_classes, in_chans)
#     return model
#
#
# def tf_efficientnet_b0(
#         pretrained=False, num_classes=None, in_chans=3,
#         dataset='imagenet', training=True, **kwargs,
# ):
#     """ EfficientNet-B0. Tensorflow compatible variant  """
#     if num_classes is None:
#         if 'imagenette' in dataset or 'imagewoof' in dataset:
#             num_classes = 10
#         elif 'imagenet' in dataset:
#             num_classes = 1000
#         elif dataset == 'cifar10':
#             num_classes = 10
#         elif dataset == 'cifar100':
#             num_classes = 100
#         else:
#             raise ValueError('Unfamiliar dataset: {}'.format(dataset))
#     kwargs.setdefault('regime', 'imagenet_normal' if dataset == 'imagenet' else 'cifar_normal')
#     if training:
#         kwargs.setdefault('drop_connect_rate', 0.2)
#         kwargs.setdefault('drop_rate', 0.2)
#     default_cfg = default_cfgs['tf_efficientnet_b0']
#     kwargs['bn_eps'] = _BN_EPS_TF_DEFAULT
#     kwargs['pad_type'] = 'same'
#     model = _gen_efficientnet(
#         channel_multiplier=1.0, depth_multiplier=1.0,
#         num_classes=num_classes, in_chans=in_chans, dataset=dataset, **kwargs)
#     model.default_cfg = default_cfg
#     if pretrained:
#         load_pretrained(model, default_cfg, num_classes, in_chans)
#     return model
#
#
# def tf_efficientnet_b1(
#         pretrained=False, num_classes=None, in_chans=3,
#         dataset='imagenet', training=True, **kwargs,
# ):
#     """ EfficientNet-B1. Tensorflow compatible variant  """
#     if num_classes is None:
#         if 'imagenette' in dataset or 'imagewoof' in dataset:
#             num_classes = 10
#         elif 'imagenet' in dataset:
#             num_classes = 1000
#         elif dataset == 'cifar10':
#             num_classes = 10
#         elif dataset == 'cifar100':
#             num_classes = 100
#         else:
#             raise ValueError('Unfamiliar dataset: {}'.format(dataset))
#     kwargs.setdefault('regime', 'imagenet_normal' if dataset == 'imagenet' else 'cifar_normal')
#     if training:
#         kwargs.setdefault('drop_connect_rate', 0.2)
#         kwargs.setdefault('drop_rate', 0.2)
#     default_cfg = default_cfgs['tf_efficientnet_b1']
#     kwargs['bn_eps'] = _BN_EPS_TF_DEFAULT
#     kwargs['pad_type'] = 'same'
#     model = _gen_efficientnet(
#         channel_multiplier=1.0, depth_multiplier=1.1,
#         num_classes=num_classes, in_chans=in_chans, dataset=dataset, **kwargs)
#     model.default_cfg = default_cfg
#     if pretrained:
#         load_pretrained(model, default_cfg, num_classes, in_chans)
#     return model
#
#
# def tf_efficientnet_b2(
#         pretrained=False, num_classes=None, in_chans=3,
#         dataset='imagenet', training=True, **kwargs,
# ):
#     """ EfficientNet-B2. Tensorflow compatible variant  """
#     if num_classes is None:
#         if 'imagenette' in dataset or 'imagewoof' in dataset:
#             num_classes = 10
#         elif 'imagenet' in dataset:
#             num_classes = 1000
#         elif dataset == 'cifar10':
#             num_classes = 10
#         elif dataset == 'cifar100':
#             num_classes = 100
#         else:
#             raise ValueError('Unfamiliar dataset: {}'.format(dataset))
#     kwargs.setdefault('regime', 'imagenet_normal' if dataset == 'imagenet' else 'cifar_normal')
#     if training:
#         kwargs.setdefault('drop_connect_rate', 0.2)
#         kwargs.setdefault('drop_rate', 0.3)
#     default_cfg = default_cfgs['tf_efficientnet_b2']
#     kwargs['bn_eps'] = _BN_EPS_TF_DEFAULT
#     kwargs['pad_type'] = 'same'
#     model = _gen_efficientnet(
#         channel_multiplier=1.1, depth_multiplier=1.2,
#         num_classes=num_classes, in_chans=in_chans, dataset=dataset, **kwargs)
#     model.default_cfg = default_cfg
#     if pretrained:
#         load_pretrained(model, default_cfg, num_classes, in_chans)
#     return model
#
#
# def tf_efficientnet_b3(
#         pretrained=False, num_classes=None, in_chans=3,
#         dataset='imagenet', training=True, **kwargs,
# ):
#     """ EfficientNet-B3. Tensorflow compatible variant """
#     if num_classes is None:
#         if 'imagenette' in dataset or 'imagewoof' in dataset:
#             num_classes = 10
#         elif 'imagenet' in dataset:
#             num_classes = 1000
#         elif dataset == 'cifar10':
#             num_classes = 10
#         elif dataset == 'cifar100':
#             num_classes = 100
#         else:
#             raise ValueError('Unfamiliar dataset: {}'.format(dataset))
#     kwargs.setdefault('regime', 'imagenet_normal' if dataset == 'imagenet' else 'cifar_normal')
#     if training:
#         kwargs.setdefault('drop_connect_rate', 0.2)
#         kwargs.setdefault('drop_rate', 0.3)
#     default_cfg = default_cfgs['tf_efficientnet_b3']
#     kwargs['bn_eps'] = _BN_EPS_TF_DEFAULT
#     kwargs['pad_type'] = 'same'
#     model = _gen_efficientnet(
#         channel_multiplier=1.2, depth_multiplier=1.4,
#         num_classes=num_classes, in_chans=in_chans, dataset=dataset, **kwargs)
#     model.default_cfg = default_cfg
#     if pretrained:
#         load_pretrained(model, default_cfg, num_classes, in_chans)
#     return model
#
#
# def tf_efficientnet_b4(
#         pretrained=False, num_classes=None, in_chans=3,
#         dataset='imagenet', training=True, **kwargs,
# ):
#     """ EfficientNet-B4. Tensorflow compatible variant """
#     if num_classes is None:
#         if 'imagenette' in dataset or 'imagewoof' in dataset:
#             num_classes = 10
#         elif 'imagenet' in dataset:
#             num_classes = 1000
#         elif dataset == 'cifar10':
#             num_classes = 10
#         elif dataset == 'cifar100':
#             num_classes = 100
#         else:
#             raise ValueError('Unfamiliar dataset: {}'.format(dataset))
#     kwargs.setdefault('regime', 'imagenet_normal' if dataset == 'imagenet' else 'cifar_normal')
#     if training:
#         kwargs.setdefault('drop_connect_rate', 0.2)
#         kwargs.setdefault('drop_rate', 0.4)
#     default_cfg = default_cfgs['tf_efficientnet_b4']
#     kwargs['bn_eps'] = _BN_EPS_TF_DEFAULT
#     kwargs['pad_type'] = 'same'
#     model = _gen_efficientnet(
#         channel_multiplier=1.4, depth_multiplier=1.8,
#         num_classes=num_classes, in_chans=in_chans, dataset=dataset, **kwargs)
#     model.default_cfg = default_cfg
#     if pretrained:
#         load_pretrained(model, default_cfg, num_classes, in_chans)
#     return model
#
#
# def tf_efficientnet_b5(
#         pretrained=False, num_classes=None, in_chans=3,
#         dataset='imagenet', training=True, **kwargs,
# ):
#     """ EfficientNet-B5. Tensorflow compatible variant """
#     if num_classes is None:
#         if 'imagenette' in dataset or 'imagewoof' in dataset:
#             num_classes = 10
#         elif 'imagenet' in dataset:
#             num_classes = 1000
#         elif dataset == 'cifar10':
#             num_classes = 10
#         elif dataset == 'cifar100':
#             num_classes = 100
#         else:
#             raise ValueError('Unfamiliar dataset: {}'.format(dataset))
#     kwargs.setdefault('regime', 'imagenet_normal' if dataset == 'imagenet' else 'cifar_normal')
#     if training:
#         kwargs.setdefault('drop_connect_rate', 0.2)
#         kwargs.setdefault('drop_rate', 0.4)
#     default_cfg = default_cfgs['tf_efficientnet_b5']
#     kwargs['bn_eps'] = _BN_EPS_TF_DEFAULT
#     kwargs['pad_type'] = 'same'
#     model = _gen_efficientnet(
#         channel_multiplier=1.6, depth_multiplier=2.2,
#         num_classes=num_classes, in_chans=in_chans, dataset=dataset, **kwargs)
#     model.default_cfg = default_cfg
#     if pretrained:
#         load_pretrained(model, default_cfg, num_classes, in_chans)
#     return model
