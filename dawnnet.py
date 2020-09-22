import torch
import torch.nn as nn

class DawnNet(nn.Module):

    def __init__(self, num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None, num_input_channels=3):
        print()