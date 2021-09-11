from . import __meta__
from .efficientnet import EfficientNet, VALID_MODELS
from .utils.efficientnet_utils import (
    GlobalParams,
    BlockArgs,
    BlockDecoder,
    efficientnet,
    get_model_params,
)

__version__ = __meta__.version
