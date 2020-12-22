import numpy as np
import util

_BOUNDS10 = {
    'mlp': {
        'mnist': {
            'ail_all_or_and': {
                'beta1': [-1.107, -0.588],
                'beta2': [-5.18, -1.493],
                'eps': [-10.403, -7.598],
                'wd': [-7.013, -4.696],
                'max_lr': [-3.026, -2.097],
                'cycle_peak': [0.142, 0.428],
            },
            'ail_all_or_and_xnor': {
                'beta1': [-1.905, -0.433],
                'beta2': [-5.116, -1.521],
                'eps': [-9.884, -7.182],
                'wd': [-6.885, -4.065],
                'max_lr': [-2.794, -2.226],
                'cycle_peak': [0.133, 0.557],
            },
            'ail_all_or_xnor': {
                'beta1': [-1.224, -0.534],
                'beta2': [-4.709, -2.407],
                'eps': [-9.523, -8.676],
                'wd': [-6.189, -5.22],
                'max_lr': [-3.357, -0.742],
                'cycle_peak': [0.071, 0.423],
            },
            'ail_or': {
                'beta1': [-1.174, -0.46],
                'beta2': [-4.353, -1.373],
                'eps': [-10.198, -7.61],
                'wd': [-6.914, -5.401],
                'max_lr': [-2.838, -1.532],
                'cycle_peak': [0.157, 0.414],
            },
            'ail_part_or_and_xnor': {
                'beta1': [-1.672, -0.472],
                'beta2': [-4.619, -2.363],
                'eps': [-11.111, -6.665],
                'wd': [-6.494, -4.077],
                'max_lr': [-3.444, -1.935],
                'cycle_peak': [0.258, 0.43],
            },
            'ail_part_or_xnor': {
                'beta1': [-1.602, -0.629],
                'beta2': [-4.854, -2.971],
                'eps': [-10.314, -7.836],
                'wd': [-6.637, -4.666],
                'max_lr': [-3.014, -2.049],
                'cycle_peak': [0.16, 0.386],
            },
            'ail_xnor': {
                'beta1': [-0.902, -0.677],
                'beta2': [-4.941, -1.896],
                'eps': [-9.996, -6.097],
                'wd': [-7.19, -2.718],
                'max_lr': [-3.914, -2.07],
                'cycle_peak': [0.192, 0.401],
            },
            'bin_all_max_min': {
                'beta1': [-1.415, -0.854],
                'beta2': [-5.311, -2.645],
                'eps': [-9.781, -7.685],
                'wd': [-6.46, -4.983],
                'max_lr': [-3.006, -1.878],
                'cycle_peak': [0.127, 0.541],
            },
            'max': {
                'beta1': [-1.386, -0.477],
                'beta2': [-4.825, -0.823],
                'eps': [-9.196, -6.609],
                'wd': [-7.138, -4.403],
                'max_lr': [-3.015, -1.82],
                'cycle_peak': [0.124, 0.443],
            },
            'relu': {
                'beta1': [-1.559, -0.534],
                'beta2': [-5.047, -0.774],
                'eps': [-9.363, -7.392],
                'wd': [-7.067, -5.525],
                'max_lr': [-3.127, -1.957],
                'cycle_peak': [0.109, 0.515],
            },
        },
        'cifar100': {
            'max': {
                'beta1': -0.669,
                'beta2': -3.694,
                'eps': -10.372,
                'wd': -5.438,
                'max_lr': -2.328,
                'cycle_peak': 0.396,
            },
            'relu': {
                'beta1': -0.639,
                'beta2': -4.195,
                'eps': -7.136,
                'wd': -5.991,
                'max_lr': -2.023,
                'cycle_peak': 0.35,
            },
        }
    },
    'cnn': {
        'cifar10': {
            'ail_all_or_and': {
                'beta1': [-1.638, -0.787],
                'beta2': [-5.32, -0.066],
                'eps': [-8.944, -5.622],
                'wd': [-6.233, -3.265],
                'max_lr': [-3.116, -2.783],
                'cycle_peak': [0.108, 0.614],
            },
            'ail_all_or_and_xnor': {
                'beta1': [-1.446, -0.781],
                'beta2': [-5.366, -0.387],
                'eps': [-9.553, -6.087],
                'wd': [-6.386, -4.241],
                'max_lr': [-3.105, -2.734],
                'cycle_peak': [0.109, 0.617],
            },
            'ail_all_or_xnor': {
                'beta1': [-1.591, -0.782],
                'beta2': [-4.387, -0.494],
                'eps': [-10.329, -5.339],
                'wd': [-5.764, -2.636],
                'max_lr': [-3.087, -2.733],
                'cycle_peak': [0.312, 0.52],
            },
            'ail_or': {
                'beta1': [-1.453, -0.747],
                'beta2': [-3.963, -0.474],
                'eps': [-10.171, -5.57],
                'wd': [-6.614, -2.821],
                'max_lr': [-3.089, -2.695],
                'cycle_peak': [0.25, 0.549],
            },
            'ail_part_or_and_xnor': {
                'beta1': [-1.032, -0.548],
                'beta2': [-4.951, -1.705],
                'eps': [-10.111, -5.168],
                'wd': [-5.296, -3.459],
                'max_lr': [-2.91, -2.448],
                'cycle_peak': [0.293, 0.443],
            },
            'ail_part_or_xnor': {
                'beta1': [-1.56, -0.591],
                'beta2': [-5.393, -2.097],
                'eps': [-10.0, -5.894],
                'wd': [-5.516, -3.124],
                'max_lr': [-2.933, -2.51],
                'cycle_peak': [0.27, 0.48],
            },
            'ail_xnor': {
                'beta1': [-0.855, -0.42],
                'beta2': [-4.512, -1.982],
                'eps': [-8.251, -5.607],
                'wd': [-5.032, -2.866],
                'max_lr': [-2.954, -1.877],
                'cycle_peak': [0.25, 0.508],
            },
            'bin_all_max_min': {
                'beta1': [-1.395, -0.642],
                'beta2': [-4.385, -0.655],
                'eps': [-9.591, -6.883],
                'wd': [-7.204, -4.503],
                'max_lr': [-2.936, -2.654],
                'cycle_peak': [0.163, 0.594],
            },
            'max': {
                'beta1': [-1.153, -0.584],
                'beta2': [-5.029, -0.228],
                'eps': [-7.78, -6.267],
                'wd': [-6.741, -3.512],
                'max_lr': [-3.072, -2.453],
                'cycle_peak': [0.073, 0.429],
            },
            'relu': {
                'beta1': [-0.938, -0.46],
                'beta2': [-3.47, -1.853],
                'eps': [-9.384, -5.47],
                'wd': [-5.038, -3.033],
                'max_lr': [-2.701, -2.339],
                'cycle_peak': [0.147, 0.523],
            },
        },
        'cifar100': {
            'max': {
                'beta1': -0.519,
                'beta2': -2.915,
                'eps': -7.804,
                'wd': -3.542,
                'max_lr': -2.589,
                'cycle_peak': 0.456,
            },
            'relu': {
                'beta1': -0.54,
                'beta2': -2.997,
                'eps': -6.384,
                'wd': -3.693,
                'max_lr': -2.932,
                'cycle_peak': 0.453,
            },
            'ail_all_or_and': {
                'beta1': [-1.601, -0.665],
                'beta2': [-3.997, 0.253],
                'eps': [-6.765, -5.998],
                'wd': [-5.393, -3.119],
                'max_lr': [-3.406, -2.884],
                'cycle_peak': [0.098, 0.53],
            },
            'ail_all_or_and_xnor': {
                'beta1': [-1.728, -0.706],
                'beta2': [-3.114, 0.069],
                'eps': [-6.892, -5.974],
                'wd': [-5.305, -3.084],
                'max_lr': [-3.303, -2.912],
                'cycle_peak': [0.101, 0.536],
            },
            'ail_all_or_xnor': {
                'beta1': [-1.411, -0.85],
                'beta2': [-4.614, -0.741],
                'eps': [-10.417, -6.094],
                'wd': [-6.576, -3.179],
                'max_lr': [-3.057, -2.696],
                'cycle_peak': [0.341, 0.528],
            },
            'ail_or': {
                'beta1': [-1.184, -0.802],
                'beta2': [-5.165, -2.17],
                'eps': [-10.259, -7.348],
                'wd': [-6.332, -4.054],
                'max_lr': [-2.949, -2.586],
                'cycle_peak': [0.286, 0.515],
            },
            'ail_part_or_and_xnor': {
                'beta1': [-0.976, -0.395],
                'beta2': [-3.992, -1.471],
                'eps': [-8.771, -5.579],
                'wd': [-5.753, -3.041],
                'max_lr': [-2.888, -2.424],
                'cycle_peak': [0.244, 0.545],
            },
            'ail_part_or_xnor': {
                'beta1': [-1.088, -0.531],
                'beta2': [-3.974, -1.202],
                'eps': [-10.162, -5.259],
                'wd': [-6.128, -3.398],
                'max_lr': [-2.99, -2.476],
                'cycle_peak': [0.214, 0.533],
            },
            'ail_xnor': {
                'beta1': [-1.715, -0.162],
                'beta2': [-4.266, -1.55],
                'eps': [-8.613, -5.499],
                'wd': [-4.308, -2.981],
                'max_lr': [-2.835, -2.489],
                'cycle_peak': [0.351, 0.5],
            },
            'bin_all_max_min': {
                'beta1': [-1.451, -0.576],
                'beta2': [-3.595, -0.774],
                'eps': [-9.435, -5.665],
                'wd': [-6.48, -2.919],
                'max_lr': [-3.108, -2.559],
                'cycle_peak': [0.26, 0.58],
            },
        }
    }
}

_BOUNDS50 = {
    'mlp': {
        'cifar100': {
            'max': {
                'beta1': -0.792,
                'beta2': -3.824,
                'eps': -5.974,
                'wd': -3.463,
                'max_lr': -2.808,
                'cycle_peak': 0.421,
            },
            'relu': {
                'beta1': -1.001,
                'beta2': -2.626,
                'eps': -8.718,
                'wd': -3.385,
                'max_lr': -3.203,
                'cycle_peak': 0.481,
            },
        }
    },
    'cnn': {
        'cifar100': {
            'max': {
                'beta1': -1.07,
                'beta2': -3.441,
                'eps': -5.845,
                'wd': -3.375,
                'max_lr': -2.637,
                'cycle_peak': 0.449,
            },
            'relu': {
                'beta1': -0.426,
                'beta2': -3.893,
                'eps': -8.535,
                'wd': -3.413,
                'max_lr': -2.736,
                'cycle_peak': 0.401,
            },
        }
    }
}

_BOUNDS100 = {
    'mlp': {
        'cifar100': {
            'max': {
                'beta1': -0.632,
                'beta2': -3.758,
                'eps': -5.835,
                'wd': -2.616,
                'max_lr': -3.227,
                'cycle_peak': 0.376,
            },
            'relu': {
                'beta1': -0.831,
                'beta2': -5.063,
                'eps': -6.446,
                'wd': -2.688,
                'max_lr': -3.19,
                'cycle_peak': 0.401,
            },
        }
    },
    'cnn': {
        'cifar100': {
            'max': {
                'beta1': -0.575,
                'beta2': -4.64,
                'eps': -6.56,
                'wd': -3.438,
                'max_lr': -2.525,
                'cycle_peak': 0.372,
            },
            'relu': {
                'beta1': -0.504,
                'beta2': -5.047,
                'eps': -6.813,
                'wd': -3.458,
                'max_lr': -2.605,
                'cycle_peak': 0.421,
            },
        }
    }
}


def get_hparams(model, dataset, actfun, seed, epochs, search=False, hp_idx=None):

    util.seed_all(seed)
    rng = np.random.RandomState(seed)
    if epochs == 10:
        b = _BOUNDS10[model][dataset][actfun]
    elif epochs == 50:
        b = _BOUNDS50[model][dataset][actfun]
    elif epochs == 100:
        b = _BOUNDS100[model][dataset][actfun]

    if search:
        hparams = {"beta1": 1 - np.power(10., rng.uniform(b['beta1'][0], b['beta1'][1])),
                   "beta2": 1 - np.power(10., rng.uniform(b['beta2'][0], b['beta2'][1])),
                   "eps": np.power(10., rng.uniform(b['eps'][0], b['eps'][1])),
                   "wd": np.power(10., rng.uniform(b['wd'][0], b['wd'][1])),
                   "max_lr": np.power(10., rng.uniform(b['max_lr'][0], b['max_lr'][1])),
                   "cycle_peak": rng.uniform(b['cycle_peak'][0], b['cycle_peak'][1])
                   }
        if hp_idx is not None:
            hparams = {"beta1": 1 - np.power(10., b['beta1'][hp_idx]),
                       "beta2": 1 - np.power(10., b['beta2'][hp_idx]),
                       "eps": np.power(10., b['eps'][hp_idx]),
                       "wd": np.power(10., b['wd'][hp_idx]),
                       "max_lr": np.power(10., b['max_lr'][hp_idx]),
                       "cycle_peak": b['cycle_peak'][hp_idx],
                       }
    else:
        hparams = {"beta1": 1 - np.power(10., b['beta1']),
                   "beta2": 1 - np.power(10., b['beta2']),
                   "eps": np.power(10., b['eps']),
                   "wd": np.power(10., b['wd']),
                   "max_lr": np.power(10., b['max_lr']),
                   "cycle_peak": b['cycle_peak'],
                   }

    return hparams