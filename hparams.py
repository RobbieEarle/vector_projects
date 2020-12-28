import numpy as np
import util

_BOUNDS10 = {
    'mlp': {
        'mnist': {
            'ail_all_or_and': {
                'beta1': -0.911,
                'beta2': -2.435,
                'eps': -10.393,
                'wd': -5.813,
                'max_lr': -2.98,
                'cycle_peak': 0.166,
            },
            'ail_all_or_and_xnor': {
                'beta1': -0.804,
                'beta2': -4.151,
                'eps': -8.91,
                'wd': -4.685,
                'max_lr': -2.417,
                'cycle_peak': 0.376,
            },
            'ail_all_or_xnor': {
                'beta1': -1.016,
                'beta2': -4.279,
                'eps': -9.249,
                'wd': -5.544,
                'max_lr': -1.874,
                'cycle_peak': 0.211,
            },
            'ail_or': {
                'beta1': -0.575,
                'beta2': -3.795,
                'eps': -9.623,
                'wd': -5.713,
                'max_lr': -1.9,
                'cycle_peak': 0.364,
            },
            'ail_part_or_and_xnor': {
                'beta1': -0.737,
                'beta2': -4.563,
                'eps': -10.485,
                'wd': -5.245,
                'max_lr': -2.789,
                'cycle_peak': 0.343,
            },
            'ail_part_or_xnor': {
                'beta1': -1.063,
                'beta2': -4.108,
                'eps': -10.195,
                'wd': -5.601,
                'max_lr': -2.866,
                'cycle_peak': 0.273,
            },
            'ail_xnor': {
                'beta1': -0.893,
                'beta2': -4.209,
                'eps': -7.833,
                'wd': -4.931,
                'max_lr': -2.288,
                'cycle_peak': 0.23,
            },
            'bin_all_max_min': {
                'beta1': -1.104,
                'beta2': -4.255,
                'eps': -9.68,
                'wd': -5.684,
                'max_lr': -2.833,
                'cycle_peak': 0.334,
            },
            'max': {
                'beta1': -0.543,
                'beta2': -1.334,
                'eps': -7.684,
                'wd': -4.662,
                'max_lr': -2.636,
                'cycle_peak': 0.171,
            },
            'relu': {
                'beta1': -0.682,
                'beta2': -3.323,
                'eps': -9.239,
                'wd': -6.32,
                'max_lr': -2.23,
                'cycle_peak': 0.4,
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
                'beta1': -0.903,
                'beta2': -2.209,
                'eps': -7.501,
                'wd': -3.606,
                'max_lr': -2.962,
                'cycle_peak': 0.284,
            },
            'ail_all_or_and_xnor': {
                'beta1': -0.856,
                'beta2': -2.105,
                'eps': -8.238,
                'wd': -5.246,
                'max_lr': -2.916,
                'cycle_peak': 0.393,
            },
            'ail_all_or_xnor': {
                'beta1': -0.974,
                'beta2': -2.692,
                'eps': -6.482,
                'wd': -2.857,
                'max_lr': -2.87,
                'cycle_peak': 0.311,
            },
            'ail_or': {
                'beta1': -1.005,
                'beta2': -1.928,
                'eps': -7.536,
                'wd': -3.148,
                'max_lr': -2.779,
                'cycle_peak': 0.368,
            },
            'ail_part_or_and_xnor': {
                'beta1': -0.845,
                'beta2': -4.598,
                'eps': -4.735,
                'wd': -3.622,
                'max_lr': -2.595,
                'cycle_peak': 0.366,
            },
            'ail_part_or_xnor': {
                'beta1': -0.869,
                'beta2': -4.102,
                'eps': -8.633,
                'wd': -3.386,
                'max_lr': -2.646,
                'cycle_peak': 0.337,
            },
            'ail_xnor': {
                'beta1': -0.664,
                'beta2': -3.346,
                'eps': -7.574,
                'wd': -3.207,
                'max_lr': -2.491,
                'cycle_peak': 0.344,
            },
            'bin_all_max_min': {
                'beta1': -0.88,
                'beta2': -1.386,
                'eps': -7.55,
                'wd': -6.329,
                'max_lr': -2.838,
                'cycle_peak': 0.504,
            },
            'max': {
                'beta1': -0.868,
                'beta2': -2.224,
                'eps': -6.472,
                'wd': -3.629,
                'max_lr': -2.609,
                'cycle_peak': 0.294,
            },
            'relu': {
                'beta1': -0.755,
                'beta2': -2.929,
                'eps': -7.027,
                'wd': -3.585,
                'max_lr': -2.355,
                'cycle_peak': 0.303,
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
                'beta1': -0.701,
                'beta2': -1.925,
                'eps': -6.823,
                'wd': -4.256,
                'max_lr': -3.051,
                'cycle_peak': 0.399,
            },
            'ail_all_or_and_xnor': {
                'beta1': -0.88,
                'beta2': -1.455,
                'eps': -6.57,
                'wd': -3.26,
                'max_lr': -2.997,
                'cycle_peak': 0.387,
            },
            'ail_all_or_xnor': {
                'beta1': -0.909,
                'beta2': -2.251,
                'eps': -8.966,
                'wd': -3.45,
                'max_lr': -2.865,
                'cycle_peak': 0.481,
            },
            'ail_or': {
                'beta1': -0.811,
                'beta2': -3.453,
                'eps': -10.211,
                'wd': -4.161,
                'max_lr': -2.704,
                'cycle_peak': 0.505,
            },
            'ail_part_or_and_xnor': {
                'beta1': -0.64,
                'beta2': -2.535,
                'eps': -7.47,
                'wd': -3.379,
                'max_lr': -2.694,
                'cycle_peak': 0.429,
            },
            'ail_part_or_xnor': {
                'beta1': -0.727,
                'beta2': -3.265,
                'eps': -6.161,
                'wd': -3.485,
                'max_lr': -2.629,
                'cycle_peak': 0.454,
            },
            'ail_xnor': {
                'beta1': -0.787,
                'beta2': -3.254,
                'eps': -7.192,
                'wd': -3.012,
                'max_lr': -2.639,
                'cycle_peak': 0.447,
            },
            'bin_all_max_min': {
                'beta1': -0.631,
                'beta2': -1.804,
                'eps': -8.554,
                'wd': -3.473,
                'max_lr': -2.883,
                'cycle_peak': 0.374,
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
        'mnist': {
            'ail_all_or_and': {
                'beta1': -0.911,
                'beta2': -2.435,
                'eps': -10.393,
                'wd': -5.813,
                'max_lr': -2.98,
                'cycle_peak': 0.166,
            },
            'ail_all_or_and_xnor': {
                'beta1': -0.804,
                'beta2': -4.151,
                'eps': -8.91,
                'wd': -4.685,
                'max_lr': -2.417,
                'cycle_peak': 0.376,
            },
            'ail_all_or_xnor': {
                'beta1': -1.016,
                'beta2': -4.279,
                'eps': -9.249,
                'wd': -5.544,
                'max_lr': -1.874,
                'cycle_peak': 0.211,
            },
            'ail_or': {
                'beta1': -0.575,
                'beta2': -3.795,
                'eps': -9.623,
                'wd': -5.713,
                'max_lr': -1.9,
                'cycle_peak': 0.364,
            },
            'ail_part_or_and_xnor': {
                'beta1': -0.737,
                'beta2': -4.563,
                'eps': -10.485,
                'wd': -5.245,
                'max_lr': -2.789,
                'cycle_peak': 0.343,
            },
            'ail_part_or_xnor': {
                'beta1': -1.063,
                'beta2': -4.108,
                'eps': -10.195,
                'wd': -5.601,
                'max_lr': -2.866,
                'cycle_peak': 0.273,
            },
            'ail_xnor': {
                'beta1': -0.893,
                'beta2': -4.209,
                'eps': -7.833,
                'wd': -4.931,
                'max_lr': -2.288,
                'cycle_peak': 0.23,
            },
            'bin_all_max_min': {
                'beta1': -1.104,
                'beta2': -4.255,
                'eps': -9.68,
                'wd': -5.684,
                'max_lr': -2.833,
                'cycle_peak': 0.334,
            },
            'max': {
                'beta1': -0.543,
                'beta2': -1.334,
                'eps': -7.684,
                'wd': -4.662,
                'max_lr': -2.636,
                'cycle_peak': 0.171,
            },
            'relu': {
                'beta1': -0.682,
                'beta2': -3.323,
                'eps': -9.239,
                'wd': -6.32,
                'max_lr': -2.23,
                'cycle_peak': 0.4,
            },
        },
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
        'cifar10': {
            'ail_all_or_and': {
                'beta1': -0.903,
                'beta2': -2.209,
                'eps': -7.501,
                'wd': -3.606,
                'max_lr': -2.962,
                'cycle_peak': 0.284,
            },
            'ail_all_or_and_xnor': {
                'beta1': -0.856,
                'beta2': -2.105,
                'eps': -8.238,
                'wd': -5.246,
                'max_lr': -2.916,
                'cycle_peak': 0.393,
            },
            'ail_all_or_xnor': {
                'beta1': -0.974,
                'beta2': -2.692,
                'eps': -6.482,
                'wd': -2.857,
                'max_lr': -2.87,
                'cycle_peak': 0.311,
            },
            'ail_or': {
                'beta1': -1.005,
                'beta2': -1.928,
                'eps': -7.536,
                'wd': -3.148,
                'max_lr': -2.779,
                'cycle_peak': 0.368,
            },
            'ail_part_or_and_xnor': {
                'beta1': -0.845,
                'beta2': -4.598,
                'eps': -4.735,
                'wd': -3.622,
                'max_lr': -2.595,
                'cycle_peak': 0.366,
            },
            'ail_part_or_xnor': {
                'beta1': -0.869,
                'beta2': -4.102,
                'eps': -8.633,
                'wd': -3.386,
                'max_lr': -2.646,
                'cycle_peak': 0.337,
            },
            'ail_xnor': {
                'beta1': -0.664,
                'beta2': -3.346,
                'eps': -7.574,
                'wd': -3.207,
                'max_lr': -2.491,
                'cycle_peak': 0.344,
            },
            'bin_all_max_min': {
                'beta1': -0.88,
                'beta2': -1.386,
                'eps': -7.55,
                'wd': -6.329,
                'max_lr': -2.838,
                'cycle_peak': 0.504,
            },
            'max': {
                'beta1': -0.868,
                'beta2': -2.224,
                'eps': -6.472,
                'wd': -3.629,
                'max_lr': -2.609,
                'cycle_peak': 0.294,
            },
            'relu': {
                'beta1': -0.755,
                'beta2': -2.929,
                'eps': -7.027,
                'wd': -3.585,
                'max_lr': -2.355,
                'cycle_peak': 0.303,
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
                'beta1': -0.701,
                'beta2': -1.925,
                'eps': -6.823,
                'wd': -4.256,
                'max_lr': -3.051,
                'cycle_peak': 0.399,
            },
            'ail_all_or_and_xnor': {
                'beta1': -0.88,
                'beta2': -1.455,
                'eps': -6.57,
                'wd': -3.26,
                'max_lr': -2.997,
                'cycle_peak': 0.387,
            },
            'ail_all_or_xnor': {
                'beta1': -0.909,
                'beta2': -2.251,
                'eps': -8.966,
                'wd': -3.45,
                'max_lr': -2.865,
                'cycle_peak': 0.481,
            },
            'ail_or': {
                'beta1': -0.811,
                'beta2': -3.453,
                'eps': -10.211,
                'wd': -4.161,
                'max_lr': -2.704,
                'cycle_peak': 0.505,
            },
            'ail_part_or_and_xnor': {
                'beta1': -0.64,
                'beta2': -2.535,
                'eps': -7.47,
                'wd': -3.379,
                'max_lr': -2.694,
                'cycle_peak': 0.429,
            },
            'ail_part_or_xnor': {
                'beta1': -0.727,
                'beta2': -3.265,
                'eps': -6.161,
                'wd': -3.485,
                'max_lr': -2.629,
                'cycle_peak': 0.454,
            },
            'ail_xnor': {
                'beta1': -0.787,
                'beta2': -3.254,
                'eps': -7.192,
                'wd': -3.012,
                'max_lr': -2.639,
                'cycle_peak': 0.447,
            },
            'bin_all_max_min': {
                'beta1': -0.631,
                'beta2': -1.804,
                'eps': -8.554,
                'wd': -3.473,
                'max_lr': -2.883,
                'cycle_peak': 0.374,
            },
        },
        'cifar100_old': {
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