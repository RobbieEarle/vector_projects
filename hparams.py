import numpy as np
import util

_BOUNDS10 = {
    'mlp': {
        'mnist': {
            'ail_all_or_and': {
                'beta1': [-1.001, -0.784],
                'beta2': [-3.47, -2.033],
                'eps': [-10.393, -9.085],
                'wd': [-5.977, -5.433],
                'max_lr': [-3.013, -2.79],
                'cycle_peak': [0.152, 0.301],
            },
            'ail_all_or_and_xnor': {
                'beta1': [-1.032, -0.875],
                'beta2': [-4.726, -4.044],
                'eps': [-9.746, -8.892],
                'wd': [-6.441, -4.963],
                'max_lr': [-2.915, -2.311],
                'cycle_peak': [0.307, 0.401],
            },
            'ail_all_or_xnor': {
                'beta1': [-1.092, -1.011],
                'beta2': [-4.335, -3.721],
                'eps': [-9.419, -9.064],
                'wd': [-5.856, -5.231],
                'max_lr': [-3.283, -3.009],
                'cycle_peak': [0.136, 0.316],
            },
            'ail_or': {
                'beta1': [-0.938, -0.515],
                'beta2': [-4.388, -2.797],
                'eps': [-9.979, -9.337],
                'wd': [-6.284, -5.687],
                'max_lr': [-2.236, -1.78],
                'cycle_peak': [0.331, 0.372],
            },
            'ail_part_or_and_xnor': {
                'beta1': [-1.001, -0.468],
                'beta2': [-4.685, -4.151],
                'eps': [-10.869, -9.366],
                'wd': [-5.622, -4.672],
                'max_lr': [-2.975, -2.482],
                'cycle_peak': [0.305, 0.343],
            },
            'ail_part_or_xnor': {
                'beta1': [-1.386, -1.236],
                'beta2': [-4.876, -3.888],
                'eps': [-9.374, -8.078],
                'wd': [-6.782, -4.746],
                'max_lr': [-2.432, -2.121],
                'cycle_peak': [0.163, 0.292],
            },
            'ail_xnor': {
                'beta1': [-0.898, -0.782],
                'beta2': [-4.867, -3.874],
                'eps': [-8.038, -7.051],
                'wd': [-7.888, -4.601],
                'max_lr': [-2.842, -2.426],
                'cycle_peak': [0.232, 0.325],
            },
            'bin_all_max_min': {
                'beta1': [-1.238, -1.009],
                'beta2': [-5.331, -4.692],
                'eps': [-9.997, -8.604],
                'wd': [-5.721, -5.54],
                'max_lr': [-2.861, -2.645],
                'cycle_peak': [0.283, 0.4],
            },
            'max': {
                'beta1': [-1.358, -1.151],
                'beta2': [-5.894, -2.966],
                'eps': [-7.709, -6.979],
                'wd': [-8.088, -6.002],
                'max_lr': [-2.914, -2.841],
                'cycle_peak': [0.117, 0.393],
            },
            'relu': {
                'beta1': [-1.108, -0.575],
                'beta2': [-3.779, -2.813],
                'eps': [-9.604, -8.671],
                'wd': [-6.979, -6.243],
                'max_lr': [-2.364, -2.236],
                'cycle_peak': [0.11, 0.508],
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
                'beta1': [-1.091, -0.897],
                'beta2': [-2.839, -1.688],
                'eps': [-8.334, -7.478],
                'wd': [-4.65, -3.19],
                'max_lr': [-3.022, -2.936],
                'cycle_peak': [0.211, 0.549],
            },
            'ail_all_or_and_xnor': {
                'beta1': [-1.075, -0.775],
                'beta2': [-2.66, -1.672],
                'eps': [-8.601, -5.694],
                'wd': [-5.784, -4.43],
                'max_lr': [-2.968, -2.902],
                'cycle_peak': [0.267, 0.514],
            },
            'ail_all_or_xnor': {
                'beta1': [-1.019, -0.809],
                'beta2': [-2.886, -2.673],
                'eps': [-9.875, -5.723],
                'wd': [-3.385, -2.801],
                'max_lr': [-2.926, -2.847],
                'cycle_peak': [0.299, 0.397],
            },
            'ail_or': {
                'beta1': [-1.021, -0.684],
                'beta2': [-2.07, -1.861],
                'eps': [-10.26, -6.849],
                'wd': [-4.505, -2.809],
                'max_lr': [-2.831, -2.777],
                'cycle_peak': [0.321, 0.426],
            },
            'ail_part_or_and_xnor': {
                'beta1': [-0.852, -0.64],
                'beta2': [-5.12, -4.052],
                'eps': [-6.717, -4.374],
                'wd': [-3.703, -3.507],
                'max_lr': [-2.666, -2.554],
                'cycle_peak': [0.34, 0.431],
            },
            'ail_part_or_xnor': {
                'beta1': [-0.883, -0.795],
                'beta2': [-4.499, -3.963],
                'eps': [-9.226, -6.974],
                'wd': [-3.536, -3.255],
                'max_lr': [-2.647, -2.547],
                'cycle_peak': [0.252, 0.345],
            },
            'ail_xnor': {
                'beta1': [-0.766, -0.646],
                'beta2': [-3.952, -2.735],
                'eps': [-7.686, -7.137],
                'wd': [-3.473, -3.119],
                'max_lr': [-2.594, -2.431],
                'cycle_peak': [0.335, 0.494],
            },
            'bin_all_max_min': {
                'beta1': [-0.895, -0.766],
                'beta2': [-1.954, -1.371],
                'eps': [-8.244, -7.291],
                'wd': [-6.763, -5.529],
                'max_lr': [-2.924, -2.803],
                'cycle_peak': [0.358, 0.54],
            },
            'max': {
                'beta1': [-0.871, -0.774],
                'beta2': [-2.444, -1.994],
                'eps': [-6.921, -6.39],
                'wd': [-3.882, -3.267],
                'max_lr': [-2.642, -2.589],
                'cycle_peak': [0.253, 0.394],
            },
            'relu': {
                'beta1': [-0.839, -0.67],
                'beta2': [-3.057, -2.558],
                'eps': [-7.671, -5.177],
                'wd': [-3.601, -3.433],
                'max_lr': [-2.409, -2.314],
                'cycle_peak': [0.302, 0.352],
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
                'beta1': [-0.838, -0.635],
                'beta2': [-2.857, -1.246],
                'eps': [-6.883, -6.613],
                'wd': [-4.525, -3.356],
                'max_lr': [-3.099, -3.05],
                'cycle_peak': [0.346, 0.496],
            },
            'ail_all_or_and_xnor': {
                'beta1': [-0.886, -0.756],
                'beta2': [-1.598, -1.388],
                'eps': [-6.787, -6.515],
                'wd': [-4.548, -2.938],
                'max_lr': [-3.078, -2.994],
                'cycle_peak': [0.304, 0.492],
            },
            'ail_all_or_xnor': {
                'beta1': [-1.028, -0.874],
                'beta2': [-2.265, -1.601],
                'eps': [-9.479, -8.669],
                'wd': [-3.776, -3.341],
                'max_lr': [-2.916, -2.814],
                'cycle_peak': [0.466, 0.533],
            },
            'ail_or': {
                'beta1': [-0.943, -0.821],
                'beta2': [-3.872, -3.687],
                'eps': [-10.048, -8.693],
                'wd': [-4.255, -4.149],
                'max_lr': [-2.734, -2.702],
                'cycle_peak': [0.436, 0.488],
            },
            'ail_part_or_and_xnor': {
                'beta1': [-0.859, -0.518],
                'beta2': [-2.873, -2.245],
                'eps': [-7.738, -7.214],
                'wd': [-3.537, -3.278],
                'max_lr': [-2.753, -2.634],
                'cycle_peak': [0.409, 0.501],
            },
            'ail_part_or_xnor': {
                'beta1': [-0.806, -0.621],
                'beta2': [-3.522, -2.038],
                'eps': [-8.183, -5.569],
                'wd': [-3.551, -3.457],
                'max_lr': [-2.647, -2.606],
                'cycle_peak': [0.393, 0.47],
            },
            'ail_xnor': {
                'beta1': [-0.781, -0.462],
                'beta2': [-3.514, -2.564],
                'eps': [-8.194, -7.321],
                'wd': [-3.056, -2.992],
                'max_lr': [-2.758, -2.598],
                'cycle_peak': [0.367, 0.467],
            },
            'bin_all_max_min': {
                'beta1': [-0.82, -0.602],
                'beta2': [-2.395, -1.64],
                'eps': [-8.749, -7.316],
                'wd': [-5.046, -3.187],
                'max_lr': [-2.987, -2.859],
                'cycle_peak': [0.32, 0.561],
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