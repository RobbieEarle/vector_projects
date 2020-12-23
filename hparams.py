import numpy as np
import util

_BOUNDS10 = {
    'mlp': {
        'mnist': {
            'ail_all_or_and': {
                'beta1': [-1.036, -0.784],
                'beta2': [-3.714, -1.796],
                'eps': [-10.123, -8.414],
                'wd': [-5.944, -4.778],
                'max_lr': [-3.024, -2.493],
                'cycle_peak': [0.146, 0.377],
            },
            'ail_all_or_and_xnor': {
                'beta1': [-1.389, -0.41],
                'beta2': [-5.004, -3.742],
                'eps': [-9.731, -6.989],
                'wd': [-7.128, -3.785],
                'max_lr': [-2.783, -2.151],
                'cycle_peak': [0.315, 0.39],
            },
            'ail_all_or_xnor': {
                'beta1': [-1.111, -0.961],
                'beta2': [-4.426, -3.556],
                'eps': [-9.392, -8.769],
                'wd': [-6.094, -5.2],
                'max_lr': [-3.478, -1.681],
                'cycle_peak': [0.146, 0.345],
            },
            'ail_or': {
                'beta1': [-1.051, -0.611],
                'beta2': [-4.476, -1.862],
                'eps': [-9.985, -9.033],
                'wd': [-6.221, -5.524],
                'max_lr': [-2.311, -1.585],
                'cycle_peak': [0.25, 0.418],
            },
            'ail_part_or_and_xnor': {
                'beta1': [-1.12, -0.456],
                'beta2': [-4.729, -3.252],
                'eps': [-11.029, -7.277],
                'wd': [-6.184, -4.529],
                'max_lr': [-3.135, -2.21],
                'cycle_peak': [0.268, 0.386],
            },
            'ail_part_or_xnor': {
                'beta1': [-1.416, -1.22],
                'beta2': [-4.965, -3.696],
                'eps': [-9.87, -7.813],
                'wd': [-6.828, -4.837],
                'max_lr': [-2.621, -2.028],
                'cycle_peak': [0.139, 0.333],
            },
            'ail_xnor': {
                'beta1': [-0.9, -0.748],
                'beta2': [-4.909, -3.018],
                'eps': [-8.247, -6.54],
                'wd': [-7.331, -3.513],
                'max_lr': [-3.224, -2.279],
                'cycle_peak': [0.209, 0.367],
            },
            'bin_all_max_min': {
                'beta1': [-1.268, -0.946],
                'beta2': [-5.292, -4.108],
                'eps': [-9.931, -8.3],
                'wd': [-5.839, -5.321],
                'max_lr': [-2.903, -2.281],
                'cycle_peak': [0.23, 0.376],
            },
            'max': {
                'beta1': [-1.35, -1.12],
                'beta2': [-5.549, -2.533],
                'eps': [-7.946, -6.968],
                'wd': [-7.721, -5.364],
                'max_lr': [-3.051, -2.336],
                'cycle_peak': [0.136, 0.36],
            },
            'relu': {
                'beta1': [-1.229, -0.65],
                'beta2': [-3.986, -2.557],
                'eps': [-9.448, -8.366],
                'wd': [-7.113, -5.751],
                'max_lr': [-2.38, -2.138],
                'cycle_peak': [0.115, 0.531],
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
                'beta1': [-1.182, -0.89],
                'beta2': [-5.133, -0.507],
                'eps': [-8.137, -7.257],
                'wd': [-5.087, -2.993],
                'max_lr': [-3.031, -2.902],
                'cycle_peak': [0.27, 0.624],
            },
            'ail_all_or_and_xnor': {
                'beta1': [-1.078, -0.831],
                'beta2': [-3.889, -1.046],
                'eps': [-8.638, -6.09],
                'wd': [-6.072, -4.225],
                'max_lr': [-2.968, -2.899],
                'cycle_peak': [0.242, 0.509],
            },
            'ail_all_or_xnor': {
                'beta1': [-1.208, -0.827],
                'beta2': [-4.235, -1.556],
                'eps': [-10.01, -6.206],
                'wd': [-3.498, -2.832],
                'max_lr': [-3.092, -2.784],
                'cycle_peak': [0.293, 0.407],
            },
            'ail_or': {
                'beta1': [-1.205, -0.665],
                'beta2': [-2.215, -1.635],
                'eps': [-9.997, -6.644],
                'wd': [-5.948, -2.816],
                'max_lr': [-2.882, -2.769],
                'cycle_peak': [0.343, 0.538],
            },
            'ail_part_or_and_xnor': {
                'beta1': [-0.896, -0.556],
                'beta2': [-5.089, -1.993],
                'eps': [-7.344, -4.681],
                'wd': [-4.006, -3.504],
                'max_lr': [-2.71, -2.542],
                'cycle_peak': [0.306, 0.431],
            },
            'ail_part_or_xnor': {
                'beta1': [-0.898, -0.791],
                'beta2': [-5.485, -3.263],
                'eps': [-9.38, -6.516],
                'wd': [-4.006, -3.172],
                'max_lr': [-2.844, -2.516],
                'cycle_peak': [0.256, 0.451],
            },
            'ail_xnor': {
                'beta1': [-0.842, -0.46],
                'beta2': [-3.958, -2.586],
                'eps': [-7.719, -6.97],
                'wd': [-3.738, -3.069],
                'max_lr': [-2.838, -2.191],
                'cycle_peak': [0.316, 0.475],
            },
            'bin_all_max_min': {
                'beta1': [-1.015, -0.67],
                'beta2': [-2.05, -0.682],
                'eps': [-8.693, -6.732],
                'wd': [-6.985, -5.393],
                'max_lr': [-2.964, -2.78],
                'cycle_peak': [0.36, 0.606],
            },
            'max': {
                'beta1': [-0.923, -0.621],
                'beta2': [-4.625, -1.682],
                'eps': [-7.161, -6.449],
                'wd': [-6.099, -2.827],
                'max_lr': [-2.684, -2.569],
                'cycle_peak': [0.221, 0.404],
            },
            'relu': {
                'beta1': [-0.97, -0.608],
                'beta2': [-3.284, -1.958],
                'eps': [-9.21, -5.12],
                'wd': [-3.599, -3.328],
                'max_lr': [-2.541, -2.33],
                'cycle_peak': [0.295, 0.368],
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
                'beta1': [-0.831, -0.652],
                'beta2': [-3.055, -1.041],
                'eps': [-6.911, -6.209],
                'wd': [-5.095, -2.986],
                'max_lr': [-3.117, -3.032],
                'cycle_peak': [0.235, 0.555],
            },
            'ail_all_or_and_xnor': {
                'beta1': [-1.026, -0.657],
                'beta2': [-1.664, -1.176],
                'eps': [-6.982, -6.478],
                'wd': [-4.716, -3.064],
                'max_lr': [-3.078, -2.999],
                'cycle_peak': [0.322, 0.536],
            },
            'ail_all_or_xnor': {
                'beta1': [-1.072, -0.832],
                'beta2': [-3.176, -1.085],
                'eps': [-10.194, -8.124],
                'wd': [-5.169, -3.313],
                'max_lr': [-3.093, -2.777],
                'cycle_peak': [0.428, 0.553],
            },
            'ail_or': {
                'beta1': [-0.973, -0.801],
                'beta2': [-3.868, -3.152],
                'eps': [-10.32, -8.458],
                'wd': [-4.325, -4.145],
                'max_lr': [-2.749, -2.679],
                'cycle_peak': [0.43, 0.51],
            },
            'ail_part_or_and_xnor': {
                'beta1': [-0.907, -0.421],
                'beta2': [-3.614, -1.967],
                'eps': [-8.83, -6.504],
                'wd': [-3.719, -3.093],
                'max_lr': [-2.814, -2.5],
                'cycle_peak': [0.357, 0.544],
            },
            'ail_part_or_xnor': {
                'beta1': [-0.8, -0.559],
                'beta2': [-3.703, -1.931],
                'eps': [-9.552, -5.891],
                'wd': [-3.612, -3.47],
                'max_lr': [-2.718, -2.588],
                'cycle_peak': [0.386, 0.506],
            },
            'ail_xnor': {
                'beta1': [-1.017, -0.254],
                'beta2': [-3.617, -2.148],
                'eps': [-8.46, -7.091],
                'wd': [-3.233, -2.985],
                'max_lr': [-2.806, -2.561],
                'cycle_peak': [0.367, 0.509],
            },
            'bin_all_max_min': {
                'beta1': [-0.786, -0.575],
                'beta2': [-2.607, -1.192],
                'eps': [-9.172, -7.121],
                'wd': [-5.117, -3.011],
                'max_lr': [-2.997, -2.769],
                'cycle_peak': [0.341, 0.621],
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