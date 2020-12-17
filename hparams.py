import numpy as np
import util


_BOUNDS10 = {
    'mlp': {
        'cifar100': {
            'max': {
                'beta1': [-1.057, -0.666],
                'beta2': [-4.66, -2.716],
                'eps': [-10.46, -8.237],
                'wd': [-6.665, -4.015],
                'max_lr': [-3.019, -2.087],
                'cycle_peak': [0.257, 0.458],
            },
            'relu': {
                'beta1': [-1.256, -0.608],
                'beta2': [-5.587, -2.507],
                'eps': [-9.988, -5.934],
                'wd': [-6.814, -4.281],
                'max_lr': [-2.684, -1.829],
                'cycle_peak': [0.148, 0.448],
            },
        }
    },
    'cnn': {
        'cifar100': {
            'max': {
                'beta1': [-0.958, -0.419],
                'beta2': [-4.563, -1.723],
                'eps': [-8.253, -5.539],
                'wd': [-5.174, -3.185],
                'max_lr': [-2.668, -2.501],
                'cycle_peak': [0.346, 0.458],
            },
            'relu': {
                'beta1': [-1.199, -0.299],
                'beta2': [-5.034, -2.065],
                'eps': [-8.528, -5.772],
                'wd': [-5.28, -3.196],
                'max_lr': [-2.805, -2.452],
                'cycle_peak': [0.362, 0.477],
            },
        }
    }
}

_BOUNDS50 = {
    'mlp': {
        'cifar100': {
            'max': {
                'beta1': [-1.795, -0.178],
                'beta2': [-4.072, -0.887],
                'eps': [-8.13, -5.911],
                'wd': [-3.88, -3.361],
                'max_lr': [-3.676, -2.384],
                'cycle_peak': [0.406, 0.466],
            },
            'relu': {
                'beta1': [-2.097, -0.554],
                'beta2': [-4.126, -0.497],
                'eps': [-10.042, -5.057],
                'wd': [-4.529, -3.174],
                'max_lr': [-3.578, -2.63],
                'cycle_peak': [0.134, 0.542],
            },
        }
    },
    'cnn': {
        'cifar100': {
            'max': {
                'beta1': [-1.763, -0.29],
                'beta2': [-4.024, -0.821],
                'eps': [-7.486, -5.836],
                'wd': [-3.985, -3.067],
                'max_lr': [-3.051, -2.453],
                'cycle_peak': [0.402, 0.479],
            },
            'relu': {
                'beta1': [-2.634, -0.183],
                'beta2': [-4.571, -1.426],
                'eps': [-8.948, -5.935],
                'wd': [-3.88, -3.135],
                'max_lr': [-2.971, -2.463],
                'cycle_peak': [0.382, 0.467],
            },

        }
    }
}

_BOUNDS100 = {
    'mlp': {
        'cifar100': {
            'max': {
                'beta1': [-1.778, -0.349],
                'beta2': [-4.509, -2.287],
                'eps': [-8.759, -5.57],
                'wd': [-4.702, -2.439],
                'max_lr': [-3.96, -3.231],
                'cycle_peak': [0.284, 0.474],
            },
            'relu': {
                'beta1': [-2.422, -0.658],
                'beta2': [-5.606, -0.661],
                'eps': [-7.995, -5.685],
                'wd': [-4.549, -2.45],
                'max_lr': [-3.692, -2.859],
                'cycle_peak': [0.311, 0.483],
            },
        }
    },
    'cnn': {
        'cifar100': {
            'max': {
                'beta1': [-2.562, -0.095],
                'beta2': [-4.881, -1.651],
                'eps': [-8.945, -6.038],
                'wd': [-4.138, -3.159],
                'max_lr': [-2.959, -2.435],
                'cycle_peak': [0.365, 0.457],
            },
            'relu': {
                'beta1': [-2.562, -0.166],
                'beta2': [-5.228, -1.1],
                'eps': [-8.995, -5.822],
                'wd': [-3.882, -3.332],
                'max_lr': [-3.055, -2.481],
                'cycle_peak': [0.216, 0.537],
            },
        }
    }
}


def get_hparams(model, dataset, actfun, seed, epochs):

    util.seed_all(seed)
    rng = np.random.RandomState(seed)
    if epochs == 10:
        b = _BOUNDS10[model][dataset][actfun]
    elif epochs == 50:
        b = _BOUNDS50[model][dataset][actfun]
    elif epochs == 100:
        b = _BOUNDS100[model][dataset][actfun]
    hparams = {"beta1": 1 - np.power(10., rng.uniform(b['beta1'][0], b['beta1'][1])),
               "beta2": 1 - np.power(10., rng.uniform(b['beta2'][0], b['beta2'][1])),
               "eps": np.power(10., rng.uniform(b['eps'][0], b['eps'][1])),
               "wd": np.power(10., rng.uniform(b['wd'][0], b['wd'][1])),
               "max_lr": np.power(10., rng.uniform(b['max_lr'][0], b['max_lr'][1])),
               "cycle_peak": rng.uniform(b['cycle_peak'][0], b['cycle_peak'][1])
               }

    return hparams