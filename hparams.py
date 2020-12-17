import numpy as np
import util


_BOUNDS10 = {
    'mlp': {
        'cifar100': {
            'max': {
                'beta1': [-0.929, -0.654],
                'beta2': [-4.031, -3.304],
                'eps': [-10.777, -8.675],
                'wd': [-6.337, -4.792],
                'max_lr': [-2.477, -2.236],
                'cycle_peak': [0.338, 0.423],
            },
            'relu': {
                'beta1': [-1.214, -0.598],
                'beta2': [-4.693, -2.578],
                'eps': [-8.303, -6.945],
                'wd': [-6.43, -5.979],
                'max_lr': [-2.33, -1.974],
                'cycle_peak': [0.228, 0.403],
            },
        }
    },
    'cnn': {
        'cifar100': {
            'max': {
                'beta1': [-0.693, -0.435],
                'beta2': [-2.932, -2.553],
                'eps': [-8.093, -7.251],
                'wd': [-3.867, -3.362],
                'max_lr': [-2.612, -2.531],
                'cycle_peak': [0.428, 0.457],
            },
            'relu': {
                'beta1': [-0.86, -0.391],
                'beta2': [-3.724, -1.96],
                'eps': [-8.229, -5.806],
                'wd': [-4.42, -3.467],
                'max_lr': [-2.909, -2.626],
                'cycle_peak': [0.434, 0.473],
            },
        }
    }
}

_BOUNDS50 = {
    'mlp': {
        'cifar100': {
            'max': {
                'beta1': [-1.339, -0.676],
                'beta2': [-4.391, -1.265],
                'eps': [-8.151, -5.67],
                'wd': [-3.527, -3.399],
                'max_lr': [-2.937, -2.803],
                'cycle_peak': [0.421, 0.449],
            },
            'relu': {
                'beta1': [-1.502, -0.614],
                'beta2': [-3.373, -2.276],
                'eps': [-9.926, -8.101],
                'wd': [-3.627, -3.278],
                'max_lr': [-3.46, -2.964],
                'cycle_peak': [0.156, 0.576],
            },
        }
    },
    'cnn': {
        'cifar100': {
            'max': {
                'beta1': [-1.7, -0.77],
                'beta2': [-4.137, -3.186],
                'eps': [-6.822, -5.746],
                'wd': [-3.437, -3.077],
                'max_lr': [-2.843, -2.535],
                'cycle_peak': [0.42, 0.457],
            },
            'relu': {
                'beta1': [-0.577, -0.396],
                'beta2': [-4.298, -2.718],
                'eps': [-9.432, -6.706],
                'wd': [-3.9, -3.243],
                'max_lr': [-2.806, -2.682],
                'cycle_peak': [0.389, 0.437],
            },
        }
    }
}

_BOUNDS100 = {
    'mlp': {
        'cifar100': {
            'max': {
                'beta1': [-0.823, -0.561],
                'beta2': [-4.194, -3.469],
                'eps': [-7.035, -5.775],
                'wd': [-2.643, -2.457],
                'max_lr': [-3.439, -3.177],
                'cycle_peak': [0.348, 0.426],
            },
            'relu': {
                'beta1': [-1.496, -0.694],
                'beta2': [-5.62, -3.088],
                'eps': [-7.919, -5.566],
                'wd': [-2.879, -2.432],
                'max_lr': [-3.28, -3.025],
                'cycle_peak': [0.386, 0.465],
            },
        }
    },
    'cnn': {
        'cifar100': {
            'max': {
                'beta1': [-0.924, -0.403],
                'beta2': [-4.778, -3.95],
                'eps': [-6.991, -6.198],
                'wd': [-3.669, -3.082],
                'max_lr': [-2.668, -2.363],
                'cycle_peak': [0.362, 0.419],
            },
            'relu': {
                'beta1': [-1.406, -0.189],
                'beta2': [-5.351, -4.208],
                'eps': [-8.59, -5.745],
                'wd': [-3.793, -3.308],
                'max_lr': [-3.093, -2.467],
                'cycle_peak': [0.303, 0.433],
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