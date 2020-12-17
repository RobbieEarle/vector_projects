import numpy as np
import util


_BOUNDS10 = {
    'mlp': {
        'cifar100': {
            'max': {
                'beta1': [-0.915, -0.641],
                'beta2': [-3.714, -3.455],
                'eps': [-10.612, -8.54],
                'wd': [-6.372, -5.011],
                'max_lr': [-2.365, -2.269],
                'cycle_peak': [0.372, 0.421],
            },
            'relu': {
                'beta1': [-0.854, -0.577],
                'beta2': [-4.582, -2.951],
                'eps': [-8.375, -6.872],
                'wd': [-6.372, -5.977],
                'max_lr': [-2.321, -2.015],
                'cycle_peak': [0.305, 0.405],
            },
        }
    },
    'cnn': {
        'cifar100': {
            'max': {
                'beta1': [-0.595, -0.446],
                'beta2': [-2.918, -2.702],
                'eps': [-8.063, -7.429],
                'wd': [-3.849, -3.435],
                'max_lr': [-2.598, -2.527],
                'cycle_peak': [0.432, 0.454],
            },
            'relu': {
                'beta1': [-0.572, -0.366],
                'beta2': [-3.158, -2.137],
                'eps': [-7.669, -6.365],
                'wd': [-4.337, -3.625],
                'max_lr': [-2.961, -2.807],
                'cycle_peak': [0.444, 0.477],
            },
        }
    }
}

_BOUNDS50 = {
    'mlp': {
        'cifar100': {
            'max': {
                'beta1': [-1.199, -0.751],
                'beta2': [-4.168, -3.108],
                'eps': [-7.922, -5.949],
                'wd': [-3.474, -3.42],
                'max_lr': [-2.925, -2.789],
                'cycle_peak': [0.42, 0.436],
            },
            'relu': {
                'beta1': [-1.316, -0.576],
                'beta2': [-3.051, -2.529],
                'eps': [-9.798, -8.329],
                'wd': [-3.537, -3.362],
                'max_lr': [-3.309, -3.032],
                'cycle_peak': [0.397, 0.483],
            },
        }
    },
    'cnn': {
        'cifar100': {
            'max': {
                'beta1': [-1.295, -1.061],
                'beta2': [-3.659, -3.348],
                'eps': [-6.974, -5.845],
                'wd': [-3.433, -3.169],
                'max_lr': [-2.704, -2.519],
                'cycle_peak': [0.436, 0.454],
            },
            'relu': {
                'beta1': [-0.573, -0.383],
                'beta2': [-3.91, -3.082],
                'eps': [-9.326, -8.077],
                'wd': [-3.569, -3.361],
                'max_lr': [-2.768, -2.703],
                'cycle_peak': [0.394, 0.423],
            },
        }
    }
}

_BOUNDS100 = {
    'mlp': {
        'cifar100': {
            'max': {
                'beta1': [-0.766, -0.614],
                'beta2': [-4.015, -3.73],
                'eps': [-6.571, -5.507],
                'wd': [-2.661, -2.474],
                'max_lr': [-3.234, -3.209],
                'cycle_peak': [0.343, 0.394],
            },
            'relu': {
                'beta1': [-1.417, -0.657],
                'beta2': [-5.101, -3.247],
                'eps': [-7.389, -5.901],
                'wd': [-2.818, -2.644],
                'max_lr': [-3.239, -3.141],
                'cycle_peak': [0.388, 0.446],
            },
        }
    },
    'cnn': {
        'cifar100': {
            'max': {
                'beta1': [-0.785, -0.458],
                'beta2': [-4.821, -4.485],
                'eps': [-6.847, -6.286],
                'wd': [-3.535, -3.376],
                'max_lr': [-2.637, -2.41],
                'cycle_peak': [0.364, 0.402],
            },
            'relu': {
                'beta1': [-0.836, -0.165],
                'beta2': [-5.206, -4.509],
                'eps': [-7.161, -5.8],
                'wd': [-3.645, -3.173],
                'max_lr': [-2.679, -2.482],
                'cycle_peak': [0.328, 0.421],
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