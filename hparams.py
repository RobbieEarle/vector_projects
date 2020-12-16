import numpy as np
import util


_BOUNDS = {
    'mlp': {
        'cifar100': {
            'relu': {
                'b1': [-3, -0.5],
                'b2': [-5, -1],
                'eps': [-10, -6],
                'wd': [-4, 0],
                'lr': [-4, 0],
                'cp': [0.05, 0.5]
            },
            'max': {
                'b1': [-3, -0.5],
                'b2': [-5, -1],
                'eps': [-10, -6],
                'wd': [-4, 0],
                'lr': [-4, 0],
                'cp': [0.05, 0.5]
            }
        }
    },
    'cnn': {
        'cifar100': {
            'relu': {
                'b1': [-3, -0.5],
                'b2': [-5, -1],
                'eps': [-10, -6],
                'wd': [-4, 0],
                'lr': [-4, 0],
                'cp': [0.05, 0.5]
            },
            'max': {
                'b1': [-3, -0.5],
                'b2': [-5, -1],
                'eps': [-10, -6],
                'wd': [-4, 0],
                'lr': [-4, 0],
                'cp': [0.05, 0.5]
            }
        }
    }
}


def get_hparams(model, dataset, actfun, seed):

    util.seed_all(seed)
    rng = np.random.RandomState(seed)
    b = _BOUNDS[model][dataset][actfun]
    hparams = {"beta1": 1 - np.power(10., rng.uniform(b['b1'][0], b['b1'][1])),
               "beta2": 1 - np.power(10., rng.uniform(b['b2'][0], b['b2'][1])),
               "eps": np.power(10., rng.uniform(b['eps'][0], b['eps'][1])),
               "wd": np.power(10., rng.uniform(b['wd'][0], b['wd'][1])),
               "max_lr": np.power(10., rng.uniform(b['lr'][0], b['lr'][1])),
               "cycle_peak": rng.uniform(b['cp'][0], b['cp'][1])
               }

    return hparams