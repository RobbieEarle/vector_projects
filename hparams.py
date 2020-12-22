import numpy as np
import util

_BOUNDS10 = {
    'mlp': {
        'mnist': {
            'max': {
                'beta1': [-3, -0.5],
                'beta2': [-5, -1],
                'eps': [-10, -6],
                'wd': [-3, -7],
                'max_lr': [-4, 0],
                'cycle_peak': [0.1, 0.5]
            },
            'relu': {
                'beta1': [-3, -0.5],
                'beta2': [-5, -1],
                'eps': [-10, -6],
                'wd': [-3, -7],
                'max_lr': [-4, 0],
                'cycle_peak': [0.1, 0.5]
            },
            'bin_all_max_min': {
                'beta1': [-3, -0.5],
                'beta2': [-5, -1],
                'eps': [-10, -6],
                'wd': [-3, -7],
                'max_lr': [-4, 0],
                'cycle_peak': [0.1, 0.5]
            },
            'ail_xnor': {
                'beta1': [-3, -0.5],
                'beta2': [-5, -1],
                'eps': [-10, -6],
                'wd': [-3, -7],
                'max_lr': [-4, 0],
                'cycle_peak': [0.1, 0.5]
            },
            'ail_or': {
                'beta1': [-3, -0.5],
                'beta2': [-5, -1],
                'eps': [-10, -6],
                'wd': [-3, -7],
                'max_lr': [-4, 0],
                'cycle_peak': [0.1, 0.5]
            },
            'ail_all_or_and': {
                'beta1': [-3, -0.5],
                'beta2': [-5, -1],
                'eps': [-10, -6],
                'wd': [-3, -7],
                'max_lr': [-4, 0],
                'cycle_peak': [0.1, 0.5]
            },
            'ail_all_or_xnor': {
                'beta1': [-3, -0.5],
                'beta2': [-5, -1],
                'eps': [-10, -6],
                'wd': [-3, -7],
                'max_lr': [-4, 0],
                'cycle_peak': [0.1, 0.5]
            },
            'ail_all_or_and_xnor': {
                'beta1': [-3, -0.5],
                'beta2': [-5, -1],
                'eps': [-10, -6],
                'wd': [-3, -7],
                'max_lr': [-4, 0],
                'cycle_peak': [0.1, 0.5]
            },
            'ail_part_or_xnor': {
                'beta1': [-3, -0.5],
                'beta2': [-5, -1],
                'eps': [-10, -6],
                'wd': [-3, -7],
                'max_lr': [-4, 0],
                'cycle_peak': [0.1, 0.5]
            },
            'ail_part_or_and_xnor': {
                'beta1': [-3, -0.5],
                'beta2': [-5, -1],
                'eps': [-10, -6],
                'wd': [-3, -7],
                'max_lr': [-4, 0],
                'cycle_peak': [0.1, 0.5]
            },
        },
        'cifar10': {
            'max': {
                'beta1': [-3, -0.5],
                'beta2': [-5, -1],
                'eps': [-10, -6],
                'wd': [-3, -7],
                'max_lr': [-4, 0],
                'cycle_peak': [0.1, 0.5]
            },
            'relu': {
                'beta1': [-3, -0.5],
                'beta2': [-5, -1],
                'eps': [-10, -6],
                'wd': [-3, -7],
                'max_lr': [-4, 0],
                'cycle_peak': [0.1, 0.5]
            },
            'bin_all_max_min': {
                'beta1': [-3, -0.5],
                'beta2': [-5, -1],
                'eps': [-10, -6],
                'wd': [-3, -7],
                'max_lr': [-4, 0],
                'cycle_peak': [0.1, 0.5]
            },
            'ail_xnor': {
                'beta1': [-3, -0.5],
                'beta2': [-5, -1],
                'eps': [-10, -6],
                'wd': [-3, -7],
                'max_lr': [-4, 0],
                'cycle_peak': [0.1, 0.5]
            },
            'ail_or': {
                'beta1': [-3, -0.5],
                'beta2': [-5, -1],
                'eps': [-10, -6],
                'wd': [-3, -7],
                'max_lr': [-4, 0],
                'cycle_peak': [0.1, 0.5]
            },
            'ail_all_or_and': {
                'beta1': [-3, -0.5],
                'beta2': [-5, -1],
                'eps': [-10, -6],
                'wd': [-3, -7],
                'max_lr': [-4, 0],
                'cycle_peak': [0.1, 0.5]
            },
            'ail_all_or_xnor': {
                'beta1': [-3, -0.5],
                'beta2': [-5, -1],
                'eps': [-10, -6],
                'wd': [-3, -7],
                'max_lr': [-4, 0],
                'cycle_peak': [0.1, 0.5]
            },
            'ail_all_or_and_xnor': {
                'beta1': [-3, -0.5],
                'beta2': [-5, -1],
                'eps': [-10, -6],
                'wd': [-3, -7],
                'max_lr': [-4, 0],
                'cycle_peak': [0.1, 0.5]
            },
            'ail_part_or_xnor': {
                'beta1': [-3, -0.5],
                'beta2': [-5, -1],
                'eps': [-10, -6],
                'wd': [-3, -7],
                'max_lr': [-4, 0],
                'cycle_peak': [0.1, 0.5]
            },
            'ail_part_or_and_xnor': {
                'beta1': [-3, -0.5],
                'beta2': [-5, -1],
                'eps': [-10, -6],
                'wd': [-3, -7],
                'max_lr': [-4, 0],
                'cycle_peak': [0.1, 0.5]
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
            'bin_all_max_min': {
                'beta1': [-3, -0.5],
                'beta2': [-5, -1],
                'eps': [-10, -6],
                'wd': [-3, -7],
                'max_lr': [-4, 0],
                'cycle_peak': [0.1, 0.5]
            },
            'ail_xnor': {
                'beta1': [-3, -0.5],
                'beta2': [-5, -1],
                'eps': [-10, -6],
                'wd': [-3, -7],
                'max_lr': [-4, 0],
                'cycle_peak': [0.1, 0.5]
            },
            'ail_or': {
                'beta1': [-3, -0.5],
                'beta2': [-5, -1],
                'eps': [-10, -6],
                'wd': [-3, -7],
                'max_lr': [-4, 0],
                'cycle_peak': [0.1, 0.5]
            },
            'ail_all_or_and': {
                'beta1': [-3, -0.5],
                'beta2': [-5, -1],
                'eps': [-10, -6],
                'wd': [-3, -7],
                'max_lr': [-4, 0],
                'cycle_peak': [0.1, 0.5]
            },
            'ail_all_or_xnor': {
                'beta1': [-3, -0.5],
                'beta2': [-5, -1],
                'eps': [-10, -6],
                'wd': [-3, -7],
                'max_lr': [-4, 0],
                'cycle_peak': [0.1, 0.5]
            },
            'ail_all_or_and_xnor': {
                'beta1': [-3, -0.5],
                'beta2': [-5, -1],
                'eps': [-10, -6],
                'wd': [-3, -7],
                'max_lr': [-4, 0],
                'cycle_peak': [0.1, 0.5]
            },
            'ail_part_or_xnor': {
                'beta1': [-3, -0.5],
                'beta2': [-5, -1],
                'eps': [-10, -6],
                'wd': [-3, -7],
                'max_lr': [-4, 0],
                'cycle_peak': [0.1, 0.5]
            },
            'ail_part_or_and_xnor': {
                'beta1': [-3, -0.5],
                'beta2': [-5, -1],
                'eps': [-10, -6],
                'wd': [-3, -7],
                'max_lr': [-4, 0],
                'cycle_peak': [0.1, 0.5]
            },
        }
    },
    'cnn': {
        'mnist': {
            'max': {
                'beta1': [-3, -0.5],
                'beta2': [-5, -1],
                'eps': [-10, -6],
                'wd': [-3, -7],
                'max_lr': [-4, 0],
                'cycle_peak': [0.1, 0.5]
            },
            'relu': {
                'beta1': [-3, -0.5],
                'beta2': [-5, -1],
                'eps': [-10, -6],
                'wd': [-3, -7],
                'max_lr': [-4, 0],
                'cycle_peak': [0.1, 0.5]
            },
            'bin_all_max_min': {
                'beta1': [-3, -0.5],
                'beta2': [-5, -1],
                'eps': [-10, -6],
                'wd': [-3, -7],
                'max_lr': [-4, 0],
                'cycle_peak': [0.1, 0.5]
            },
            'ail_xnor': {
                'beta1': [-3, -0.5],
                'beta2': [-5, -1],
                'eps': [-10, -6],
                'wd': [-3, -7],
                'max_lr': [-4, 0],
                'cycle_peak': [0.1, 0.5]
            },
            'ail_or': {
                'beta1': [-3, -0.5],
                'beta2': [-5, -1],
                'eps': [-10, -6],
                'wd': [-3, -7],
                'max_lr': [-4, 0],
                'cycle_peak': [0.1, 0.5]
            },
            'ail_all_or_and': {
                'beta1': [-3, -0.5],
                'beta2': [-5, -1],
                'eps': [-10, -6],
                'wd': [-3, -7],
                'max_lr': [-4, 0],
                'cycle_peak': [0.1, 0.5]
            },
            'ail_all_or_xnor': {
                'beta1': [-3, -0.5],
                'beta2': [-5, -1],
                'eps': [-10, -6],
                'wd': [-3, -7],
                'max_lr': [-4, 0],
                'cycle_peak': [0.1, 0.5]
            },
            'ail_all_or_and_xnor': {
                'beta1': [-3, -0.5],
                'beta2': [-5, -1],
                'eps': [-10, -6],
                'wd': [-3, -7],
                'max_lr': [-4, 0],
                'cycle_peak': [0.1, 0.5]
            },
            'ail_part_or_xnor': {
                'beta1': [-3, -0.5],
                'beta2': [-5, -1],
                'eps': [-10, -6],
                'wd': [-3, -7],
                'max_lr': [-4, 0],
                'cycle_peak': [0.1, 0.5]
            },
            'ail_part_or_and_xnor': {
                'beta1': [-3, -0.5],
                'beta2': [-5, -1],
                'eps': [-10, -6],
                'wd': [-3, -7],
                'max_lr': [-4, 0],
                'cycle_peak': [0.1, 0.5]
            },
        },
        'cifar10': {
            'max': {
                'beta1': [-3, -0.5],
                'beta2': [-5, -1],
                'eps': [-10, -6],
                'wd': [-3, -7],
                'max_lr': [-4, 0],
                'cycle_peak': [0.1, 0.5]
            },
            'relu': {
                'beta1': [-3, -0.5],
                'beta2': [-5, -1],
                'eps': [-10, -6],
                'wd': [-3, -7],
                'max_lr': [-4, 0],
                'cycle_peak': [0.1, 0.5]
            },
            'bin_all_max_min': {
                'beta1': [-3, -0.5],
                'beta2': [-5, -1],
                'eps': [-10, -6],
                'wd': [-3, -7],
                'max_lr': [-4, 0],
                'cycle_peak': [0.1, 0.5]
            },
            'ail_xnor': {
                'beta1': [-3, -0.5],
                'beta2': [-5, -1],
                'eps': [-10, -6],
                'wd': [-3, -7],
                'max_lr': [-4, 0],
                'cycle_peak': [0.1, 0.5]
            },
            'ail_or': {
                'beta1': [-3, -0.5],
                'beta2': [-5, -1],
                'eps': [-10, -6],
                'wd': [-3, -7],
                'max_lr': [-4, 0],
                'cycle_peak': [0.1, 0.5]
            },
            'ail_all_or_and': {
                'beta1': [-3, -0.5],
                'beta2': [-5, -1],
                'eps': [-10, -6],
                'wd': [-3, -7],
                'max_lr': [-4, 0],
                'cycle_peak': [0.1, 0.5]
            },
            'ail_all_or_xnor': {
                'beta1': [-3, -0.5],
                'beta2': [-5, -1],
                'eps': [-10, -6],
                'wd': [-3, -7],
                'max_lr': [-4, 0],
                'cycle_peak': [0.1, 0.5]
            },
            'ail_all_or_and_xnor': {
                'beta1': [-3, -0.5],
                'beta2': [-5, -1],
                'eps': [-10, -6],
                'wd': [-3, -7],
                'max_lr': [-4, 0],
                'cycle_peak': [0.1, 0.5]
            },
            'ail_part_or_xnor': {
                'beta1': [-3, -0.5],
                'beta2': [-5, -1],
                'eps': [-10, -6],
                'wd': [-3, -7],
                'max_lr': [-4, 0],
                'cycle_peak': [0.1, 0.5]
            },
            'ail_part_or_and_xnor': {
                'beta1': [-3, -0.5],
                'beta2': [-5, -1],
                'eps': [-10, -6],
                'wd': [-3, -7],
                'max_lr': [-4, 0],
                'cycle_peak': [0.1, 0.5]
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
            'bin_all_max_min': {
                'beta1': [-3, -0.5],
                'beta2': [-5, -1],
                'eps': [-10, -6],
                'wd': [-3, -7],
                'max_lr': [-4, 0],
                'cycle_peak': [0.1, 0.5]
            },
            'ail_xnor': {
                'beta1': [-3, -0.5],
                'beta2': [-5, -1],
                'eps': [-10, -6],
                'wd': [-3, -7],
                'max_lr': [-4, 0],
                'cycle_peak': [0.1, 0.5]
            },
            'ail_or': {
                'beta1': [-3, -0.5],
                'beta2': [-5, -1],
                'eps': [-10, -6],
                'wd': [-3, -7],
                'max_lr': [-4, 0],
                'cycle_peak': [0.1, 0.5]
            },
            'ail_all_or_and': {
                'beta1': [-3, -0.5],
                'beta2': [-5, -1],
                'eps': [-10, -6],
                'wd': [-3, -7],
                'max_lr': [-4, 0],
                'cycle_peak': [0.1, 0.5]
            },
            'ail_all_or_xnor': {
                'beta1': [-3, -0.5],
                'beta2': [-5, -1],
                'eps': [-10, -6],
                'wd': [-3, -7],
                'max_lr': [-4, 0],
                'cycle_peak': [0.1, 0.5]
            },
            'ail_all_or_and_xnor': {
                'beta1': [-3, -0.5],
                'beta2': [-5, -1],
                'eps': [-10, -6],
                'wd': [-3, -7],
                'max_lr': [-4, 0],
                'cycle_peak': [0.1, 0.5]
            },
            'ail_part_or_xnor': {
                'beta1': [-3, -0.5],
                'beta2': [-5, -1],
                'eps': [-10, -6],
                'wd': [-3, -7],
                'max_lr': [-4, 0],
                'cycle_peak': [0.1, 0.5]
            },
            'ail_part_or_and_xnor': {
                'beta1': [-3, -0.5],
                'beta2': [-5, -1],
                'eps': [-10, -6],
                'wd': [-3, -7],
                'max_lr': [-4, 0],
                'cycle_peak': [0.1, 0.5]
            },
        }
    }
}

_BOUNDS50 = {
    'mlp': {
        'mnist': {
            'max': {
                'beta1': [-3, -0.5],
                'beta2': [-5, -1],
                'eps': [-10, -6],
                'wd': [-3, -7],
                'max_lr': [-4, 0],
                'cycle_peak': [0.1, 0.5]
            },
            'relu': {
                'beta1': [-3, -0.5],
                'beta2': [-5, -1],
                'eps': [-10, -6],
                'wd': [-3, -7],
                'max_lr': [-4, 0],
                'cycle_peak': [0.1, 0.5]
            },
            'bin_all_max_min': {
                'beta1': [-3, -0.5],
                'beta2': [-5, -1],
                'eps': [-10, -6],
                'wd': [-3, -7],
                'max_lr': [-4, 0],
                'cycle_peak': [0.1, 0.5]
            },
            'ail_xnor': {
                'beta1': [-3, -0.5],
                'beta2': [-5, -1],
                'eps': [-10, -6],
                'wd': [-3, -7],
                'max_lr': [-4, 0],
                'cycle_peak': [0.1, 0.5]
            },
            'ail_or': {
                'beta1': [-3, -0.5],
                'beta2': [-5, -1],
                'eps': [-10, -6],
                'wd': [-3, -7],
                'max_lr': [-4, 0],
                'cycle_peak': [0.1, 0.5]
            },
            'ail_all_or_and': {
                'beta1': [-3, -0.5],
                'beta2': [-5, -1],
                'eps': [-10, -6],
                'wd': [-3, -7],
                'max_lr': [-4, 0],
                'cycle_peak': [0.1, 0.5]
            },
            'ail_all_or_xnor': {
                'beta1': [-3, -0.5],
                'beta2': [-5, -1],
                'eps': [-10, -6],
                'wd': [-3, -7],
                'max_lr': [-4, 0],
                'cycle_peak': [0.1, 0.5]
            },
            'ail_all_or_and_xnor': {
                'beta1': [-3, -0.5],
                'beta2': [-5, -1],
                'eps': [-10, -6],
                'wd': [-3, -7],
                'max_lr': [-4, 0],
                'cycle_peak': [0.1, 0.5]
            },
            'ail_part_or_xnor': {
                'beta1': [-3, -0.5],
                'beta2': [-5, -1],
                'eps': [-10, -6],
                'wd': [-3, -7],
                'max_lr': [-4, 0],
                'cycle_peak': [0.1, 0.5]
            },
            'ail_part_or_and_xnor': {
                'beta1': [-3, -0.5],
                'beta2': [-5, -1],
                'eps': [-10, -6],
                'wd': [-3, -7],
                'max_lr': [-4, 0],
                'cycle_peak': [0.1, 0.5]
            },
        },
        'cifar10': {
            'max': {
                'beta1': [-3, -0.5],
                'beta2': [-5, -1],
                'eps': [-10, -6],
                'wd': [-3, -7],
                'max_lr': [-4, 0],
                'cycle_peak': [0.1, 0.5]
            },
            'relu': {
                'beta1': [-3, -0.5],
                'beta2': [-5, -1],
                'eps': [-10, -6],
                'wd': [-3, -7],
                'max_lr': [-4, 0],
                'cycle_peak': [0.1, 0.5]
            },
            'bin_all_max_min': {
                'beta1': [-3, -0.5],
                'beta2': [-5, -1],
                'eps': [-10, -6],
                'wd': [-3, -7],
                'max_lr': [-4, 0],
                'cycle_peak': [0.1, 0.5]
            },
            'ail_xnor': {
                'beta1': [-3, -0.5],
                'beta2': [-5, -1],
                'eps': [-10, -6],
                'wd': [-3, -7],
                'max_lr': [-4, 0],
                'cycle_peak': [0.1, 0.5]
            },
            'ail_or': {
                'beta1': [-3, -0.5],
                'beta2': [-5, -1],
                'eps': [-10, -6],
                'wd': [-3, -7],
                'max_lr': [-4, 0],
                'cycle_peak': [0.1, 0.5]
            },
            'ail_all_or_and': {
                'beta1': [-3, -0.5],
                'beta2': [-5, -1],
                'eps': [-10, -6],
                'wd': [-3, -7],
                'max_lr': [-4, 0],
                'cycle_peak': [0.1, 0.5]
            },
            'ail_all_or_xnor': {
                'beta1': [-3, -0.5],
                'beta2': [-5, -1],
                'eps': [-10, -6],
                'wd': [-3, -7],
                'max_lr': [-4, 0],
                'cycle_peak': [0.1, 0.5]
            },
            'ail_all_or_and_xnor': {
                'beta1': [-3, -0.5],
                'beta2': [-5, -1],
                'eps': [-10, -6],
                'wd': [-3, -7],
                'max_lr': [-4, 0],
                'cycle_peak': [0.1, 0.5]
            },
            'ail_part_or_xnor': {
                'beta1': [-3, -0.5],
                'beta2': [-5, -1],
                'eps': [-10, -6],
                'wd': [-3, -7],
                'max_lr': [-4, 0],
                'cycle_peak': [0.1, 0.5]
            },
            'ail_part_or_and_xnor': {
                'beta1': [-3, -0.5],
                'beta2': [-5, -1],
                'eps': [-10, -6],
                'wd': [-3, -7],
                'max_lr': [-4, 0],
                'cycle_peak': [0.1, 0.5]
            },
        },
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
            'bin_all_max_min': {
                'beta1': [-3, -0.5],
                'beta2': [-5, -1],
                'eps': [-10, -6],
                'wd': [-3, -7],
                'max_lr': [-4, 0],
                'cycle_peak': [0.1, 0.5]
            },
            'ail_xnor': {
                'beta1': [-3, -0.5],
                'beta2': [-5, -1],
                'eps': [-10, -6],
                'wd': [-3, -7],
                'max_lr': [-4, 0],
                'cycle_peak': [0.1, 0.5]
            },
            'ail_or': {
                'beta1': [-3, -0.5],
                'beta2': [-5, -1],
                'eps': [-10, -6],
                'wd': [-3, -7],
                'max_lr': [-4, 0],
                'cycle_peak': [0.1, 0.5]
            },
            'ail_all_or_and': {
                'beta1': [-3, -0.5],
                'beta2': [-5, -1],
                'eps': [-10, -6],
                'wd': [-3, -7],
                'max_lr': [-4, 0],
                'cycle_peak': [0.1, 0.5]
            },
            'ail_all_or_xnor': {
                'beta1': [-3, -0.5],
                'beta2': [-5, -1],
                'eps': [-10, -6],
                'wd': [-3, -7],
                'max_lr': [-4, 0],
                'cycle_peak': [0.1, 0.5]
            },
            'ail_all_or_and_xnor': {
                'beta1': [-3, -0.5],
                'beta2': [-5, -1],
                'eps': [-10, -6],
                'wd': [-3, -7],
                'max_lr': [-4, 0],
                'cycle_peak': [0.1, 0.5]
            },
            'ail_part_or_xnor': {
                'beta1': [-3, -0.5],
                'beta2': [-5, -1],
                'eps': [-10, -6],
                'wd': [-3, -7],
                'max_lr': [-4, 0],
                'cycle_peak': [0.1, 0.5]
            },
            'ail_part_or_and_xnor': {
                'beta1': [-3, -0.5],
                'beta2': [-5, -1],
                'eps': [-10, -6],
                'wd': [-3, -7],
                'max_lr': [-4, 0],
                'cycle_peak': [0.1, 0.5]
            },
        }
    },
    'cnn': {
        'mnist': {
            'max': {
                'beta1': [-3, -0.5],
                'beta2': [-5, -1],
                'eps': [-10, -6],
                'wd': [-3, -7],
                'max_lr': [-4, 0],
                'cycle_peak': [0.1, 0.5]
            },
            'relu': {
                'beta1': [-3, -0.5],
                'beta2': [-5, -1],
                'eps': [-10, -6],
                'wd': [-3, -7],
                'max_lr': [-4, 0],
                'cycle_peak': [0.1, 0.5]
            },
            'bin_all_max_min': {
                'beta1': [-3, -0.5],
                'beta2': [-5, -1],
                'eps': [-10, -6],
                'wd': [-3, -7],
                'max_lr': [-4, 0],
                'cycle_peak': [0.1, 0.5]
            },
            'ail_xnor': {
                'beta1': [-3, -0.5],
                'beta2': [-5, -1],
                'eps': [-10, -6],
                'wd': [-3, -7],
                'max_lr': [-4, 0],
                'cycle_peak': [0.1, 0.5]
            },
            'ail_or': {
                'beta1': [-3, -0.5],
                'beta2': [-5, -1],
                'eps': [-10, -6],
                'wd': [-3, -7],
                'max_lr': [-4, 0],
                'cycle_peak': [0.1, 0.5]
            },
            'ail_all_or_and': {
                'beta1': [-3, -0.5],
                'beta2': [-5, -1],
                'eps': [-10, -6],
                'wd': [-3, -7],
                'max_lr': [-4, 0],
                'cycle_peak': [0.1, 0.5]
            },
            'ail_all_or_xnor': {
                'beta1': [-3, -0.5],
                'beta2': [-5, -1],
                'eps': [-10, -6],
                'wd': [-3, -7],
                'max_lr': [-4, 0],
                'cycle_peak': [0.1, 0.5]
            },
            'ail_all_or_and_xnor': {
                'beta1': [-3, -0.5],
                'beta2': [-5, -1],
                'eps': [-10, -6],
                'wd': [-3, -7],
                'max_lr': [-4, 0],
                'cycle_peak': [0.1, 0.5]
            },
            'ail_part_or_xnor': {
                'beta1': [-3, -0.5],
                'beta2': [-5, -1],
                'eps': [-10, -6],
                'wd': [-3, -7],
                'max_lr': [-4, 0],
                'cycle_peak': [0.1, 0.5]
            },
            'ail_part_or_and_xnor': {
                'beta1': [-3, -0.5],
                'beta2': [-5, -1],
                'eps': [-10, -6],
                'wd': [-3, -7],
                'max_lr': [-4, 0],
                'cycle_peak': [0.1, 0.5]
            },
        },
        'cifar10': {
            'max': {
                'beta1': [-3, -0.5],
                'beta2': [-5, -1],
                'eps': [-10, -6],
                'wd': [-3, -7],
                'max_lr': [-4, 0],
                'cycle_peak': [0.1, 0.5]
            },
            'relu': {
                'beta1': [-3, -0.5],
                'beta2': [-5, -1],
                'eps': [-10, -6],
                'wd': [-3, -7],
                'max_lr': [-4, 0],
                'cycle_peak': [0.1, 0.5]
            },
            'bin_all_max_min': {
                'beta1': [-3, -0.5],
                'beta2': [-5, -1],
                'eps': [-10, -6],
                'wd': [-3, -7],
                'max_lr': [-4, 0],
                'cycle_peak': [0.1, 0.5]
            },
            'ail_xnor': {
                'beta1': [-3, -0.5],
                'beta2': [-5, -1],
                'eps': [-10, -6],
                'wd': [-3, -7],
                'max_lr': [-4, 0],
                'cycle_peak': [0.1, 0.5]
            },
            'ail_or': {
                'beta1': [-3, -0.5],
                'beta2': [-5, -1],
                'eps': [-10, -6],
                'wd': [-3, -7],
                'max_lr': [-4, 0],
                'cycle_peak': [0.1, 0.5]
            },
            'ail_all_or_and': {
                'beta1': [-3, -0.5],
                'beta2': [-5, -1],
                'eps': [-10, -6],
                'wd': [-3, -7],
                'max_lr': [-4, 0],
                'cycle_peak': [0.1, 0.5]
            },
            'ail_all_or_xnor': {
                'beta1': [-3, -0.5],
                'beta2': [-5, -1],
                'eps': [-10, -6],
                'wd': [-3, -7],
                'max_lr': [-4, 0],
                'cycle_peak': [0.1, 0.5]
            },
            'ail_all_or_and_xnor': {
                'beta1': [-3, -0.5],
                'beta2': [-5, -1],
                'eps': [-10, -6],
                'wd': [-3, -7],
                'max_lr': [-4, 0],
                'cycle_peak': [0.1, 0.5]
            },
            'ail_part_or_xnor': {
                'beta1': [-3, -0.5],
                'beta2': [-5, -1],
                'eps': [-10, -6],
                'wd': [-3, -7],
                'max_lr': [-4, 0],
                'cycle_peak': [0.1, 0.5]
            },
            'ail_part_or_and_xnor': {
                'beta1': [-3, -0.5],
                'beta2': [-5, -1],
                'eps': [-10, -6],
                'wd': [-3, -7],
                'max_lr': [-4, 0],
                'cycle_peak': [0.1, 0.5]
            },
        },
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
            'bin_all_max_min': {
                'beta1': [-3, -0.5],
                'beta2': [-5, -1],
                'eps': [-10, -6],
                'wd': [-3, -7],
                'max_lr': [-4, 0],
                'cycle_peak': [0.1, 0.5]
            },
            'ail_xnor': {
                'beta1': [-3, -0.5],
                'beta2': [-5, -1],
                'eps': [-10, -6],
                'wd': [-3, -7],
                'max_lr': [-4, 0],
                'cycle_peak': [0.1, 0.5]
            },
            'ail_or': {
                'beta1': [-3, -0.5],
                'beta2': [-5, -1],
                'eps': [-10, -6],
                'wd': [-3, -7],
                'max_lr': [-4, 0],
                'cycle_peak': [0.1, 0.5]
            },
            'ail_all_or_and': {
                'beta1': [-3, -0.5],
                'beta2': [-5, -1],
                'eps': [-10, -6],
                'wd': [-3, -7],
                'max_lr': [-4, 0],
                'cycle_peak': [0.1, 0.5]
            },
            'ail_all_or_xnor': {
                'beta1': [-3, -0.5],
                'beta2': [-5, -1],
                'eps': [-10, -6],
                'wd': [-3, -7],
                'max_lr': [-4, 0],
                'cycle_peak': [0.1, 0.5]
            },
            'ail_all_or_and_xnor': {
                'beta1': [-3, -0.5],
                'beta2': [-5, -1],
                'eps': [-10, -6],
                'wd': [-3, -7],
                'max_lr': [-4, 0],
                'cycle_peak': [0.1, 0.5]
            },
            'ail_part_or_xnor': {
                'beta1': [-3, -0.5],
                'beta2': [-5, -1],
                'eps': [-10, -6],
                'wd': [-3, -7],
                'max_lr': [-4, 0],
                'cycle_peak': [0.1, 0.5]
            },
            'ail_part_or_and_xnor': {
                'beta1': [-3, -0.5],
                'beta2': [-5, -1],
                'eps': [-10, -6],
                'wd': [-3, -7],
                'max_lr': [-4, 0],
                'cycle_peak': [0.1, 0.5]
            },
        }
    }
}

_BOUNDS100 = {
    'mlp': {
        'mnist': {
            'max': {
                'beta1': [-3, -0.5],
                'beta2': [-5, -1],
                'eps': [-10, -6],
                'wd': [-3, -7],
                'max_lr': [-4, 0],
                'cycle_peak': [0.1, 0.5]
            },
            'relu': {
                'beta1': [-3, -0.5],
                'beta2': [-5, -1],
                'eps': [-10, -6],
                'wd': [-3, -7],
                'max_lr': [-4, 0],
                'cycle_peak': [0.1, 0.5]
            },
            'bin_all_max_min': {
                'beta1': [-3, -0.5],
                'beta2': [-5, -1],
                'eps': [-10, -6],
                'wd': [-3, -7],
                'max_lr': [-4, 0],
                'cycle_peak': [0.1, 0.5]
            },
            'ail_xnor': {
                'beta1': [-3, -0.5],
                'beta2': [-5, -1],
                'eps': [-10, -6],
                'wd': [-3, -7],
                'max_lr': [-4, 0],
                'cycle_peak': [0.1, 0.5]
            },
            'ail_or': {
                'beta1': [-3, -0.5],
                'beta2': [-5, -1],
                'eps': [-10, -6],
                'wd': [-3, -7],
                'max_lr': [-4, 0],
                'cycle_peak': [0.1, 0.5]
            },
            'ail_all_or_and': {
                'beta1': [-3, -0.5],
                'beta2': [-5, -1],
                'eps': [-10, -6],
                'wd': [-3, -7],
                'max_lr': [-4, 0],
                'cycle_peak': [0.1, 0.5]
            },
            'ail_all_or_xnor': {
                'beta1': [-3, -0.5],
                'beta2': [-5, -1],
                'eps': [-10, -6],
                'wd': [-3, -7],
                'max_lr': [-4, 0],
                'cycle_peak': [0.1, 0.5]
            },
            'ail_all_or_and_xnor': {
                'beta1': [-3, -0.5],
                'beta2': [-5, -1],
                'eps': [-10, -6],
                'wd': [-3, -7],
                'max_lr': [-4, 0],
                'cycle_peak': [0.1, 0.5]
            },
            'ail_part_or_xnor': {
                'beta1': [-3, -0.5],
                'beta2': [-5, -1],
                'eps': [-10, -6],
                'wd': [-3, -7],
                'max_lr': [-4, 0],
                'cycle_peak': [0.1, 0.5]
            },
            'ail_part_or_and_xnor': {
                'beta1': [-3, -0.5],
                'beta2': [-5, -1],
                'eps': [-10, -6],
                'wd': [-3, -7],
                'max_lr': [-4, 0],
                'cycle_peak': [0.1, 0.5]
            },
        },
        'cifar10': {
            'max': {
                'beta1': [-3, -0.5],
                'beta2': [-5, -1],
                'eps': [-10, -6],
                'wd': [-3, -7],
                'max_lr': [-4, 0],
                'cycle_peak': [0.1, 0.5]
            },
            'relu': {
                'beta1': [-3, -0.5],
                'beta2': [-5, -1],
                'eps': [-10, -6],
                'wd': [-3, -7],
                'max_lr': [-4, 0],
                'cycle_peak': [0.1, 0.5]
            },
            'bin_all_max_min': {
                'beta1': [-3, -0.5],
                'beta2': [-5, -1],
                'eps': [-10, -6],
                'wd': [-3, -7],
                'max_lr': [-4, 0],
                'cycle_peak': [0.1, 0.5]
            },
            'ail_xnor': {
                'beta1': [-3, -0.5],
                'beta2': [-5, -1],
                'eps': [-10, -6],
                'wd': [-3, -7],
                'max_lr': [-4, 0],
                'cycle_peak': [0.1, 0.5]
            },
            'ail_or': {
                'beta1': [-3, -0.5],
                'beta2': [-5, -1],
                'eps': [-10, -6],
                'wd': [-3, -7],
                'max_lr': [-4, 0],
                'cycle_peak': [0.1, 0.5]
            },
            'ail_all_or_and': {
                'beta1': [-3, -0.5],
                'beta2': [-5, -1],
                'eps': [-10, -6],
                'wd': [-3, -7],
                'max_lr': [-4, 0],
                'cycle_peak': [0.1, 0.5]
            },
            'ail_all_or_xnor': {
                'beta1': [-3, -0.5],
                'beta2': [-5, -1],
                'eps': [-10, -6],
                'wd': [-3, -7],
                'max_lr': [-4, 0],
                'cycle_peak': [0.1, 0.5]
            },
            'ail_all_or_and_xnor': {
                'beta1': [-3, -0.5],
                'beta2': [-5, -1],
                'eps': [-10, -6],
                'wd': [-3, -7],
                'max_lr': [-4, 0],
                'cycle_peak': [0.1, 0.5]
            },
            'ail_part_or_xnor': {
                'beta1': [-3, -0.5],
                'beta2': [-5, -1],
                'eps': [-10, -6],
                'wd': [-3, -7],
                'max_lr': [-4, 0],
                'cycle_peak': [0.1, 0.5]
            },
            'ail_part_or_and_xnor': {
                'beta1': [-3, -0.5],
                'beta2': [-5, -1],
                'eps': [-10, -6],
                'wd': [-3, -7],
                'max_lr': [-4, 0],
                'cycle_peak': [0.1, 0.5]
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
            'bin_all_max_min': {
                'beta1': [-3, -0.5],
                'beta2': [-5, -1],
                'eps': [-10, -6],
                'wd': [-3, -7],
                'max_lr': [-4, 0],
                'cycle_peak': [0.1, 0.5]
            },
            'ail_xnor': {
                'beta1': [-3, -0.5],
                'beta2': [-5, -1],
                'eps': [-10, -6],
                'wd': [-3, -7],
                'max_lr': [-4, 0],
                'cycle_peak': [0.1, 0.5]
            },
            'ail_or': {
                'beta1': [-3, -0.5],
                'beta2': [-5, -1],
                'eps': [-10, -6],
                'wd': [-3, -7],
                'max_lr': [-4, 0],
                'cycle_peak': [0.1, 0.5]
            },
            'ail_all_or_and': {
                'beta1': [-3, -0.5],
                'beta2': [-5, -1],
                'eps': [-10, -6],
                'wd': [-3, -7],
                'max_lr': [-4, 0],
                'cycle_peak': [0.1, 0.5]
            },
            'ail_all_or_xnor': {
                'beta1': [-3, -0.5],
                'beta2': [-5, -1],
                'eps': [-10, -6],
                'wd': [-3, -7],
                'max_lr': [-4, 0],
                'cycle_peak': [0.1, 0.5]
            },
            'ail_all_or_and_xnor': {
                'beta1': [-3, -0.5],
                'beta2': [-5, -1],
                'eps': [-10, -6],
                'wd': [-3, -7],
                'max_lr': [-4, 0],
                'cycle_peak': [0.1, 0.5]
            },
            'ail_part_or_xnor': {
                'beta1': [-3, -0.5],
                'beta2': [-5, -1],
                'eps': [-10, -6],
                'wd': [-3, -7],
                'max_lr': [-4, 0],
                'cycle_peak': [0.1, 0.5]
            },
            'ail_part_or_and_xnor': {
                'beta1': [-3, -0.5],
                'beta2': [-5, -1],
                'eps': [-10, -6],
                'wd': [-3, -7],
                'max_lr': [-4, 0],
                'cycle_peak': [0.1, 0.5]
            },
        }
    },
    'cnn': {
        'mnist': {
            'max': {
                'beta1': [-3, -0.5],
                'beta2': [-5, -1],
                'eps': [-10, -6],
                'wd': [-3, -7],
                'max_lr': [-4, 0],
                'cycle_peak': [0.1, 0.5]
            },
            'relu': {
                'beta1': [-3, -0.5],
                'beta2': [-5, -1],
                'eps': [-10, -6],
                'wd': [-3, -7],
                'max_lr': [-4, 0],
                'cycle_peak': [0.1, 0.5]
            },
            'bin_all_max_min': {
                'beta1': [-3, -0.5],
                'beta2': [-5, -1],
                'eps': [-10, -6],
                'wd': [-3, -7],
                'max_lr': [-4, 0],
                'cycle_peak': [0.1, 0.5]
            },
            'ail_xnor': {
                'beta1': [-3, -0.5],
                'beta2': [-5, -1],
                'eps': [-10, -6],
                'wd': [-3, -7],
                'max_lr': [-4, 0],
                'cycle_peak': [0.1, 0.5]
            },
            'ail_or': {
                'beta1': [-3, -0.5],
                'beta2': [-5, -1],
                'eps': [-10, -6],
                'wd': [-3, -7],
                'max_lr': [-4, 0],
                'cycle_peak': [0.1, 0.5]
            },
            'ail_all_or_and': {
                'beta1': [-3, -0.5],
                'beta2': [-5, -1],
                'eps': [-10, -6],
                'wd': [-3, -7],
                'max_lr': [-4, 0],
                'cycle_peak': [0.1, 0.5]
            },
            'ail_all_or_xnor': {
                'beta1': [-3, -0.5],
                'beta2': [-5, -1],
                'eps': [-10, -6],
                'wd': [-3, -7],
                'max_lr': [-4, 0],
                'cycle_peak': [0.1, 0.5]
            },
            'ail_all_or_and_xnor': {
                'beta1': [-3, -0.5],
                'beta2': [-5, -1],
                'eps': [-10, -6],
                'wd': [-3, -7],
                'max_lr': [-4, 0],
                'cycle_peak': [0.1, 0.5]
            },
            'ail_part_or_xnor': {
                'beta1': [-3, -0.5],
                'beta2': [-5, -1],
                'eps': [-10, -6],
                'wd': [-3, -7],
                'max_lr': [-4, 0],
                'cycle_peak': [0.1, 0.5]
            },
            'ail_part_or_and_xnor': {
                'beta1': [-3, -0.5],
                'beta2': [-5, -1],
                'eps': [-10, -6],
                'wd': [-3, -7],
                'max_lr': [-4, 0],
                'cycle_peak': [0.1, 0.5]
            },
        },
        'cifar10': {
            'max': {
                'beta1': [-3, -0.5],
                'beta2': [-5, -1],
                'eps': [-10, -6],
                'wd': [-3, -7],
                'max_lr': [-4, 0],
                'cycle_peak': [0.1, 0.5]
            },
            'relu': {
                'beta1': [-3, -0.5],
                'beta2': [-5, -1],
                'eps': [-10, -6],
                'wd': [-3, -7],
                'max_lr': [-4, 0],
                'cycle_peak': [0.1, 0.5]
            },
            'bin_all_max_min': {
                'beta1': [-3, -0.5],
                'beta2': [-5, -1],
                'eps': [-10, -6],
                'wd': [-3, -7],
                'max_lr': [-4, 0],
                'cycle_peak': [0.1, 0.5]
            },
            'ail_xnor': {
                'beta1': [-3, -0.5],
                'beta2': [-5, -1],
                'eps': [-10, -6],
                'wd': [-3, -7],
                'max_lr': [-4, 0],
                'cycle_peak': [0.1, 0.5]
            },
            'ail_or': {
                'beta1': [-3, -0.5],
                'beta2': [-5, -1],
                'eps': [-10, -6],
                'wd': [-3, -7],
                'max_lr': [-4, 0],
                'cycle_peak': [0.1, 0.5]
            },
            'ail_all_or_and': {
                'beta1': [-3, -0.5],
                'beta2': [-5, -1],
                'eps': [-10, -6],
                'wd': [-3, -7],
                'max_lr': [-4, 0],
                'cycle_peak': [0.1, 0.5]
            },
            'ail_all_or_xnor': {
                'beta1': [-3, -0.5],
                'beta2': [-5, -1],
                'eps': [-10, -6],
                'wd': [-3, -7],
                'max_lr': [-4, 0],
                'cycle_peak': [0.1, 0.5]
            },
            'ail_all_or_and_xnor': {
                'beta1': [-3, -0.5],
                'beta2': [-5, -1],
                'eps': [-10, -6],
                'wd': [-3, -7],
                'max_lr': [-4, 0],
                'cycle_peak': [0.1, 0.5]
            },
            'ail_part_or_xnor': {
                'beta1': [-3, -0.5],
                'beta2': [-5, -1],
                'eps': [-10, -6],
                'wd': [-3, -7],
                'max_lr': [-4, 0],
                'cycle_peak': [0.1, 0.5]
            },
            'ail_part_or_and_xnor': {
                'beta1': [-3, -0.5],
                'beta2': [-5, -1],
                'eps': [-10, -6],
                'wd': [-3, -7],
                'max_lr': [-4, 0],
                'cycle_peak': [0.1, 0.5]
            },
        },
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
            'bin_all_max_min': {
                'beta1': [-3, -0.5],
                'beta2': [-5, -1],
                'eps': [-10, -6],
                'wd': [-3, -7],
                'max_lr': [-4, 0],
                'cycle_peak': [0.1, 0.5]
            },
            'ail_xnor': {
                'beta1': [-3, -0.5],
                'beta2': [-5, -1],
                'eps': [-10, -6],
                'wd': [-3, -7],
                'max_lr': [-4, 0],
                'cycle_peak': [0.1, 0.5]
            },
            'ail_or': {
                'beta1': [-3, -0.5],
                'beta2': [-5, -1],
                'eps': [-10, -6],
                'wd': [-3, -7],
                'max_lr': [-4, 0],
                'cycle_peak': [0.1, 0.5]
            },
            'ail_all_or_and': {
                'beta1': [-3, -0.5],
                'beta2': [-5, -1],
                'eps': [-10, -6],
                'wd': [-3, -7],
                'max_lr': [-4, 0],
                'cycle_peak': [0.1, 0.5]
            },
            'ail_all_or_xnor': {
                'beta1': [-3, -0.5],
                'beta2': [-5, -1],
                'eps': [-10, -6],
                'wd': [-3, -7],
                'max_lr': [-4, 0],
                'cycle_peak': [0.1, 0.5]
            },
            'ail_all_or_and_xnor': {
                'beta1': [-3, -0.5],
                'beta2': [-5, -1],
                'eps': [-10, -6],
                'wd': [-3, -7],
                'max_lr': [-4, 0],
                'cycle_peak': [0.1, 0.5]
            },
            'ail_part_or_xnor': {
                'beta1': [-3, -0.5],
                'beta2': [-5, -1],
                'eps': [-10, -6],
                'wd': [-3, -7],
                'max_lr': [-4, 0],
                'cycle_peak': [0.1, 0.5]
            },
            'ail_part_or_and_xnor': {
                'beta1': [-3, -0.5],
                'beta2': [-5, -1],
                'eps': [-10, -6],
                'wd': [-3, -7],
                'max_lr': [-4, 0],
                'cycle_peak': [0.1, 0.5]
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