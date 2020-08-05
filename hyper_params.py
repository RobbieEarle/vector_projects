import numpy as np


def get_hyper_params(model, dataset, actfun, rng=None):

    if model == 'nn':
        if dataset == 'mnist':
            return nn_mnist(rng)[actfun]
        elif dataset == 'cifar10':
            return nn_cifar10(rng)[actfun]
        elif dataset == 'cifar100':
            return nn_cifar100(rng)[actfun]
        elif dataset == 'fashion_mnist':
            return nn_fashion_mnist(rng)[actfun]
        elif dataset == 'svhn':
            return nn_svhn(rng)[actfun]
    elif model == 'cnn':
        if dataset == 'mnist':
            return cnn_mnist(rng)[actfun]
        elif dataset == 'cifar10':
            return cnn_cifar10(rng)[actfun]
        elif dataset == 'cifar100':
            return cnn_cifar100(rng)[actfun]
        elif dataset == 'fashion_mnist':
            return cnn_fashion_mnist(rng)[actfun]
        elif dataset == 'svhn':
            return cnn_svhn(rng)[actfun]
    else:
        raise ValueError("Error: No hyper-parameters found for {}, {}".format(model, dataset))


# Opt
def nn_mnist(rng):
    return {
        "relu": {"adam_beta_1": np.exp(-2.375018573261741),
                 "adam_beta_2": np.exp(-6.565065478550015),
                 "adam_eps": np.exp(-19.607731090387627),
                 "adam_wd": np.exp(-11.86635747404571),
                 "max_lr": np.exp(-5.7662952418075175),
                 "cycle_peak": 0.2935155263985412
                 },
        "cf_relu": {"adam_beta_1": np.exp(-4.44857338551192),
                    "adam_beta_2": np.exp(-4.669825410890087),
                    "adam_eps": np.exp(-17.69933166220988),
                    "adam_wd": np.exp(-12.283288733512373),
                    "max_lr": np.exp(-8.563504990329884),
                    "cycle_peak": 0.10393251332079881
                    },
        "multi_relu": {"adam_beta_1": np.exp(-2.859441513546877),
                       "adam_beta_2": np.exp(-5.617992566623951),
                       "adam_eps": np.exp(-20.559015044774018),
                       "adam_wd": np.exp(-12.693844976989661),
                       "max_lr": np.exp(-5.802816398828524),
                       "cycle_peak": 0.28499869111025217
                       },
        "combinact": {"adam_beta_1": np.exp(-2.6436039683427253),
                      "adam_beta_2": np.exp(-7.371516988658699),
                      "adam_eps": np.exp(-16.989022147994522),
                      "adam_wd": np.exp(-12.113778466374383),
                      "max_lr": np.exp(-8),
                      "cycle_peak": 0.4661308739740898
                      },
        "l2": {"adam_beta_1": np.exp(-2.244614412525641),
               "adam_beta_2": np.exp(-5.502197648895974),
               "adam_eps": np.exp(-16.919215725249092),
               "adam_wd": np.exp(-13.99956243808541),
               "max_lr": np.exp(-5.383090612225605),
               "cycle_peak": 0.35037784343793205
               },
        "abs": {"adam_beta_1": np.exp(-3.1576858739457845),
                "adam_beta_2": np.exp(-4.165206705873042),
                "adam_eps": np.exp(-20.430988799955056),
                "adam_wd": np.exp(-13.049933891070697),
                "max_lr": np.exp(-5.809683797646132),
                "cycle_peak": 0.34244342851740034
                },
        "cf_abs": {"adam_beta_1": np.exp(-5.453380890632929),
                   "adam_beta_2": np.exp(-5.879222236954101),
                   "adam_eps": np.exp(-18.303333640483068),
                   "adam_wd": np.exp(-15.152599023560422),
                   "max_lr": np.exp(-6.604045812173043),
                   "cycle_peak": 0.11189158130301018
                   },
        "l2_lae": {"adam_beta_1": np.exp(-2.4561852034212),
                   "adam_beta_2": np.exp(-5.176943480470942),
                   "adam_eps": np.exp(-16.032458209235187),
                   "adam_wd": np.exp(-12.860274699438266),
                   "max_lr": np.exp(-5.540947578537945),
                   "cycle_peak": 0.40750994546983904
                   },
        "max": {"adam_beta_1": np.exp(-2.2169207045481505),
                "adam_beta_2": np.exp(-7.793567052557596),
                "adam_eps": np.exp(-18.23187258333265),
                "adam_wd": np.exp(-12.867866026516422),
                "max_lr": np.exp(-5.416840501318637),
                "cycle_peak": 0.28254869607601146
                },
        "prod": {"adam_beta_1": np.exp(rng.uniform(-5, 0)),
                 "adam_beta_2": np.exp(rng.uniform(-12, -6)),
                 "adam_eps": np.exp(rng.uniform(-22, -15)),
                 "adam_wd": np.exp(rng.uniform(-15, -9)),
                 "max_lr": np.exp(rng.uniform(-6, -2)),
                 "cycle_peak": rng.uniform(0.1, 0.5)
                 },
        "binary_ops_partition": {"adam_beta_1": np.exp(rng.uniform(-5, 0)),
                                 "adam_beta_2": np.exp(rng.uniform(-12, -6)),
                                 "adam_eps": np.exp(rng.uniform(-22, -15)),
                                 "adam_wd": np.exp(rng.uniform(-15, -9)),
                                 "max_lr": np.exp(rng.uniform(-6, -2)),
                                 "cycle_peak": rng.uniform(0.1, 0.5)
                                 },
        "binary_ops_all": {"adam_beta_1": np.exp(rng.uniform(-5, 0)),
                           "adam_beta_2": np.exp(rng.uniform(-12, -6)),
                           "adam_eps": np.exp(rng.uniform(-22, -15)),
                           "adam_wd": np.exp(rng.uniform(-15, -9)),
                           "max_lr": np.exp(rng.uniform(-6, -2)),
                           "cycle_peak": rng.uniform(0.1, 0.5)
                           },
        "lae": {"adam_beta_1": np.exp(rng.uniform(-5, 0)),
                "adam_beta_2": np.exp(rng.uniform(-12, -6)),
                "adam_eps": np.exp(rng.uniform(-22, -15)),
                "adam_wd": np.exp(rng.uniform(-15, -9)),
                "max_lr": np.exp(rng.uniform(-6, -2)),
                "cycle_peak": rng.uniform(0.1, 0.5)
                },
        "signed_geomean": {"adam_beta_1": np.exp(rng.uniform(-5, 0)),
                           "adam_beta_2": np.exp(rng.uniform(-12, -6)),
                           "adam_eps": np.exp(rng.uniform(-22, -15)),
                           "adam_wd": np.exp(rng.uniform(-15, -9)),
                           "max_lr": np.exp(rng.uniform(-6, -2)),
                           "cycle_peak": rng.uniform(0.1, 0.5)
                           },
        "linf": {"adam_beta_1": np.exp(rng.uniform(-5, 0)),
                 "adam_beta_2": np.exp(rng.uniform(-12, -6)),
                 "adam_eps": np.exp(rng.uniform(-22, -15)),
                 "adam_wd": np.exp(rng.uniform(-15, -9)),
                 "max_lr": np.exp(rng.uniform(-6, -2)),
                 "cycle_peak": rng.uniform(0.1, 0.5)
                 },
        "swishk": {"adam_beta_1": np.exp(rng.uniform(-5, 0)),
                   "adam_beta_2": np.exp(rng.uniform(-12, -6)),
                   "adam_eps": np.exp(rng.uniform(-22, -15)),
                   "adam_wd": np.exp(rng.uniform(-15, -9)),
                   "max_lr": np.exp(rng.uniform(-6, -2)),
                   "cycle_peak": rng.uniform(0.1, 0.5)
                   },
        "lse": {"adam_beta_1": np.exp(rng.uniform(-5, 0)),
                "adam_beta_2": np.exp(rng.uniform(-12, -6)),
                "adam_eps": np.exp(rng.uniform(-22, -15)),
                "adam_wd": np.exp(rng.uniform(-15, -9)),
                "max_lr": np.exp(rng.uniform(-6, -2)),
                "cycle_peak": rng.uniform(0.1, 0.5)
                },
        "min": {"adam_beta_1": np.exp(rng.uniform(-5, 0)),
                "adam_beta_2": np.exp(rng.uniform(-12, -6)),
                "adam_eps": np.exp(rng.uniform(-22, -15)),
                "adam_wd": np.exp(rng.uniform(-15, -9)),
                "max_lr": np.exp(rng.uniform(-6, -2)),
                "cycle_peak": rng.uniform(0.1, 0.5)
                }
    }


def nn_cifar10(rng):
    return {
        "relu": {"adam_beta_1": np.exp(rng.uniform(-5, 0)),
                 "adam_beta_2": np.exp(rng.uniform(-12, -6)),
                 "adam_eps": np.exp(rng.uniform(-22, -15)),
                 "adam_wd": np.exp(rng.uniform(-15, -9)),
                 "max_lr": np.exp(rng.uniform(-6, -2)),
                 "cycle_peak": rng.uniform(0.1, 0.5)
                 },
        "abs": {"adam_beta_1": np.exp(rng.uniform(-5, 0)),
                "adam_beta_2": np.exp(rng.uniform(-12, -6)),
                "adam_eps": np.exp(rng.uniform(-22, -15)),
                "adam_wd": np.exp(rng.uniform(-15, -9)),
                "max_lr": np.exp(rng.uniform(-6, -2)),
                "cycle_peak": rng.uniform(0.1, 0.5)
                },
        "combinact": {"adam_beta_1": np.exp(rng.uniform(-5, 0)),
                      "adam_beta_2": np.exp(rng.uniform(-12, -6)),
                      "adam_eps": np.exp(rng.uniform(-22, -15)),
                      "adam_wd": np.exp(rng.uniform(-15, -9)),
                      "max_lr": np.exp(rng.uniform(-6, -2)),
                      "cycle_peak": rng.uniform(0.1, 0.5)
                      },
        "max": {"adam_beta_1": np.exp(rng.uniform(-5, 0)),
                "adam_beta_2": np.exp(rng.uniform(-12, -6)),
                "adam_eps": np.exp(rng.uniform(-22, -15)),
                "adam_wd": np.exp(rng.uniform(-15, -9)),
                "max_lr": np.exp(rng.uniform(-6, -2)),
                "cycle_peak": rng.uniform(0.1, 0.5)
                },
        "min": {"adam_beta_1": np.exp(rng.uniform(-5, 0)),
                "adam_beta_2": np.exp(rng.uniform(-12, -6)),
                "adam_eps": np.exp(rng.uniform(-22, -15)),
                "adam_wd": np.exp(rng.uniform(-15, -9)),
                "max_lr": np.exp(rng.uniform(-6, -2)),
                "cycle_peak": rng.uniform(0.1, 0.5)
                },
        "l2": {"adam_beta_1": np.exp(rng.uniform(-5, 0)),
               "adam_beta_2": np.exp(rng.uniform(-12, -6)),
               "adam_eps": np.exp(rng.uniform(-22, -15)),
               "adam_wd": np.exp(rng.uniform(-15, -9)),
               "max_lr": np.exp(rng.uniform(-6, -2)),
               "cycle_peak": rng.uniform(0.1, 0.5)
               },
        "linf": {"adam_beta_1": np.exp(rng.uniform(-5, 0)),
                 "adam_beta_2": np.exp(rng.uniform(-12, -6)),
                 "adam_eps": np.exp(rng.uniform(-22, -15)),
                 "adam_wd": np.exp(rng.uniform(-15, -9)),
                 "max_lr": np.exp(rng.uniform(-6, -2)),
                 "cycle_peak": rng.uniform(0.1, 0.5)
                 },
        "lse": {"adam_beta_1": np.exp(rng.uniform(-5, 0)),
                "adam_beta_2": np.exp(rng.uniform(-12, -6)),
                "adam_eps": np.exp(rng.uniform(-22, -15)),
                "adam_wd": np.exp(rng.uniform(-15, -9)),
                "max_lr": np.exp(rng.uniform(-6, -2)),
                "cycle_peak": rng.uniform(0.1, 0.5)
                },
        "lae": {"adam_beta_1": np.exp(rng.uniform(-5, 0)),
                "adam_beta_2": np.exp(rng.uniform(-12, -6)),
                "adam_eps": np.exp(rng.uniform(-22, -15)),
                "adam_wd": np.exp(rng.uniform(-15, -9)),
                "max_lr": np.exp(rng.uniform(-6, -2)),
                "cycle_peak": rng.uniform(0.1, 0.5)
                },
        "prod": {"adam_beta_1": np.exp(rng.uniform(-5, 0)),
                 "adam_beta_2": np.exp(rng.uniform(-12, -6)),
                 "adam_eps": np.exp(rng.uniform(-22, -15)),
                 "adam_wd": np.exp(rng.uniform(-15, -9)),
                 "max_lr": np.exp(rng.uniform(-6, -2)),
                 "cycle_peak": rng.uniform(0.1, 0.5)
                 },
        "binary_ops_partition": {"adam_beta_1": np.exp(rng.uniform(-5, 0)),
                                 "adam_beta_2": np.exp(rng.uniform(-12, -6)),
                                 "adam_eps": np.exp(rng.uniform(-22, -15)),
                                 "adam_wd": np.exp(rng.uniform(-15, -9)),
                                 "max_lr": np.exp(rng.uniform(-6, -2)),
                                 "cycle_peak": rng.uniform(0.1, 0.5)
                                 },
        "binary_ops_all": {"adam_beta_1": np.exp(rng.uniform(-5, 0)),
                           "adam_beta_2": np.exp(rng.uniform(-12, -6)),
                           "adam_eps": np.exp(rng.uniform(-22, -15)),
                           "adam_wd": np.exp(rng.uniform(-15, -9)),
                           "max_lr": np.exp(rng.uniform(-6, -2)),
                           "cycle_peak": rng.uniform(0.1, 0.5)
                           },
        "signed_geomean": {"adam_beta_1": np.exp(rng.uniform(-5, 0)),
                           "adam_beta_2": np.exp(rng.uniform(-12, -6)),
                           "adam_eps": np.exp(rng.uniform(-22, -15)),
                           "adam_wd": np.exp(rng.uniform(-15, -9)),
                           "max_lr": np.exp(rng.uniform(-6, -2)),
                           "cycle_peak": rng.uniform(0.1, 0.5)
                           },
        "swishk": {"adam_beta_1": np.exp(rng.uniform(-5, 0)),
                   "adam_beta_2": np.exp(rng.uniform(-12, -6)),
                   "adam_eps": np.exp(rng.uniform(-22, -15)),
                   "adam_wd": np.exp(rng.uniform(-15, -9)),
                   "max_lr": np.exp(rng.uniform(-6, -2)),
                   "cycle_peak": rng.uniform(0.1, 0.5)
                   }
    }


def nn_cifar100(rng):
    return {
        "relu": {"adam_beta_1": np.exp(rng.uniform(-5, 0)),
                 "adam_beta_2": np.exp(rng.uniform(-12, -6)),
                 "adam_eps": np.exp(rng.uniform(-22, -15)),
                 "adam_wd": np.exp(rng.uniform(-15, -9)),
                 "max_lr": np.exp(rng.uniform(-6, -2)),
                 "cycle_peak": rng.uniform(0.1, 0.5)
                 },
        "abs": {"adam_beta_1": np.exp(rng.uniform(-5, 0)),
                "adam_beta_2": np.exp(rng.uniform(-12, -6)),
                "adam_eps": np.exp(rng.uniform(-22, -15)),
                "adam_wd": np.exp(rng.uniform(-15, -9)),
                "max_lr": np.exp(rng.uniform(-6, -2)),
                "cycle_peak": rng.uniform(0.1, 0.5)
                },
        "combinact": {"adam_beta_1": np.exp(rng.uniform(-5, 0)),
                      "adam_beta_2": np.exp(rng.uniform(-12, -6)),
                      "adam_eps": np.exp(rng.uniform(-22, -15)),
                      "adam_wd": np.exp(rng.uniform(-15, -9)),
                      "max_lr": np.exp(rng.uniform(-6, -2)),
                      "cycle_peak": rng.uniform(0.1, 0.5)
                      },
        "max": {"adam_beta_1": np.exp(rng.uniform(-5, 0)),
                "adam_beta_2": np.exp(rng.uniform(-12, -6)),
                "adam_eps": np.exp(rng.uniform(-22, -15)),
                "adam_wd": np.exp(rng.uniform(-15, -9)),
                "max_lr": np.exp(rng.uniform(-6, -2)),
                "cycle_peak": rng.uniform(0.1, 0.5)
                },
        "min": {"adam_beta_1": np.exp(rng.uniform(-5, 0)),
                "adam_beta_2": np.exp(rng.uniform(-12, -6)),
                "adam_eps": np.exp(rng.uniform(-22, -15)),
                "adam_wd": np.exp(rng.uniform(-15, -9)),
                "max_lr": np.exp(rng.uniform(-6, -2)),
                "cycle_peak": rng.uniform(0.1, 0.5)
                },
        "l2": {"adam_beta_1": np.exp(rng.uniform(-5, 0)),
               "adam_beta_2": np.exp(rng.uniform(-12, -6)),
               "adam_eps": np.exp(rng.uniform(-22, -15)),
               "adam_wd": np.exp(rng.uniform(-15, -9)),
               "max_lr": np.exp(rng.uniform(-6, -2)),
               "cycle_peak": rng.uniform(0.1, 0.5)
               },
        "linf": {"adam_beta_1": np.exp(rng.uniform(-5, 0)),
                 "adam_beta_2": np.exp(rng.uniform(-12, -6)),
                 "adam_eps": np.exp(rng.uniform(-22, -15)),
                 "adam_wd": np.exp(rng.uniform(-15, -9)),
                 "max_lr": np.exp(rng.uniform(-6, -2)),
                 "cycle_peak": rng.uniform(0.1, 0.5)
                 },
        "lse": {"adam_beta_1": np.exp(rng.uniform(-5, 0)),
                "adam_beta_2": np.exp(rng.uniform(-12, -6)),
                "adam_eps": np.exp(rng.uniform(-22, -15)),
                "adam_wd": np.exp(rng.uniform(-15, -9)),
                "max_lr": np.exp(rng.uniform(-6, -2)),
                "cycle_peak": rng.uniform(0.1, 0.5)
                },
        "lae": {"adam_beta_1": np.exp(rng.uniform(-5, 0)),
                "adam_beta_2": np.exp(rng.uniform(-12, -6)),
                "adam_eps": np.exp(rng.uniform(-22, -15)),
                "adam_wd": np.exp(rng.uniform(-15, -9)),
                "max_lr": np.exp(rng.uniform(-6, -2)),
                "cycle_peak": rng.uniform(0.1, 0.5)
                },
        "prod": {"adam_beta_1": np.exp(rng.uniform(-5, 0)),
                 "adam_beta_2": np.exp(rng.uniform(-12, -6)),
                 "adam_eps": np.exp(rng.uniform(-22, -15)),
                 "adam_wd": np.exp(rng.uniform(-15, -9)),
                 "max_lr": np.exp(rng.uniform(-6, -2)),
                 "cycle_peak": rng.uniform(0.1, 0.5)
                 },
        "binary_ops_partition": {"adam_beta_1": np.exp(rng.uniform(-5, 0)),
                                 "adam_beta_2": np.exp(rng.uniform(-12, -6)),
                                 "adam_eps": np.exp(rng.uniform(-22, -15)),
                                 "adam_wd": np.exp(rng.uniform(-15, -9)),
                                 "max_lr": np.exp(rng.uniform(-6, -2)),
                                 "cycle_peak": rng.uniform(0.1, 0.5)
                                 },
        "binary_ops_all": {"adam_beta_1": np.exp(rng.uniform(-5, 0)),
                           "adam_beta_2": np.exp(rng.uniform(-12, -6)),
                           "adam_eps": np.exp(rng.uniform(-22, -15)),
                           "adam_wd": np.exp(rng.uniform(-15, -9)),
                           "max_lr": np.exp(rng.uniform(-6, -2)),
                           "cycle_peak": rng.uniform(0.1, 0.5)
                           },
        "signed_geomean": {"adam_beta_1": np.exp(rng.uniform(-5, 0)),
                           "adam_beta_2": np.exp(rng.uniform(-12, -6)),
                           "adam_eps": np.exp(rng.uniform(-22, -15)),
                           "adam_wd": np.exp(rng.uniform(-15, -9)),
                           "max_lr": np.exp(rng.uniform(-6, -2)),
                           "cycle_peak": rng.uniform(0.1, 0.5)
                           },
        "swishk": {"adam_beta_1": np.exp(rng.uniform(-5, 0)),
                   "adam_beta_2": np.exp(rng.uniform(-12, -6)),
                   "adam_eps": np.exp(rng.uniform(-22, -15)),
                   "adam_wd": np.exp(rng.uniform(-15, -9)),
                   "max_lr": np.exp(rng.uniform(-6, -2)),
                   "cycle_peak": rng.uniform(0.1, 0.5)
                   }
    }


def nn_fashion_mnist(rng):
    return {
        "relu": {"adam_beta_1": np.exp(rng.uniform(-5, 0)),
                 "adam_beta_2": np.exp(rng.uniform(-12, -6)),
                 "adam_eps": np.exp(rng.uniform(-22, -15)),
                 "adam_wd": np.exp(rng.uniform(-15, -9)),
                 "max_lr": np.exp(rng.uniform(-6, -2)),
                 "cycle_peak": rng.uniform(0.1, 0.5)
                 },
        "abs": {"adam_beta_1": np.exp(rng.uniform(-5, 0)),
                "adam_beta_2": np.exp(rng.uniform(-12, -6)),
                "adam_eps": np.exp(rng.uniform(-22, -15)),
                "adam_wd": np.exp(rng.uniform(-15, -9)),
                "max_lr": np.exp(rng.uniform(-6, -2)),
                "cycle_peak": rng.uniform(0.1, 0.5)
                },
        "combinact": {"adam_beta_1": np.exp(rng.uniform(-5, 0)),
                      "adam_beta_2": np.exp(rng.uniform(-12, -6)),
                      "adam_eps": np.exp(rng.uniform(-22, -15)),
                      "adam_wd": np.exp(rng.uniform(-15, -9)),
                      "max_lr": np.exp(rng.uniform(-6, -2)),
                      "cycle_peak": rng.uniform(0.1, 0.5)
                      },
        "max": {"adam_beta_1": np.exp(rng.uniform(-5, 0)),
                "adam_beta_2": np.exp(rng.uniform(-12, -6)),
                "adam_eps": np.exp(rng.uniform(-22, -15)),
                "adam_wd": np.exp(rng.uniform(-15, -9)),
                "max_lr": np.exp(rng.uniform(-6, -2)),
                "cycle_peak": rng.uniform(0.1, 0.5)
                },
        "min": {"adam_beta_1": np.exp(rng.uniform(-5, 0)),
                "adam_beta_2": np.exp(rng.uniform(-12, -6)),
                "adam_eps": np.exp(rng.uniform(-22, -15)),
                "adam_wd": np.exp(rng.uniform(-15, -9)),
                "max_lr": np.exp(rng.uniform(-6, -2)),
                "cycle_peak": rng.uniform(0.1, 0.5)
                },
        "l2": {"adam_beta_1": np.exp(rng.uniform(-5, 0)),
               "adam_beta_2": np.exp(rng.uniform(-12, -6)),
               "adam_eps": np.exp(rng.uniform(-22, -15)),
               "adam_wd": np.exp(rng.uniform(-15, -9)),
               "max_lr": np.exp(rng.uniform(-6, -2)),
               "cycle_peak": rng.uniform(0.1, 0.5)
               },
        "linf": {"adam_beta_1": np.exp(rng.uniform(-5, 0)),
                 "adam_beta_2": np.exp(rng.uniform(-12, -6)),
                 "adam_eps": np.exp(rng.uniform(-22, -15)),
                 "adam_wd": np.exp(rng.uniform(-15, -9)),
                 "max_lr": np.exp(rng.uniform(-6, -2)),
                 "cycle_peak": rng.uniform(0.1, 0.5)
                 },
        "lse": {"adam_beta_1": np.exp(rng.uniform(-5, 0)),
                "adam_beta_2": np.exp(rng.uniform(-12, -6)),
                "adam_eps": np.exp(rng.uniform(-22, -15)),
                "adam_wd": np.exp(rng.uniform(-15, -9)),
                "max_lr": np.exp(rng.uniform(-6, -2)),
                "cycle_peak": rng.uniform(0.1, 0.5)
                },
        "lae": {"adam_beta_1": np.exp(rng.uniform(-5, 0)),
                "adam_beta_2": np.exp(rng.uniform(-12, -6)),
                "adam_eps": np.exp(rng.uniform(-22, -15)),
                "adam_wd": np.exp(rng.uniform(-15, -9)),
                "max_lr": np.exp(rng.uniform(-6, -2)),
                "cycle_peak": rng.uniform(0.1, 0.5)
                },
        "prod": {"adam_beta_1": np.exp(rng.uniform(-5, 0)),
                 "adam_beta_2": np.exp(rng.uniform(-12, -6)),
                 "adam_eps": np.exp(rng.uniform(-22, -15)),
                 "adam_wd": np.exp(rng.uniform(-15, -9)),
                 "max_lr": np.exp(rng.uniform(-6, -2)),
                 "cycle_peak": rng.uniform(0.1, 0.5)
                 },
        "binary_ops_partition": {"adam_beta_1": np.exp(rng.uniform(-5, 0)),
                                 "adam_beta_2": np.exp(rng.uniform(-12, -6)),
                                 "adam_eps": np.exp(rng.uniform(-22, -15)),
                                 "adam_wd": np.exp(rng.uniform(-15, -9)),
                                 "max_lr": np.exp(rng.uniform(-6, -2)),
                                 "cycle_peak": rng.uniform(0.1, 0.5)
                                 },
        "binary_ops_all": {"adam_beta_1": np.exp(rng.uniform(-5, 0)),
                           "adam_beta_2": np.exp(rng.uniform(-12, -6)),
                           "adam_eps": np.exp(rng.uniform(-22, -15)),
                           "adam_wd": np.exp(rng.uniform(-15, -9)),
                           "max_lr": np.exp(rng.uniform(-6, -2)),
                           "cycle_peak": rng.uniform(0.1, 0.5)
                           },
        "signed_geomean": {"adam_beta_1": np.exp(rng.uniform(-5, 0)),
                           "adam_beta_2": np.exp(rng.uniform(-12, -6)),
                           "adam_eps": np.exp(rng.uniform(-22, -15)),
                           "adam_wd": np.exp(rng.uniform(-15, -9)),
                           "max_lr": np.exp(rng.uniform(-6, -2)),
                           "cycle_peak": rng.uniform(0.1, 0.5)
                           },
        "swishk": {"adam_beta_1": np.exp(rng.uniform(-5, 0)),
                   "adam_beta_2": np.exp(rng.uniform(-12, -6)),
                   "adam_eps": np.exp(rng.uniform(-22, -15)),
                   "adam_wd": np.exp(rng.uniform(-15, -9)),
                   "max_lr": np.exp(rng.uniform(-6, -2)),
                   "cycle_peak": rng.uniform(0.1, 0.5)
                   }
    }


def nn_svhn(rng):
    return {
        "relu": {"adam_beta_1": np.exp(rng.uniform(-5, 0)),
                 "adam_beta_2": np.exp(rng.uniform(-12, -6)),
                 "adam_eps": np.exp(rng.uniform(-22, -15)),
                 "adam_wd": np.exp(rng.uniform(-15, -9)),
                 "max_lr": np.exp(rng.uniform(-6, -2)),
                 "cycle_peak": rng.uniform(0.1, 0.5)
                 },
        "abs": {"adam_beta_1": np.exp(rng.uniform(-5, 0)),
                "adam_beta_2": np.exp(rng.uniform(-12, -6)),
                "adam_eps": np.exp(rng.uniform(-22, -15)),
                "adam_wd": np.exp(rng.uniform(-15, -9)),
                "max_lr": np.exp(rng.uniform(-6, -2)),
                "cycle_peak": rng.uniform(0.1, 0.5)
                },
        "combinact": {"adam_beta_1": np.exp(rng.uniform(-5, 0)),
                      "adam_beta_2": np.exp(rng.uniform(-12, -6)),
                      "adam_eps": np.exp(rng.uniform(-22, -15)),
                      "adam_wd": np.exp(rng.uniform(-15, -9)),
                      "max_lr": np.exp(rng.uniform(-6, -2)),
                      "cycle_peak": rng.uniform(0.1, 0.5)
                      },
        "max": {"adam_beta_1": np.exp(rng.uniform(-5, 0)),
                "adam_beta_2": np.exp(rng.uniform(-12, -6)),
                "adam_eps": np.exp(rng.uniform(-22, -15)),
                "adam_wd": np.exp(rng.uniform(-15, -9)),
                "max_lr": np.exp(rng.uniform(-6, -2)),
                "cycle_peak": rng.uniform(0.1, 0.5)
                },
        "min": {"adam_beta_1": np.exp(rng.uniform(-5, 0)),
                "adam_beta_2": np.exp(rng.uniform(-12, -6)),
                "adam_eps": np.exp(rng.uniform(-22, -15)),
                "adam_wd": np.exp(rng.uniform(-15, -9)),
                "max_lr": np.exp(rng.uniform(-6, -2)),
                "cycle_peak": rng.uniform(0.1, 0.5)
                },
        "l2": {"adam_beta_1": np.exp(rng.uniform(-5, 0)),
               "adam_beta_2": np.exp(rng.uniform(-12, -6)),
               "adam_eps": np.exp(rng.uniform(-22, -15)),
               "adam_wd": np.exp(rng.uniform(-15, -9)),
               "max_lr": np.exp(rng.uniform(-6, -2)),
               "cycle_peak": rng.uniform(0.1, 0.5)
               },
        "linf": {"adam_beta_1": np.exp(rng.uniform(-5, 0)),
                 "adam_beta_2": np.exp(rng.uniform(-12, -6)),
                 "adam_eps": np.exp(rng.uniform(-22, -15)),
                 "adam_wd": np.exp(rng.uniform(-15, -9)),
                 "max_lr": np.exp(rng.uniform(-6, -2)),
                 "cycle_peak": rng.uniform(0.1, 0.5)
                 },
        "lse": {"adam_beta_1": np.exp(rng.uniform(-5, 0)),
                "adam_beta_2": np.exp(rng.uniform(-12, -6)),
                "adam_eps": np.exp(rng.uniform(-22, -15)),
                "adam_wd": np.exp(rng.uniform(-15, -9)),
                "max_lr": np.exp(rng.uniform(-6, -2)),
                "cycle_peak": rng.uniform(0.1, 0.5)
                },
        "lae": {"adam_beta_1": np.exp(rng.uniform(-5, 0)),
                "adam_beta_2": np.exp(rng.uniform(-12, -6)),
                "adam_eps": np.exp(rng.uniform(-22, -15)),
                "adam_wd": np.exp(rng.uniform(-15, -9)),
                "max_lr": np.exp(rng.uniform(-6, -2)),
                "cycle_peak": rng.uniform(0.1, 0.5)
                },
        "prod": {"adam_beta_1": np.exp(rng.uniform(-5, 0)),
                 "adam_beta_2": np.exp(rng.uniform(-12, -6)),
                 "adam_eps": np.exp(rng.uniform(-22, -15)),
                 "adam_wd": np.exp(rng.uniform(-15, -9)),
                 "max_lr": np.exp(rng.uniform(-6, -2)),
                 "cycle_peak": rng.uniform(0.1, 0.5)
                 },
        "binary_ops_partition": {"adam_beta_1": np.exp(rng.uniform(-5, 0)),
                                 "adam_beta_2": np.exp(rng.uniform(-12, -6)),
                                 "adam_eps": np.exp(rng.uniform(-22, -15)),
                                 "adam_wd": np.exp(rng.uniform(-15, -9)),
                                 "max_lr": np.exp(rng.uniform(-6, -2)),
                                 "cycle_peak": rng.uniform(0.1, 0.5)
                                 },
        "binary_ops_all": {"adam_beta_1": np.exp(rng.uniform(-5, 0)),
                           "adam_beta_2": np.exp(rng.uniform(-12, -6)),
                           "adam_eps": np.exp(rng.uniform(-22, -15)),
                           "adam_wd": np.exp(rng.uniform(-15, -9)),
                           "max_lr": np.exp(rng.uniform(-6, -2)),
                           "cycle_peak": rng.uniform(0.1, 0.5)
                           },
        "signed_geomean": {"adam_beta_1": np.exp(rng.uniform(-5, 0)),
                           "adam_beta_2": np.exp(rng.uniform(-12, -6)),
                           "adam_eps": np.exp(rng.uniform(-22, -15)),
                           "adam_wd": np.exp(rng.uniform(-15, -9)),
                           "max_lr": np.exp(rng.uniform(-6, -2)),
                           "cycle_peak": rng.uniform(0.1, 0.5)
                           },
        "swishk": {"adam_beta_1": np.exp(rng.uniform(-5, 0)),
                   "adam_beta_2": np.exp(rng.uniform(-12, -6)),
                   "adam_eps": np.exp(rng.uniform(-22, -15)),
                   "adam_wd": np.exp(rng.uniform(-15, -9)),
                   "max_lr": np.exp(rng.uniform(-6, -2)),
                   "cycle_peak": rng.uniform(0.1, 0.5)
                   }
    }


# Opt
def cnn_mnist(rng):
    return {
        "relu": {"adam_beta_1": np.exp(-2.388609721693893),
                 "adam_beta_2": np.exp(-5.390639433192069),
                 "adam_eps": np.exp(-16.796444850789),
                 "adam_wd": np.exp(-13.253336539963065),
                 "max_lr": np.exp(-7.422704622466667),
                 "cycle_peak": 0.44656736904941696
                 },
        "multi_relu": {"adam_beta_1": np.exp(-3.0574547429631496),
                       "adam_beta_2": np.exp(-4.247448624972093),
                       "adam_eps": np.exp(-19.92328795485814),
                       "adam_wd": np.exp(-12.591192480548075),
                       "max_lr": np.exp(-7.183540698498811),
                       "cycle_peak": 0.1847688059239794
                       },
        "combinact": {"adam_beta_1": np.exp(-2.114439459425412),
                      "adam_beta_2": np.exp(-5.354100834352032),
                      "adam_eps": np.exp(-20.485014830268515),
                      "adam_wd": np.exp(-13.947985139710585),
                      "max_lr": np.exp(-6.710677476450968),
                      "cycle_peak": 0.45592998059813267
                      },
        "l2": {"adam_beta_1": np.exp(-2.798405250416228),
               "adam_beta_2": np.exp(-4.852502321085275),
               "adam_eps": np.exp(-17.69422318951969),
               "adam_wd": np.exp(-13.28251690242802),
               "max_lr": np.exp(-6.61986865125839),
               "cycle_peak": 0.2908993031473296
               },
        "abs": {"adam_beta_1": np.exp(-2.3046899444891737),
                "adam_beta_2": np.exp(-4.081976310255959),
                "adam_eps": np.exp(-18.81496719551105),
                "adam_wd": np.exp(-12.889583720257814),
                "max_lr": np.exp(-6.294826015471599),
                "cycle_peak": 0.43927944846465605
                },
        "l2_lae": {"adam_beta_1": np.exp(-3.819899346335893),
                   "adam_beta_2": np.exp(-4.187621156111093),
                   "adam_eps": np.exp(-15.889624513084568),
                   "adam_wd": np.exp(-13.735330355123828),
                   "max_lr": np.exp(-6.796509116066224),
                   "cycle_peak": 0.30452694109961825
                   },
        "max": {"adam_beta_1": np.exp(-3.0681674156782175),
                "adam_beta_2": np.exp(-4.08491441310968),
                "adam_eps": np.exp(-16.879716225530277),
                "adam_wd": np.exp(-12.851561693962033),
                "max_lr": np.exp(-5.689213936356616),
                "cycle_peak": 0.3131204615398092
                },
        "prod": {"adam_beta_1": np.exp(rng.uniform(-5, 0)),
                 "adam_beta_2": np.exp(rng.uniform(-12, -6)),
                 "adam_eps": np.exp(rng.uniform(-22, -15)),
                 "adam_wd": np.exp(rng.uniform(-15, -9)),
                 "max_lr": np.exp(rng.uniform(-6, -2)),
                 "cycle_peak": rng.uniform(0.1, 0.5)
                 },
        "binary_ops_partition": {"adam_beta_1": np.exp(rng.uniform(-5, 0)),
                                 "adam_beta_2": np.exp(rng.uniform(-12, -6)),
                                 "adam_eps": np.exp(rng.uniform(-22, -15)),
                                 "adam_wd": np.exp(rng.uniform(-15, -9)),
                                 "max_lr": np.exp(rng.uniform(-6, -2)),
                                 "cycle_peak": rng.uniform(0.1, 0.5)
                                 },
        "binary_ops_all": {"adam_beta_1": np.exp(rng.uniform(-5, 0)),
                           "adam_beta_2": np.exp(rng.uniform(-12, -6)),
                           "adam_eps": np.exp(rng.uniform(-22, -15)),
                           "adam_wd": np.exp(rng.uniform(-15, -9)),
                           "max_lr": np.exp(rng.uniform(-6, -2)),
                           "cycle_peak": rng.uniform(0.1, 0.5)
                           },
        "lae": {"adam_beta_1": np.exp(rng.uniform(-5, 0)),
                "adam_beta_2": np.exp(rng.uniform(-12, -6)),
                "adam_eps": np.exp(rng.uniform(-22, -15)),
                "adam_wd": np.exp(rng.uniform(-15, -9)),
                "max_lr": np.exp(rng.uniform(-6, -2)),
                "cycle_peak": rng.uniform(0.1, 0.5)
                },
        "signed_geomean": {"adam_beta_1": np.exp(rng.uniform(-5, 0)),
                           "adam_beta_2": np.exp(rng.uniform(-12, -6)),
                           "adam_eps": np.exp(rng.uniform(-22, -15)),
                           "adam_wd": np.exp(rng.uniform(-15, -9)),
                           "max_lr": np.exp(rng.uniform(-6, -2)),
                           "cycle_peak": rng.uniform(0.1, 0.5)
                           },
        "linf": {"adam_beta_1": np.exp(rng.uniform(-5, 0)),
                 "adam_beta_2": np.exp(rng.uniform(-12, -6)),
                 "adam_eps": np.exp(rng.uniform(-22, -15)),
                 "adam_wd": np.exp(rng.uniform(-15, -9)),
                 "max_lr": np.exp(rng.uniform(-6, -2)),
                 "cycle_peak": rng.uniform(0.1, 0.5)
                 },
        "swishk": {"adam_beta_1": np.exp(rng.uniform(-5, 0)),
                   "adam_beta_2": np.exp(rng.uniform(-12, -6)),
                   "adam_eps": np.exp(rng.uniform(-22, -15)),
                   "adam_wd": np.exp(rng.uniform(-15, -9)),
                   "max_lr": np.exp(rng.uniform(-6, -2)),
                   "cycle_peak": rng.uniform(0.1, 0.5)
                   },
        "lse": {"adam_beta_1": np.exp(rng.uniform(-5, 0)),
                "adam_beta_2": np.exp(rng.uniform(-12, -6)),
                "adam_eps": np.exp(rng.uniform(-22, -15)),
                "adam_wd": np.exp(rng.uniform(-15, -9)),
                "max_lr": np.exp(rng.uniform(-6, -2)),
                "cycle_peak": rng.uniform(0.1, 0.5)
                },
        "min": {"adam_beta_1": np.exp(rng.uniform(-5, 0)),
                "adam_beta_2": np.exp(rng.uniform(-12, -6)),
                "adam_eps": np.exp(rng.uniform(-22, -15)),
                "adam_wd": np.exp(rng.uniform(-15, -9)),
                "max_lr": np.exp(rng.uniform(-6, -2)),
                "cycle_peak": rng.uniform(0.1, 0.5)
                }
    }


# Opt
def cnn_cifar10(rng):
    return {
        "relu": {"adam_beta_1": np.exp(-2.2190390756250093),
                 "adam_beta_2": np.exp(-7.366914026786562),
                 "adam_eps": np.exp(-18.10735819580839),
                 "adam_wd": np.exp(-12.13318302542048),
                 "max_lr": np.exp(-7.467215859430617),
                 "cycle_peak": 0.49156736259596534
                 },
        "multi_relu": {"adam_beta_1": np.exp(-2.3543718655934547),
                       "adam_beta_2": np.exp(-7.063659565937045),
                       "adam_eps": np.exp(-19.22492182089545),
                       "adam_wd": np.exp(-10.269718286116909),
                       "max_lr": np.exp(-6.8770611857136075),
                       "cycle_peak": 0.4253002234015769
                       },
        "combinact": {"adam_beta_1": np.exp(-2.016683834468364),
                      "adam_beta_2": np.exp(-7.820800773443709),
                      "adam_eps": np.exp(-18.01936303461807),
                      "adam_wd": np.exp(-14.443234599437305),
                      "max_lr": np.exp(-6.7810979033379875),
                      "cycle_peak": 0.5439417885046983
                      },
        "l2": {"adam_beta_1": np.exp(-1.5749000540594622),
               "adam_beta_2": np.exp(-3.6702433473885767),
               "adam_eps": np.exp(-17.788820155080888),
               "adam_wd": np.exp(-14.297423169143356),
               "max_lr": np.exp(-7.246379919517856),
               "cycle_peak": 0.4721781379107825
               },
        "abs": {"adam_beta_1": np.exp(-2.13818391159692),
                "adam_beta_2": np.exp(-5.343641237772224),
                "adam_eps": np.exp(-20.681560434670658),
                "adam_wd": np.exp(-13.95451768488049),
                "max_lr": np.exp(-6.719724510164729),
                "cycle_peak": 0.4296646156803744
                },
        "l2_lae": {"adam_beta_1": np.exp(-1.511652530521991),
                   "adam_beta_2": np.exp(-5.10036591613782),
                   "adam_eps": np.exp(-20.158860548398614),
                   "adam_wd": np.exp(-11.630968574087534),
                   "max_lr": np.exp(-6.992522933149952),
                   "cycle_peak": 0.41503241211381126
                   },
        "max": {"adam_beta_1": np.exp(-2.3151753028565794),
                "adam_beta_2": np.exp(-4.660984944761118),
                "adam_eps": np.exp(-19.231174065933367),
                "adam_wd": np.exp(-8.028370292260313),
                "max_lr": np.exp(-6.720521846837062),
                "cycle_peak": 0.4677382752348381
                },
        "prod": {"adam_beta_1": np.exp(rng.uniform(-5, 0)),
                 "adam_beta_2": np.exp(rng.uniform(-12, -6)),
                 "adam_eps": np.exp(rng.uniform(-22, -15)),
                 "adam_wd": np.exp(rng.uniform(-15, -9)),
                 "max_lr": np.exp(rng.uniform(-6, -2)),
                 "cycle_peak": rng.uniform(0.1, 0.5)
                 },
        "binary_ops_partition": {"adam_beta_1": np.exp(rng.uniform(-5, 0)),
                                 "adam_beta_2": np.exp(rng.uniform(-12, -6)),
                                 "adam_eps": np.exp(rng.uniform(-22, -15)),
                                 "adam_wd": np.exp(rng.uniform(-15, -9)),
                                 "max_lr": np.exp(rng.uniform(-6, -2)),
                                 "cycle_peak": rng.uniform(0.1, 0.5)
                                 },
        "binary_ops_all": {"adam_beta_1": np.exp(rng.uniform(-5, 0)),
                           "adam_beta_2": np.exp(rng.uniform(-12, -6)),
                           "adam_eps": np.exp(rng.uniform(-22, -15)),
                           "adam_wd": np.exp(rng.uniform(-15, -9)),
                           "max_lr": np.exp(rng.uniform(-6, -2)),
                           "cycle_peak": rng.uniform(0.1, 0.5)
                           },
        "lae": {"adam_beta_1": np.exp(rng.uniform(-5, 0)),
                "adam_beta_2": np.exp(rng.uniform(-12, -6)),
                "adam_eps": np.exp(rng.uniform(-22, -15)),
                "adam_wd": np.exp(rng.uniform(-15, -9)),
                "max_lr": np.exp(rng.uniform(-6, -2)),
                "cycle_peak": rng.uniform(0.1, 0.5)
                },
        "signed_geomean": {"adam_beta_1": np.exp(rng.uniform(-5, 0)),
                           "adam_beta_2": np.exp(rng.uniform(-12, -6)),
                           "adam_eps": np.exp(rng.uniform(-22, -15)),
                           "adam_wd": np.exp(rng.uniform(-15, -9)),
                           "max_lr": np.exp(rng.uniform(-6, -2)),
                           "cycle_peak": rng.uniform(0.1, 0.5)
                           },
        "linf": {"adam_beta_1": np.exp(rng.uniform(-5, 0)),
                 "adam_beta_2": np.exp(rng.uniform(-12, -6)),
                 "adam_eps": np.exp(rng.uniform(-22, -15)),
                 "adam_wd": np.exp(rng.uniform(-15, -9)),
                 "max_lr": np.exp(rng.uniform(-6, -2)),
                 "cycle_peak": rng.uniform(0.1, 0.5)
                 },
        "swishk": {"adam_beta_1": np.exp(rng.uniform(-5, 0)),
                   "adam_beta_2": np.exp(rng.uniform(-12, -6)),
                   "adam_eps": np.exp(rng.uniform(-22, -15)),
                   "adam_wd": np.exp(rng.uniform(-15, -9)),
                   "max_lr": np.exp(rng.uniform(-6, -2)),
                   "cycle_peak": rng.uniform(0.1, 0.5)
                   },
        "lse": {"adam_beta_1": np.exp(rng.uniform(-5, 0)),
                "adam_beta_2": np.exp(rng.uniform(-12, -6)),
                "adam_eps": np.exp(rng.uniform(-22, -15)),
                "adam_wd": np.exp(rng.uniform(-15, -9)),
                "max_lr": np.exp(rng.uniform(-6, -2)),
                "cycle_peak": rng.uniform(0.1, 0.5)
                },
        "min": {"adam_beta_1": np.exp(rng.uniform(-5, 0)),
                "adam_beta_2": np.exp(rng.uniform(-12, -6)),
                "adam_eps": np.exp(rng.uniform(-22, -15)),
                "adam_wd": np.exp(rng.uniform(-15, -9)),
                "max_lr": np.exp(rng.uniform(-6, -2)),
                "cycle_peak": rng.uniform(0.1, 0.5)
                }
    }


# Opt
def cnn_cifar100(rng):
    return {
        "relu": {"adam_beta_1": np.exp(-2.426797853932341),
                 "adam_beta_2": np.exp(-8.32944462067869),
                 "adam_eps": np.exp(-19.791364579217475),
                 "adam_wd": np.exp(-11.772470713882507),
                 "max_lr": np.exp(-7.267477071254713),
                 "cycle_peak": 0.4517569495205036
                 },
        "multi_relu": {"adam_beta_1": np.exp(-1.9834923464612166),
                       "adam_beta_2": np.exp(-5.558187320969269),
                       "adam_eps": np.exp(-20.847973902813152),
                       "adam_wd": np.exp(-11.72610495446386),
                       "max_lr": np.exp(-7.177889555045069),
                       "cycle_peak": 0.36399876953225746
                       },
        "combinact": {"adam_beta_1": np.exp(-1.90329956854611),
                      "adam_beta_2": np.exp(-7.286469824634688),
                      "adam_eps": np.exp(-19.72445941906418),
                      "adam_wd": np.exp(-10.818233549453154),
                      "max_lr": np.exp(-6.861297361914738),
                      "cycle_peak": 0.5659379474281131
                      },
        "l2": {"adam_beta_1": np.exp(-1.8662979339075287),
               "adam_beta_2": np.exp(-4.588487764348233),
               "adam_eps": np.exp(-16.502805171408045),
               "adam_wd": np.exp(-10.825735290142346),
               "max_lr": np.exp(-7.23765587612481),
               "cycle_peak": 0.4491990506275296
               },
        "abs": {"adam_beta_1": np.exp(-2.4807814370731145),
                "adam_beta_2": np.exp(-7.619981735822295),
                "adam_eps": np.exp(-18.931867256862652),
                "adam_wd": np.exp(-11.812467208793322),
                "max_lr": np.exp(-7.195658475528992),
                "cycle_peak": 0.44128128173478703
                },
        "l2_lae": {"adam_beta_1": np.exp(-1.8490078448379565),
                   "adam_beta_2": np.exp(-5.281849941263149),
                   "adam_eps": np.exp(-17.490838571636928),
                   "adam_wd": np.exp(-11.897886419488167),
                   "max_lr": np.exp(-7.020510360640074),
                   "cycle_peak": 0.43013832410896696
                   },
        "max": {"adam_beta_1": np.exp(-2.7994828802749576),
                "adam_beta_2": np.exp(-5.369720801695676),
                "adam_eps": np.exp(-17.285952081693686),
                "adam_wd": np.exp(-8.600017225696668),
                "max_lr": np.exp(-6.220972625072611),
                "cycle_peak": 0.41433149467346075
                },
        "prod": {"adam_beta_1": np.exp(rng.uniform(-5, 0)),
                 "adam_beta_2": np.exp(rng.uniform(-12, -6)),
                 "adam_eps": np.exp(rng.uniform(-22, -15)),
                 "adam_wd": np.exp(rng.uniform(-15, -9)),
                 "max_lr": np.exp(rng.uniform(-6, -2)),
                 "cycle_peak": rng.uniform(0.1, 0.5)
                 },
        "binary_ops_partition": {"adam_beta_1": np.exp(rng.uniform(-5, 0)),
                                 "adam_beta_2": np.exp(rng.uniform(-12, -6)),
                                 "adam_eps": np.exp(rng.uniform(-22, -15)),
                                 "adam_wd": np.exp(rng.uniform(-15, -9)),
                                 "max_lr": np.exp(rng.uniform(-6, -2)),
                                 "cycle_peak": rng.uniform(0.1, 0.5)
                                 },
        "binary_ops_all": {"adam_beta_1": np.exp(rng.uniform(-5, 0)),
                           "adam_beta_2": np.exp(rng.uniform(-12, -6)),
                           "adam_eps": np.exp(rng.uniform(-22, -15)),
                           "adam_wd": np.exp(rng.uniform(-15, -9)),
                           "max_lr": np.exp(rng.uniform(-6, -2)),
                           "cycle_peak": rng.uniform(0.1, 0.5)
                           },
        "lae": {"adam_beta_1": np.exp(rng.uniform(-5, 0)),
                "adam_beta_2": np.exp(rng.uniform(-12, -6)),
                "adam_eps": np.exp(rng.uniform(-22, -15)),
                "adam_wd": np.exp(rng.uniform(-15, -9)),
                "max_lr": np.exp(rng.uniform(-6, -2)),
                "cycle_peak": rng.uniform(0.1, 0.5)
                },
        "signed_geomean": {"adam_beta_1": np.exp(rng.uniform(-5, 0)),
                           "adam_beta_2": np.exp(rng.uniform(-12, -6)),
                           "adam_eps": np.exp(rng.uniform(-22, -15)),
                           "adam_wd": np.exp(rng.uniform(-15, -9)),
                           "max_lr": np.exp(rng.uniform(-6, -2)),
                           "cycle_peak": rng.uniform(0.1, 0.5)
                           },
        "linf": {"adam_beta_1": np.exp(rng.uniform(-5, 0)),
                 "adam_beta_2": np.exp(rng.uniform(-12, -6)),
                 "adam_eps": np.exp(rng.uniform(-22, -15)),
                 "adam_wd": np.exp(rng.uniform(-15, -9)),
                 "max_lr": np.exp(rng.uniform(-6, -2)),
                 "cycle_peak": rng.uniform(0.1, 0.5)
                 },
        "swishk": {"adam_beta_1": np.exp(rng.uniform(-5, 0)),
                   "adam_beta_2": np.exp(rng.uniform(-12, -6)),
                   "adam_eps": np.exp(rng.uniform(-22, -15)),
                   "adam_wd": np.exp(rng.uniform(-15, -9)),
                   "max_lr": np.exp(rng.uniform(-6, -2)),
                   "cycle_peak": rng.uniform(0.1, 0.5)
                   },
        "lse": {"adam_beta_1": np.exp(rng.uniform(-5, 0)),
                "adam_beta_2": np.exp(rng.uniform(-12, -6)),
                "adam_eps": np.exp(rng.uniform(-22, -15)),
                "adam_wd": np.exp(rng.uniform(-15, -9)),
                "max_lr": np.exp(rng.uniform(-6, -2)),
                "cycle_peak": rng.uniform(0.1, 0.5)
                },
        "min": {"adam_beta_1": np.exp(rng.uniform(-5, 0)),
                "adam_beta_2": np.exp(rng.uniform(-12, -6)),
                "adam_eps": np.exp(rng.uniform(-22, -15)),
                "adam_wd": np.exp(rng.uniform(-15, -9)),
                "max_lr": np.exp(rng.uniform(-6, -2)),
                "cycle_peak": rng.uniform(0.1, 0.5)
                },
    }


def cnn_fashion_mnist(rng):
    return {
        "relu": {"adam_beta_1": np.exp(rng.uniform(-5, 0)),
                 "adam_beta_2": np.exp(rng.uniform(-12, -6)),
                 "adam_eps": np.exp(rng.uniform(-22, -15)),
                 "adam_wd": np.exp(rng.uniform(-15, -9)),
                 "max_lr": np.exp(rng.uniform(-6, -2)),
                 "cycle_peak": rng.uniform(0.1, 0.5)
                 },
        "abs": {"adam_beta_1": np.exp(rng.uniform(-5, 0)),
                "adam_beta_2": np.exp(rng.uniform(-12, -6)),
                "adam_eps": np.exp(rng.uniform(-22, -15)),
                "adam_wd": np.exp(rng.uniform(-15, -9)),
                "max_lr": np.exp(rng.uniform(-6, -2)),
                "cycle_peak": rng.uniform(0.1, 0.5)
                },
        "combinact": {"adam_beta_1": np.exp(rng.uniform(-5, 0)),
                      "adam_beta_2": np.exp(rng.uniform(-12, -6)),
                      "adam_eps": np.exp(rng.uniform(-22, -15)),
                      "adam_wd": np.exp(rng.uniform(-15, -9)),
                      "max_lr": np.exp(rng.uniform(-6, -2)),
                      "cycle_peak": rng.uniform(0.1, 0.5)
                      },
        "max": {"adam_beta_1": np.exp(rng.uniform(-5, 0)),
                "adam_beta_2": np.exp(rng.uniform(-12, -6)),
                "adam_eps": np.exp(rng.uniform(-22, -15)),
                "adam_wd": np.exp(rng.uniform(-15, -9)),
                "max_lr": np.exp(rng.uniform(-6, -2)),
                "cycle_peak": rng.uniform(0.1, 0.5)
                },
        "min": {"adam_beta_1": np.exp(rng.uniform(-5, 0)),
                "adam_beta_2": np.exp(rng.uniform(-12, -6)),
                "adam_eps": np.exp(rng.uniform(-22, -15)),
                "adam_wd": np.exp(rng.uniform(-15, -9)),
                "max_lr": np.exp(rng.uniform(-6, -2)),
                "cycle_peak": rng.uniform(0.1, 0.5)
                },
        "l2": {"adam_beta_1": np.exp(rng.uniform(-5, 0)),
               "adam_beta_2": np.exp(rng.uniform(-12, -6)),
               "adam_eps": np.exp(rng.uniform(-22, -15)),
               "adam_wd": np.exp(rng.uniform(-15, -9)),
               "max_lr": np.exp(rng.uniform(-6, -2)),
               "cycle_peak": rng.uniform(0.1, 0.5)
               },
        "linf": {"adam_beta_1": np.exp(rng.uniform(-5, 0)),
                 "adam_beta_2": np.exp(rng.uniform(-12, -6)),
                 "adam_eps": np.exp(rng.uniform(-22, -15)),
                 "adam_wd": np.exp(rng.uniform(-15, -9)),
                 "max_lr": np.exp(rng.uniform(-6, -2)),
                 "cycle_peak": rng.uniform(0.1, 0.5)
                 },
        "lse": {"adam_beta_1": np.exp(rng.uniform(-5, 0)),
                "adam_beta_2": np.exp(rng.uniform(-12, -6)),
                "adam_eps": np.exp(rng.uniform(-22, -15)),
                "adam_wd": np.exp(rng.uniform(-15, -9)),
                "max_lr": np.exp(rng.uniform(-6, -2)),
                "cycle_peak": rng.uniform(0.1, 0.5)
                },
        "lae": {"adam_beta_1": np.exp(rng.uniform(-5, 0)),
                "adam_beta_2": np.exp(rng.uniform(-12, -6)),
                "adam_eps": np.exp(rng.uniform(-22, -15)),
                "adam_wd": np.exp(rng.uniform(-15, -9)),
                "max_lr": np.exp(rng.uniform(-6, -2)),
                "cycle_peak": rng.uniform(0.1, 0.5)
                },
        "prod": {"adam_beta_1": np.exp(rng.uniform(-5, 0)),
                 "adam_beta_2": np.exp(rng.uniform(-12, -6)),
                 "adam_eps": np.exp(rng.uniform(-22, -15)),
                 "adam_wd": np.exp(rng.uniform(-15, -9)),
                 "max_lr": np.exp(rng.uniform(-6, -2)),
                 "cycle_peak": rng.uniform(0.1, 0.5)
                 },
        "binary_ops_partition": {"adam_beta_1": np.exp(rng.uniform(-5, 0)),
                                 "adam_beta_2": np.exp(rng.uniform(-12, -6)),
                                 "adam_eps": np.exp(rng.uniform(-22, -15)),
                                 "adam_wd": np.exp(rng.uniform(-15, -9)),
                                 "max_lr": np.exp(rng.uniform(-6, -2)),
                                 "cycle_peak": rng.uniform(0.1, 0.5)
                                 },
        "binary_ops_all": {"adam_beta_1": np.exp(rng.uniform(-5, 0)),
                           "adam_beta_2": np.exp(rng.uniform(-12, -6)),
                           "adam_eps": np.exp(rng.uniform(-22, -15)),
                           "adam_wd": np.exp(rng.uniform(-15, -9)),
                           "max_lr": np.exp(rng.uniform(-6, -2)),
                           "cycle_peak": rng.uniform(0.1, 0.5)
                           },
        "signed_geomean": {"adam_beta_1": np.exp(rng.uniform(-5, 0)),
                           "adam_beta_2": np.exp(rng.uniform(-12, -6)),
                           "adam_eps": np.exp(rng.uniform(-22, -15)),
                           "adam_wd": np.exp(rng.uniform(-15, -9)),
                           "max_lr": np.exp(rng.uniform(-6, -2)),
                           "cycle_peak": rng.uniform(0.1, 0.5)
                           },
        "swishk": {"adam_beta_1": np.exp(rng.uniform(-5, 0)),
                   "adam_beta_2": np.exp(rng.uniform(-12, -6)),
                   "adam_eps": np.exp(rng.uniform(-22, -15)),
                   "adam_wd": np.exp(rng.uniform(-15, -9)),
                   "max_lr": np.exp(rng.uniform(-6, -2)),
                   "cycle_peak": rng.uniform(0.1, 0.5)
                   }
    }


def cnn_svhn(rng):
    return {
        "relu": {"adam_beta_1": np.exp(rng.uniform(-5, 0)),
                 "adam_beta_2": np.exp(rng.uniform(-12, -6)),
                 "adam_eps": np.exp(rng.uniform(-22, -15)),
                 "adam_wd": np.exp(rng.uniform(-15, -9)),
                 "max_lr": np.exp(rng.uniform(-6, -2)),
                 "cycle_peak": rng.uniform(0.1, 0.5)
                 },
        "abs": {"adam_beta_1": np.exp(rng.uniform(-5, 0)),
                "adam_beta_2": np.exp(rng.uniform(-12, -6)),
                "adam_eps": np.exp(rng.uniform(-22, -15)),
                "adam_wd": np.exp(rng.uniform(-15, -9)),
                "max_lr": np.exp(rng.uniform(-6, -2)),
                "cycle_peak": rng.uniform(0.1, 0.5)
                },
        "combinact": {"adam_beta_1": np.exp(rng.uniform(-5, 0)),
                      "adam_beta_2": np.exp(rng.uniform(-12, -6)),
                      "adam_eps": np.exp(rng.uniform(-22, -15)),
                      "adam_wd": np.exp(rng.uniform(-15, -9)),
                      "max_lr": np.exp(rng.uniform(-6, -2)),
                      "cycle_peak": rng.uniform(0.1, 0.5)
                      },
        "max": {"adam_beta_1": np.exp(rng.uniform(-5, 0)),
                "adam_beta_2": np.exp(rng.uniform(-12, -6)),
                "adam_eps": np.exp(rng.uniform(-22, -15)),
                "adam_wd": np.exp(rng.uniform(-15, -9)),
                "max_lr": np.exp(rng.uniform(-6, -2)),
                "cycle_peak": rng.uniform(0.1, 0.5)
                },
        "min": {"adam_beta_1": np.exp(rng.uniform(-5, 0)),
                "adam_beta_2": np.exp(rng.uniform(-12, -6)),
                "adam_eps": np.exp(rng.uniform(-22, -15)),
                "adam_wd": np.exp(rng.uniform(-15, -9)),
                "max_lr": np.exp(rng.uniform(-6, -2)),
                "cycle_peak": rng.uniform(0.1, 0.5)
                },
        "l2": {"adam_beta_1": np.exp(rng.uniform(-5, 0)),
               "adam_beta_2": np.exp(rng.uniform(-12, -6)),
               "adam_eps": np.exp(rng.uniform(-22, -15)),
               "adam_wd": np.exp(rng.uniform(-15, -9)),
               "max_lr": np.exp(rng.uniform(-6, -2)),
               "cycle_peak": rng.uniform(0.1, 0.5)
               },
        "linf": {"adam_beta_1": np.exp(rng.uniform(-5, 0)),
                 "adam_beta_2": np.exp(rng.uniform(-12, -6)),
                 "adam_eps": np.exp(rng.uniform(-22, -15)),
                 "adam_wd": np.exp(rng.uniform(-15, -9)),
                 "max_lr": np.exp(rng.uniform(-6, -2)),
                 "cycle_peak": rng.uniform(0.1, 0.5)
                 },
        "lse": {"adam_beta_1": np.exp(rng.uniform(-5, 0)),
                "adam_beta_2": np.exp(rng.uniform(-12, -6)),
                "adam_eps": np.exp(rng.uniform(-22, -15)),
                "adam_wd": np.exp(rng.uniform(-15, -9)),
                "max_lr": np.exp(rng.uniform(-6, -2)),
                "cycle_peak": rng.uniform(0.1, 0.5)
                },
        "lae": {"adam_beta_1": np.exp(rng.uniform(-5, 0)),
                "adam_beta_2": np.exp(rng.uniform(-12, -6)),
                "adam_eps": np.exp(rng.uniform(-22, -15)),
                "adam_wd": np.exp(rng.uniform(-15, -9)),
                "max_lr": np.exp(rng.uniform(-6, -2)),
                "cycle_peak": rng.uniform(0.1, 0.5)
                },
        "prod": {"adam_beta_1": np.exp(rng.uniform(-5, 0)),
                 "adam_beta_2": np.exp(rng.uniform(-12, -6)),
                 "adam_eps": np.exp(rng.uniform(-22, -15)),
                 "adam_wd": np.exp(rng.uniform(-15, -9)),
                 "max_lr": np.exp(rng.uniform(-6, -2)),
                 "cycle_peak": rng.uniform(0.1, 0.5)
                 },
        "binary_ops_partition": {"adam_beta_1": np.exp(rng.uniform(-5, 0)),
                                 "adam_beta_2": np.exp(rng.uniform(-12, -6)),
                                 "adam_eps": np.exp(rng.uniform(-22, -15)),
                                 "adam_wd": np.exp(rng.uniform(-15, -9)),
                                 "max_lr": np.exp(rng.uniform(-6, -2)),
                                 "cycle_peak": rng.uniform(0.1, 0.5)
                                 },
        "binary_ops_all": {"adam_beta_1": np.exp(rng.uniform(-5, 0)),
                           "adam_beta_2": np.exp(rng.uniform(-12, -6)),
                           "adam_eps": np.exp(rng.uniform(-22, -15)),
                           "adam_wd": np.exp(rng.uniform(-15, -9)),
                           "max_lr": np.exp(rng.uniform(-6, -2)),
                           "cycle_peak": rng.uniform(0.1, 0.5)
                           },
        "signed_geomean": {"adam_beta_1": np.exp(rng.uniform(-5, 0)),
                           "adam_beta_2": np.exp(rng.uniform(-12, -6)),
                           "adam_eps": np.exp(rng.uniform(-22, -15)),
                           "adam_wd": np.exp(rng.uniform(-15, -9)),
                           "max_lr": np.exp(rng.uniform(-6, -2)),
                           "cycle_peak": rng.uniform(0.1, 0.5)
                           },
        "swishk": {"adam_beta_1": np.exp(rng.uniform(-5, 0)),
                   "adam_beta_2": np.exp(rng.uniform(-12, -6)),
                   "adam_eps": np.exp(rng.uniform(-22, -15)),
                   "adam_wd": np.exp(rng.uniform(-15, -9)),
                   "max_lr": np.exp(rng.uniform(-6, -2)),
                   "cycle_peak": rng.uniform(0.1, 0.5)
                   }
    }
