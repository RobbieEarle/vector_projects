import numpy as np


def get_hyper_params(model, dataset, actfun, rng=None):

    if model == 'nn' and dataset == 'mnist':
        return nn_mnist(rng)[actfun]
    elif model == 'cnn':
        if dataset == 'cifar10':
            return cnn_cifar10(rng)[actfun]
        if dataset == 'cifar100':
            return cnn_cifar100(rng)[actfun]
        if dataset == 'mnist':
            return cnn_mnist(rng)[actfun]
    else:
        raise ValueError("Error: No hyper-parameters found for {}, {}".format(model, dataset))


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
                }
    }


def cnn_cifar10(rng):
    return {
        "relu": {"adam_beta_1": np.exp(rng.uniform(-2.24, -2.2)),
                 "adam_beta_2": np.exp(rng.uniform(-7.4, -7.11)),
                 "adam_eps": np.exp(rng.uniform(-18.25, -18)),
                 "adam_wd": np.exp(rng.uniform(-12.19, -12.05)),
                 "max_lr": np.exp(rng.uniform(-7.5, -7.32)),
                 "cycle_peak": rng.uniform(0.49, 0.5)
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
        "abs": {"adam_beta_1": np.exp(rng.uniform(-2.14, -2.1)),
                "adam_beta_2": np.exp(rng.uniform(-5.45, -5.31)),
                "adam_eps": np.exp(rng.uniform(-20.7, -20.49)),
                "adam_wd": np.exp(rng.uniform(-13.97, -13.86)),
                "max_lr": np.exp(rng.uniform(-6.74, -6.71)),
                "cycle_peak": rng.uniform(0.42, 0.43)
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
                }
    }


def cnn_cifar100(rng):
    return {
        "relu": {"adam_beta_1": np.exp(rng.uniform(-2.43, -2.4)),
                 "adam_beta_2": np.exp(rng.uniform(-8.48, -8.26)),
                 "adam_eps": np.exp(rng.uniform(-20, -19.61)),
                 "adam_wd": np.exp(rng.uniform(-11.85, -11.64)),
                 "max_lr": np.exp(rng.uniform(-7.35, -7.15)),
                 "cycle_peak": rng.uniform(0.44, 0.46)
                 },
        "multi_relu": {"adam_beta_1": np.exp(rng.uniform(-2.04, -1.97)),
                       "adam_beta_2": np.exp(rng.uniform(-5.58, -5.35)),
                       "adam_eps": np.exp(rng.uniform(-21.03, -20.77)),
                       "adam_wd": np.exp(rng.uniform(-11.95, -11.71)),
                       "max_lr": np.exp(rng.uniform(-7.22, -7.17)),
                       "cycle_peak": rng.uniform(0.34, 0.37)
                       },
        "combinact": {"adam_beta_1": np.exp(rng.uniform(-2, -1.9)),
                      "adam_beta_2": np.exp(rng.uniform(-7.34, -7.16)),
                      "adam_eps": np.exp(rng.uniform(-19.92, -19.62)),
                      "adam_wd": np.exp(rng.uniform(-10.92, -10.73)),
                      "max_lr": np.exp(rng.uniform(-6.95, -6.85)),
                      "cycle_peak": rng.uniform(0.54, 0.57)
                      },
        "l2": {"adam_beta_1": np.exp(rng.uniform(-1.92, -1.8)),
               "adam_beta_2": np.exp(rng.uniform(-4.83, -4.38)),
               "adam_eps": np.exp(rng.uniform(-16.64, -16.33)),
               "adam_wd": np.exp(rng.uniform(-10.94, -10.81)),
               "max_lr": np.exp(rng.uniform(-7.25, -7.09)),
               "cycle_peak": rng.uniform(0.4, 0.48)
               },
        "abs": {"adam_beta_1": np.exp(rng.uniform(-2.53, -2.45)),
                "adam_beta_2": np.exp(rng.uniform(-7.71, -7.57)),
                "adam_eps": np.exp(rng.uniform(-19.05, -18.61)),
                "adam_wd": np.exp(rng.uniform(-11.97, -11.76)),
                "max_lr": np.exp(rng.uniform(-7.21, -7.16)),
                "cycle_peak": rng.uniform(0.42, 0.45)
                },
        "l2_lae": {"adam_beta_1": np.exp(rng.uniform(-1.9, -1.82)),
                   "adam_beta_2": np.exp(rng.uniform(-5.63, -5.26)),
                   "adam_eps": np.exp(rng.uniform(-17.7, -16.92)),
                   "adam_wd": np.exp(rng.uniform(-11.92, -11.66)),
                   "max_lr": np.exp(rng.uniform(-7.16, -7.02)),
                   "cycle_peak": rng.uniform(0.43, 0.45)
                   },
        "max": {"adam_beta_1": np.exp(rng.uniform(-2.83, -2.73)),
                "adam_beta_2": np.exp(rng.uniform(-5.6, -5.3)),
                "adam_eps": np.exp(rng.uniform(-17.3, -17.12)),
                "adam_wd": np.exp(rng.uniform(-8.65, -8.55)),
                "max_lr": np.exp(rng.uniform(-6.3, -6.11)),
                "cycle_peak": rng.uniform(0.4, 0.42)
                }
    }


def cnn_mnist(rng):
    return {
        "relu": {"adam_beta_1": np.exp(rng.uniform(-2.48, -2.3)),
                 "adam_beta_2": np.exp(rng.uniform(-5.46, -5.35)),
                 "adam_eps": np.exp(rng.uniform(-17.53, -16.46)),
                 "adam_wd": np.exp(rng.uniform(-13.48, -13.25)),
                 "max_lr": np.exp(rng.uniform(-7.7, -7.38)),
                 "cycle_peak": rng.uniform(0.43, 0.45)
                 },
        "multi_relu": {"adam_beta_1": np.exp(rng.uniform(-3.18, -3.02)),
                       "adam_beta_2": np.exp(rng.uniform(-4.46, -4.05)),
                       "adam_eps": np.exp(rng.uniform(-19.95, -19.86)),
                       "adam_wd": np.exp(rng.uniform(-12.64, -12.38)),
                       "max_lr": np.exp(rng.uniform(-7.19, -7.11)),
                       "cycle_peak": rng.uniform(0.17, 0.19)
                       },
        "combinact": {"adam_beta_1": np.exp(rng.uniform(-2.16, -1.88)),
                      "adam_beta_2": np.exp(rng.uniform(-5.48, -5.25)),
                      "adam_eps": np.exp(rng.uniform(-20.68, -20.43)),
                      "adam_wd": np.exp(rng.uniform(-14.05, -13.71)),
                      "max_lr": np.exp(rng.uniform(-6.9, -6.61)),
                      "cycle_peak": rng.uniform(0.45, 0.47)
                      },
        "l2": {"adam_beta_1": np.exp(rng.uniform(-2.81, -2.74)),
               "adam_beta_2": np.exp(rng.uniform(-4.93, -4.82)),
               "adam_eps": np.exp(rng.uniform(-18.02, -17.5)),
               "adam_wd": np.exp(rng.uniform(-13.48, -13.26)),
               "max_lr": np.exp(rng.uniform(-6.71, -6.61)),
               "cycle_peak": rng.uniform(0.28, 0.3)
               },
        "abs": {"adam_beta_1": np.exp(rng.uniform(-2.35, -2.22)),
                "adam_beta_2": np.exp(rng.uniform(-4.15, -3.95)),
                "adam_eps": np.exp(rng.uniform(-19.05, -18.79)),
                "adam_wd": np.exp(rng.uniform(-13.22, -12.86)),
                "max_lr": np.exp(rng.uniform(-6.32, -6.06)),
                "cycle_peak": rng.uniform(0.43, 0.44)
                },
        "l2_lae": {"adam_beta_1": np.exp(rng.uniform(-3.9, -3.76)),
                   "adam_beta_2": np.exp(rng.uniform(-4.33, -4.11)),
                   "adam_eps": np.exp(rng.uniform(-15.94, -15.65)),
                   "adam_wd": np.exp(rng.uniform(-13.84, -13.61)),
                   "max_lr": np.exp(rng.uniform(-6.9, -6.78)),
                   "cycle_peak": rng.uniform(0.3, 0.31)
                   },
        "max": {"adam_beta_1": np.exp(rng.uniform(-3.16, -3.04)),
                "adam_beta_2": np.exp(rng.uniform(-4.21, -4.05)),
                "adam_eps": np.exp(rng.uniform(-17.17, -16.74)),
                "adam_wd": np.exp(rng.uniform(-12.87, -12.73)),
                "max_lr": np.exp(rng.uniform(-5.69, -5.67)),
                "cycle_peak": rng.uniform(0.3, 0.32)
                }
    }
