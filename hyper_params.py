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
        "relu": {"adam_beta_1": np.exp(rng.uniform(-2.35, -2.19)),
                 "adam_beta_2": np.exp(rng.uniform(-7.55, -6.45)),
                 "adam_eps": np.exp(rng.uniform(-18.8, -18)),
                 "adam_wd": np.exp(rng.uniform(-12.19, -11.9)),
                 "max_lr": np.exp(rng.uniform(-7.75, -7.27)),
                 "cycle_peak": rng.uniform(0.48, 0.5)
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
        "abs": {"adam_beta_1": np.exp(rng.uniform(-2.31, -2.1)),
                "adam_beta_2": np.exp(rng.uniform(-5.75, -5.2)),
                "adam_eps": np.exp(rng.uniform(-20.84, -20.1)),
                "adam_wd": np.exp(rng.uniform(-14.1, -13.63)),
                "max_lr": np.exp(rng.uniform(-6.75, -6.68)),
                "cycle_peak": rng.uniform(0.42, 0.44)
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
        "relu": {"adam_beta_1": np.exp(rng.uniform(-2.65, -2.4)),
                 "adam_beta_2": np.exp(rng.uniform(-8.85, -7.66)),
                 "adam_eps": np.exp(rng.uniform(-20, -18)),
                 "adam_wd": np.exp(rng.uniform(-12.45, -11)),
                 "max_lr": np.exp(rng.uniform(-7.5, -6.8)),
                 "cycle_peak": rng.uniform(0.4, 0.5)
                 },
        "multi_relu": {"adam_beta_1": np.exp(rng.uniform(-2.21, -1.8)),
                       "adam_beta_2": np.exp(rng.uniform(-6.65, -5.25)),
                       "adam_eps": np.exp(rng.uniform(-21.5, -20)),
                       "adam_wd": np.exp(rng.uniform(-12.2, -11.4)),
                       "max_lr": np.exp(rng.uniform(-7.25, -6.65)),
                       "cycle_peak": rng.uniform(0.3, 0.37)
                       },
        "combinact": {"adam_beta_1": np.exp(rng.uniform(-2.15, -1.8)),
                      "adam_beta_2": np.exp(rng.uniform(-9.3, -5.3)),
                      "adam_eps": np.exp(rng.uniform(-20, -19)),
                      "adam_wd": np.exp(rng.uniform(-12, -10.5)),
                      "max_lr": np.exp(rng.uniform(-7.25, -6.35)),
                      "cycle_peak": rng.uniform(0.45, 0.55)
                      },
        "l2": {"adam_beta_1": np.exp(rng.uniform(-2.9, -1.8)),
               "adam_beta_2": np.exp(rng.uniform(-5, -3.8)),
               "adam_eps": np.exp(rng.uniform(-17.1, -16)),
               "adam_wd": np.exp(rng.uniform(-11, -9.5)),
               "max_lr": np.exp(rng.uniform(-7.25, -6.73)),
               "cycle_peak": rng.uniform(0.1, 0.45)
               },
        "abs": {"adam_beta_1": np.exp(rng.uniform(-2.59, -2.22)),
                "adam_beta_2": np.exp(rng.uniform(-9.5, -7.5)),
                "adam_eps": np.exp(rng.uniform(-19, -17.5)),
                "adam_wd": np.exp(rng.uniform(-13, -11.5)),
                "max_lr": np.exp(rng.uniform(-7.25, -6.43)),
                "cycle_peak": rng.uniform(0.35, 0.45)
                },
        "l2_lae": {"adam_beta_1": np.exp(rng.uniform(-2.7, -1.8)),
                   "adam_beta_2": np.exp(rng.uniform(-7, -5)),
                   "adam_eps": np.exp(rng.uniform(-19, -16)),
                   "adam_wd": np.exp(rng.uniform(-12.85, -11)),
                   "max_lr": np.exp(rng.uniform(-7.25, -6.7)),
                   "cycle_peak": rng.uniform(0.35, 0.45)
                   },
        "max": {"adam_beta_1": np.exp(rng.uniform(-3.25, -2.5)),
                "adam_beta_2": np.exp(rng.uniform(-7, -5)),
                "adam_eps": np.exp(rng.uniform(-17.6, -16.6)),
                "adam_wd": np.exp(rng.uniform(-9.14, -8.5)),
                "max_lr": np.exp(rng.uniform(-6.5, -5.85)),
                "cycle_peak": rng.uniform(0.35, 0.45)
                }
    }


def cnn_mnist(rng):
    return {
        "relu": {"adam_beta_1": np.exp(rng.uniform(-3.5, -2)),
                 "adam_beta_2": np.exp(rng.uniform(-5.5, -5.2)),
                 "adam_eps": np.exp(rng.uniform(-21, -16)),
                 "adam_wd": np.exp(rng.uniform(-14, -12.5)),
                 "max_lr": np.exp(rng.uniform(-7.25, -6.25)),
                 "cycle_peak": rng.uniform(0.375, 0.45)
                 },
        "multi_relu": {"adam_beta_1": np.exp(rng.uniform(-4, -3)),
                       "adam_beta_2": np.exp(rng.uniform(-6, -4)),
                       "adam_eps": np.exp(rng.uniform(-21, -19.5)),
                       "adam_wd": np.exp(rng.uniform(-13, -12)),
                       "max_lr": np.exp(rng.uniform(-7.25, -6.5)),
                       "cycle_peak": rng.uniform(0.15, 0.25)
                       },
        "combinact": {"adam_beta_1": np.exp(rng.uniform(-3.75, -1.75)),
                      "adam_beta_2": np.exp(rng.uniform(-6.3, -4.85)),
                      "adam_eps": np.exp(rng.uniform(-21, -19)),
                      "adam_wd": np.exp(rng.uniform(-15, -12.5)),
                      "max_lr": np.exp(rng.uniform(-7, -6)),
                      "cycle_peak": rng.uniform(0.4, 0.5)
                      },
        "l2": {"adam_beta_1": np.exp(rng.uniform(-2.95, -2.54)),
               "adam_beta_2": np.exp(rng.uniform(-5.77, -4.17)),
               "adam_eps": np.exp(rng.uniform(-18, -16)),
               "adam_wd": np.exp(rng.uniform(-15.5, -13)),
               "max_lr": np.exp(rng.uniform(-6.85, -6.5)),
               "cycle_peak": rng.uniform(0.25, 0.4)
               },
        "abs": {"adam_beta_1": np.exp(rng.uniform(-3.23, -2.13)),
                "adam_beta_2": np.exp(rng.uniform(-4.75, -3.75)),
                "adam_eps": np.exp(rng.uniform(-20.25, -18.25)),
                "adam_wd": np.exp(rng.uniform(-14.75, -12.5)),
                "max_lr": np.exp(rng.uniform(-6.7, -5.85)),
                "cycle_peak": rng.uniform(0.375, 0.45)
                },
        "l2_lae": {"adam_beta_1": np.exp(rng.uniform(-4.05, -3.66)),
                   "adam_beta_2": np.exp(rng.uniform(-5.5, -3.75)),
                   "adam_eps": np.exp(rng.uniform(-18.5, -15)),
                   "adam_wd": np.exp(rng.uniform(-14, -12)),
                   "max_lr": np.exp(rng.uniform(-7, -6.15)),
                   "cycle_peak": rng.uniform(0.3, 0.45)
                   },
        "max": {"adam_beta_1": np.exp(rng.uniform(-3.5, -3)),
                "adam_beta_2": np.exp(rng.uniform(-5, -3.75)),
                "adam_eps": np.exp(rng.uniform(-18.75, -16)),
                "adam_wd": np.exp(rng.uniform(-13, -10.5)),
                "max_lr": np.exp(rng.uniform(-5.9, -5.55)),
                "cycle_peak": rng.uniform(0.15, 0.35)
                }
    }
