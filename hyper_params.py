import numpy as np


def get_hyper_params(model, dataset, actfun):

    if model == 'nn' and dataset == 'mnist':
        return NN_MNIST[actfun]
    elif model == 'cnn' and dataset == 'cifar10':
        return CNN_CIFAR10[actfun]
    else:
        raise ValueError("Error: No hyper-parameters found for {}, {}".format(model, dataset))


NN_MNIST = {
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


CNN_CIFAR10 = {
    "relu": {"adam_beta_1": np.exp(-2.5713946178339486),
             "adam_beta_2": np.exp(-8.088852495066451),
             "adam_eps": np.exp(-18.24053115491185),
             "adam_wd": np.exp(-12.007877998144522),
             "max_lr": np.exp(-7.277101799190481),
             "cycle_peak": 0.364970594416471
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
    "abs": {"adam_beta_1": np.exp(-1.6509199183692422),
            "adam_beta_2": np.exp(-5.449016919866456),
            "adam_eps": np.exp(-18.97360098070963),
            "adam_wd": np.exp(-11.927993917764805),
            "max_lr": np.exp(-7.591007314708498),
            "cycle_peak": 0.48552168878517715
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
