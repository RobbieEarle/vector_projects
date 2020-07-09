import torch
import torch.utils.data
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CyclicLR

import models
import util

import argparse
import os
import numpy as np
import datetime
import csv
import time

# -------------------- Optimized Hyper Parameter Settings

_HYPERPARAMS = {
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


# -------------------- Setting Up & Running Training Function

def train_model(args,
                outfile_path,
                fieldnames,
                train_loader,
                validation_loader,
                sample_size,
                device):
    """
    Runs training session for a given randomized model
    :param args: arguments for this job
    :param outfile_path: path to save outputs from training session
    :param fieldnames: column names for output file
    :param train_loader: training data loader
    :param validation_loader: validation data loader
    :param sample_size: number of training samples used in this experiment
    :param device: reference to CUDA device for GPU support
    :return:
    """

    # ---- Initialization

    model_params = []
    if args.dataset == 'mnist':
        model = models.CombinactNN(actfun=args.actfun, input_dim=784,
                                   output_dim=10, num_layers=2, k=2, p=1).to(device)
    if args.model == 'nn':
        pass
    elif args.model == 'cnn':
        if args.dataset == 'mnist':
            model = models.CombinactCNN(actfun=args.actfun, num_input_channels=1, input_dim=28,
                                        num_outputs=10, k=2, p=1).to(device)
        elif args.dataset == 'cifar10':
            model = models.CombinactCNN(actfun=args.actfun, num_input_channels=3, input_dim=32,
                                        num_outputs=10, k=2, p=1).to(device)
        model_params.append({'params': model.conv_layers.parameters()})
        model_params.append({'params': model.pooling.parameters()})

    model_params.append({'params': model.batch_norms.parameters(), 'weight_decay': 0})
    model_params.append({'params': model.linear_layers.parameters()})
    if args.actfun == 'combinact':
        model_params.append({'params': model.all_alpha_primes.parameters(), 'weight_decay': 0})

    util.seed_all(args.seed)
    rng = np.random.RandomState(args.seed)
    model.apply(util.weights_init)

    if args.randsearch:
        hyper_params = util.get_random_hyper_params(rng)[args.actfun]
    else:
        hyper_params = _HYPERPARAMS[args.actfun]

    util.print_exp_settings(args.seed, args.dataset, outfile_path, args.model, args.actfun, hyper_params)

    optimizer = optim.Adam(model_params,
                           lr=10 ** -8,
                           betas=(hyper_params['adam_beta_1'], hyper_params['adam_beta_2']),
                           eps=hyper_params['adam_eps'],
                           weight_decay=hyper_params['adam_wd']
                           )
    criterion = nn.CrossEntropyLoss()
    num_batches = args.sample_size / args.batch_size * args.num_epochs
    scheduler = CyclicLR(optimizer,
                         base_lr=10 ** -8,
                         max_lr=hyper_params['max_lr'],
                         step_size_up=int(hyper_params['cycle_peak'] * num_batches),
                         step_size_down=int((1 - hyper_params['cycle_peak']) * num_batches),
                         cycle_momentum=False
                         )

    # ---- Start Training
    epoch = 1
    while epoch <= args.num_epochs:
        util.seed_all(args.seed+epoch)
        start_time = time.time()
        final_train_loss = 0
        # ---- Training
        model.train()
        for batch_idx, (x, targetx) in enumerate(train_loader):
            x, targetx = x.to(device), targetx.to(device)
            optimizer.zero_grad()
            output = model(x)
            train_loss = criterion(output, targetx)
            train_loss.backward()
            optimizer.step()
            scheduler.step()
            final_train_loss = train_loss

        # ---- Testing
        num_correct = 0
        num_total = 0
        final_val_loss = 0
        model.eval()
        with torch.no_grad():
            for batch_idx2, (y, targety) in enumerate(validation_loader):
                y, targety = y.to(device), targety.to(device)
                output = model(y)
                val_loss = criterion(output, targety)
                final_val_loss = val_loss
                _, prediction = torch.max(output.data, 1)
                num_correct += torch.sum(prediction == targety.data)
                num_total += len(prediction)
        accuracy = num_correct * 1.0 / num_total

        # Logging test results
        print(
            "    Epoch {}: train_loss = {:1.6f}  |  val_loss = {:1.6f}  |  accuracy = {:1.6f}  |  time = {}"
                .format(epoch, final_train_loss, final_val_loss, accuracy, (time.time() - start_time)), flush=True
        )

        # Outputting data to CSV at end of epoch
        with open(outfile_path, mode='a') as out_file:
            writer = csv.DictWriter(out_file, fieldnames=fieldnames, lineterminator='\n')
            writer.writerow({'dataset': args.dataset,
                             'seed': args.seed,
                             'epoch': epoch,
                             'train_loss': float(final_train_loss),
                             'val_loss': float(final_val_loss),
                             'acc': float(accuracy),
                             'time': (time.time() - start_time),
                             'actfun': model.actfun,
                             'sample_size': sample_size,
                             'hyper_params': hyper_params,
                             'model': args.model,
                             'batch_size': args.batch_size
                             })

        epoch += 1


def setup_experiment(args, outfile_path):
    """
    Retrieves training / validation data, randomizes network structure and activation functions, creates model,
    creates new output file, sets hyperparameters for optimizer and scheduler during training, initializes training
    :param seed: seed for parameter randomization
    :param outfile_path: path to save outputs from experiment
    :param actfun: model architecture
    :return:
    """

    # ---- Create new output file
    fieldnames = ['dataset', 'seed', 'epoch', 'train_loss', 'val_loss', 'acc', 'time', 'actfun',
                  'sample_size', 'hyper_params', 'model', 'batch_size']
    with open(outfile_path, mode='w') as out_file:
        writer = csv.DictWriter(out_file, fieldnames=fieldnames, lineterminator='\n')
        writer.writeheader()

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

    # ---- Loading Dataset
    curr_seed = args.seed
    print()
    train_loader, validation_loader, sample_size = util.load_dataset(args.dataset,
                                                                     seed=curr_seed,
                                                                     batch_size=args.batch_size,
                                                                     sample_size=args.sample_size,
                                                                     kwargs=kwargs)

    # ---- Begin training model
    util.seed_all(curr_seed)
    train_model(args, outfile_path, fieldnames, train_loader, validation_loader, sample_size, device)
    print()


# --------------------  Entry Point
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--seed', type=int, default=0, help='Job seed')
    parser.add_argument('--actfun', type=str, default='relu',
                        help='relu, multi_relu, cf_relu, combinact, l1, l2, l2_lae, abs, max'
                        )
    parser.add_argument('--save_path', type=str, default='', help='Where to save results')
    parser.add_argument('--dataset', type=str, default='cifar10', help='Dataset being used. mnist or cifar10')
    parser.add_argument('--model', type=str, default='cnn', help='What type of model to use')
    parser.add_argument('--sample_size', type=int, default=50000, help='Training sample size')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size during training')
    parser.add_argument('--num_epochs', type=int, default=10, help='Number of training epochs')
    parser.add_argument('--randsearch', action='store_true', help='Creates random hyper-parameter search')
    args = parser.parse_args()

    out = os.path.join(
        args.save_path,
        '{}-{}-{}-{}-{}.csv'.format(
            datetime.date.today(),
            args.actfun,
            args.seed,
            args.dataset,
            args.model)
    )

    setup_experiment(args, out)
