import torch
import torch.utils.data
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CyclicLR
import torch.nn.functional as F

import models
import util
import hyper_params as hp

import argparse
import os
import numpy as np
import datetime
import csv
import time


# -------------------- Setting Up & Running Training Function

def train_model(args,
                curr_seed,
                outfile_path,
                fieldnames,
                train_loader,
                validation_loader,
                sample_size,
                device,
                pfact=1):
    """
    Runs training session for a given randomized model
    :param args: arguments for this job
    :param curr_seed: seed being used by current job
    :param outfile_path: path to save outputs from training session
    :param fieldnames: column names for output file
    :param train_loader: training data loader
    :param validation_loader: validation data loader
    :param sample_size: number of training samples used in this experiment
    :param device: reference to CUDA device for GPU support
    :param pfact: factor by which we reduce the size of our network layers
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
                                        num_outputs=10, k=2, p=1,
                                        pfact=pfact).to(device)
        elif args.dataset == 'cifar10':
            model = models.CombinactCNN(actfun=args.actfun, num_input_channels=3, input_dim=32,
                                        num_outputs=10, k=2, p=1,
                                        pfact=pfact).to(device)
        elif args.dataset == 'cifar100':
            model = models.CombinactCNN(actfun=args.actfun, num_input_channels=3, input_dim=32,
                                        num_outputs=100, k=2, p=1,
                                        pfact=pfact).to(device)
        model_params.append({'params': model.conv_layers.parameters()})
        model_params.append({'params': model.pooling.parameters()})

    model_params.append({'params': model.batch_norms.parameters(), 'weight_decay': 0})
    model_params.append({'params': model.linear_layers.parameters()})
    if args.actfun == 'combinact':
        model_params.append({'params': model.all_alpha_primes.parameters(), 'weight_decay': 0})

    util.seed_all(curr_seed)
    rng = np.random.RandomState(curr_seed)
    model.apply(util.weights_init)

    if args.randsearch:
        hyper_params = util.get_random_hyper_params(rng)[args.actfun]
    else:
        hyper_params = hp.get_hyper_params(args.model, args.dataset, args.actfun)

    util.print_exp_settings(curr_seed, args.dataset, outfile_path, args.model, args.actfun, hyper_params,
                            util.get_n_params(model), sample_size)

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
        util.seed_all(curr_seed+epoch)
        start_time = time.time()
        final_train_loss = 0
        # ---- Training
        model.train()
        for batch_idx, (x, targetx) in enumerate(train_loader):
            # print(batch_idx)
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

        alpha_primes = []
        alphas = []
        for i, layer_alpha_primes in enumerate(model.all_alpha_primes):
            curr_alpha_primes = torch.mean(layer_alpha_primes, dim=0)
            curr_alphas = F.softmax(curr_alpha_primes, dim=0).data.tolist()
            curr_alpha_primes = curr_alpha_primes.tolist()
            alpha_primes.append(curr_alpha_primes)
            alphas.append(curr_alphas)

        # Outputting data to CSV at end of epoch
        with open(outfile_path, mode='a') as out_file:
            writer = csv.DictWriter(out_file, fieldnames=fieldnames, lineterminator='\n')
            writer.writerow({'dataset': args.dataset,
                             'seed': curr_seed,
                             'epoch': epoch,
                             'train_loss': float(final_train_loss),
                             'val_loss': float(final_val_loss),
                             'acc': float(accuracy),
                             'time': (time.time() - start_time),
                             'actfun': model.actfun,
                             'sample_size': sample_size,
                             'hyper_params': hyper_params,
                             'model': args.model,
                             'batch_size': args.batch_size,
                             'alpha_primes': alpha_primes,
                             'alphas': alphas,
                             'num_params': util.get_n_params(model)
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
                  'sample_size', 'hyper_params', 'model', 'batch_size', 'alpha_primes', 'alphas',
                  'num_params']
    with open(outfile_path, mode='w') as out_file:
        writer = csv.DictWriter(out_file, fieldnames=fieldnames, lineterminator='\n')
        writer.writeheader()

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

    if args.var_n_params:
        param_factors = [1.01, 0.925, 0.83, 0.72, 0.59, 0.41, 0.29]
    elif args.var_n_relu_params:
        param_factors = [0.36, 0.33, 0.295, 0.255, 0.21, 0.1475, 0.105]
    else:
        if args.actfun == 'relu' or args.actfun == 'abs':
            param_factors = [0.36]
        else:
            param_factors = [1]

    if args.var_n_samples:
        train_samples = [50000, 45000, 40000, 35000, 30000, 25000, 20000, 15000, 10000, 5000]
    else:
        train_samples = [args.sample_size]

    for i, pfact in enumerate(param_factors):
        for j, curr_sample_size in enumerate(train_samples):

            # ---- Loading Dataset
            print()
            curr_seed = args.seed + ((i+j) * 50)
            train_loader, validation_loader, sample_size = util.load_dataset(args.dataset,
                                                                             seed=curr_seed,
                                                                             batch_size=args.batch_size,
                                                                             sample_size=curr_sample_size,
                                                                             kwargs=kwargs)
            # ---- Begin training model
            util.seed_all(curr_seed)
            train_model(args, curr_seed, outfile_path, fieldnames, train_loader, validation_loader, sample_size, device, pfact=pfact)
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
    parser.add_argument('--var_n_params', action='store_true', help='When true, varies number of network parameters')
    parser.add_argument('--var_n_relu_params', action='store_true', help='Same as var_n_params, but for 1d actfuns')
    parser.add_argument('--var_n_samples', action='store_true', help='When true, varies number of training samples')
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
