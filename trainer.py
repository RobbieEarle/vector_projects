import torch
import torch.utils.data
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CyclicLR
import torch.nn.functional as F

import models
import util
import hyper_params as hp

import os
import numpy as np
import csv
import time


# -------------------- Setting Up & Running Training Function

def train(args, actfun, curr_seed, outfile_path, checkpoint, fieldnames, train_loader, validation_loader,
          sample_size, batch_size, device, pfact=1.0):
    """
    Runs training session for a given randomized model
    :param args: arguments for this job
    :param actfun: activation function currently being used
    :param curr_seed: seed being used by current job
    :param outfile_path: path to save outputs from training session
    :param checkpoint: loaded checkpoint
    :param fieldnames: column names for output file
    :param train_loader: training data loader
    :param validation_loader: validation data loader
    :param sample_size: number of training samples used in this experiment
    :param batch_size: number of samples per batch
    :param device: reference to CUDA device for GPU support
    :param pfact: factor by which we reduce the size of our network layers
    :return:
    """

    # ---- Initialization

    model_params = []
    if args.model == 'nn':
        if args.dataset == 'mnist' or args.dataset == 'fashion_mnist':
            input_dim, output_dim = 784, 10
        elif args.dataset == 'cifar10' or args.dataset == 'svhn':
            input_dim, output_dim = 3072, 10
        elif args.dataset == 'cifar100':
            input_dim, output_dim = 3072, 100
        model = models.CombinactNN(actfun=actfun, input_dim=input_dim, output_dim=output_dim, num_layers=2,
                                   k=2, p=1, reduce_actfuns=args.reduce_actfuns, pfact=pfact).to(device)
    elif args.model == 'cnn':
        if args.dataset == 'mnist' or args.dataset == 'fashion_mnist':
            input_channels, input_dim, output_dim = 1, 28, 10
        elif args.dataset == 'cifar10' or args.dataset == 'svhn':
            input_channels, input_dim, output_dim = 3, 32, 10
        elif args.dataset == 'cifar100':
            input_channels, input_dim, output_dim = 3, 32, 100
        model = models.CombinactCNN(actfun=actfun, num_input_channels=input_channels, input_dim=input_dim,
                                    num_outputs=output_dim, k=2, p=1, pfact=pfact,
                                    reduce_actfuns=args.reduce_actfuns).to(device)

        model_params.append({'params': model.conv_layers.parameters()})
        model_params.append({'params': model.pooling.parameters()})

    model_params.append({'params': model.batch_norms.parameters(), 'weight_decay': 0})
    model_params.append({'params': model.linear_layers.parameters()})
    if actfun == 'combinact':
        model_params.append({'params': model.all_alpha_primes.parameters(), 'weight_decay': 0})

    util.seed_all(curr_seed)
    rng = np.random.RandomState(curr_seed)
    model.apply(util.weights_init)

    hyper_params = hp.get_hyper_params(args.model, args.dataset, actfun, rng=rng)

    optimizer = optim.Adam(model_params,
                           lr=10 ** -8,
                           betas=(hyper_params['adam_beta_1'], hyper_params['adam_beta_2']),
                           eps=hyper_params['adam_eps'],
                           weight_decay=hyper_params['adam_wd']
                           )
    criterion = nn.CrossEntropyLoss()
    num_batches = sample_size / batch_size * args.num_epochs
    scheduler = CyclicLR(optimizer,
                         base_lr=10 ** -8,
                         max_lr=hyper_params['max_lr'],
                         step_size_up=int(hyper_params['cycle_peak'] * num_batches),
                         step_size_down=int((1 - hyper_params['cycle_peak']) * num_batches),
                         cycle_momentum=False
                         )

    epoch = 1
    checkpoint_location = os.path.join(args.check_path, "cp_{}.pth".format(args.seed))
    if checkpoint is not None:
        checkpoint = torch.load(checkpoint_location)
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        epoch = checkpoint['epoch']
        model.to(device)
        print("*** LOADED CHECKPOINT ***"
              "\n  Epoch: {}"
              "\n  Actfun: {}"
              "\n  Parameter Factor: {}"
              "\n  Num Training Samples: {}"
              "\n  Seed: {}".format(epoch, actfun, pfact, sample_size, curr_seed))

    util.print_exp_settings(curr_seed, args.dataset, outfile_path, args.model, actfun, hyper_params,
                            util.get_n_params(model), sample_size)

    # ---- Start Training
    while epoch <= args.num_epochs:

        if args.check_path != '':
            torch.save({'state_dict': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'scheduler': scheduler.state_dict(),
                        'epoch': epoch,
                        'actfun': actfun,
                        'param_factor': pfact,
                        'train_sample': sample_size,
                        'curr_seed': curr_seed}, checkpoint_location)
            print("*** SAVED CHECKPOINT ***")

        print("------> Epoch {}".format(epoch))
        util.seed_all((curr_seed * args.num_epochs) + epoch)
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
            "    Epoch {} Completed: train_loss = {:1.6f}  |  val_loss = {:1.6f}  |  accuracy = {:1.6f}  |  time = {}"
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
                             'batch_size': batch_size,
                             'alpha_primes': alpha_primes,
                             'alphas': alphas,
                             'num_params': util.get_n_params(model),
                             'var_nparams': args.var_n_params,
                             'var_nsamples': args.var_n_samples
                             })

        epoch += 1
        print()