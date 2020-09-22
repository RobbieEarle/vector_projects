import torch
import torch.utils.data
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CyclicLR
from torch.optim.lr_scheduler import OneCycleLR
import torch.nn.functional as F

import math
import models
import util
import hyper_params as hp

import numpy as np
import csv
import time


# -------------------- Setting Up & Running Training Function

def train(args, checkpoint, checkpoint_location, actfun, curr_seed, outfile_path, fieldnames, train_loader,
          validation_loader, sample_size, batch_size, device, num_params, curr_k=2, curr_p=1, curr_g=1,
          perm_method='shuffle'):
    """
    Runs training session for a given randomized model
    :param args: arguments for this job
    :param checkpoint: current checkpoint
    :param actfun: activation function currently being used
    :param curr_seed: seed being used by current job
    :param outfile_path: path to save outputs from training session
    :param fieldnames: column names for output file
    :param train_loader: training data loader
    :param validation_loader: validation data loader
    :param sample_size: number of training samples used in this experiment
    :param batch_size: number of samples per batch
    :param device: reference to CUDA device for GPU support
    :return:
    """

    model = models.DawnNet().to(device)

    util.seed_all(curr_seed)
    rng = np.random.RandomState(curr_seed)
    # model.apply(util.weights_init)

    criterion = nn.CrossEntropyLoss()
    hyper_params = hp.get_hyper_params(args.model, args.dataset, actfun, rng=rng, exp=args.hyper_params, p=curr_p)

    # optimizer = optim.SGD(model.parameters(), lr=0.000001, momentum=0.9, weight_decay=5e-4)
    optimizer = optim.Adam(model.parameters(), lr=0.000001, betas=(0.9, 0.99), weight_decay=5e-4)
    hyper_params['max_lr'] = 0.001
    hyper_params['cycle_peak'] = 0.4

    scheduler = OneCycleLR(optimizer,
                           max_lr=hyper_params['max_lr'],
                           epochs=args.num_epochs,
                           steps_per_epoch=int(math.ceil(sample_size / batch_size)),
                           pct_start=hyper_params['cycle_peak'],
                           cycle_momentum=False
                           )

    epoch = 1

    if checkpoint is not None:
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        epoch = checkpoint['epoch']
        model.to(device)
        print("*** LOADED CHECKPOINT ***"
              "\n{}"
              "\nSeed: {}"
              "\nEpoch: {}"
              "\nActfun: {}"
              "\nNum Params: {}"
              "\nSample Size: {}"
              "\np: {}"
              "\nk: {}"
              "\ng: {}"
              "\nperm_method: {}".format(checkpoint_location, checkpoint['curr_seed'],
                                         checkpoint['epoch'], checkpoint['actfun'],
                                         checkpoint['num_params'], checkpoint['sample_size'],
                                         checkpoint['p'], checkpoint['k'], checkpoint['g'],
                                         checkpoint['perm_method']))

    util.print_exp_settings(curr_seed, args.dataset, outfile_path, args.model, actfun, hyper_params,
                            util.get_model_params(model), sample_size, 1, 1, 1,
                            perm_method, 0, 0, False)

    # ---- Start Training
    while epoch <= args.num_epochs:

        if args.check_path != '':
            torch.save({'state_dict': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'scheduler': scheduler.state_dict(),
                        'curr_seed': curr_seed,
                        'epoch': epoch,
                        'actfun': actfun,
                        'num_params': num_params,
                        'sample_size': sample_size,
                        'p': curr_p, 'k': curr_k, 'g': curr_g,
                        'perm_method': perm_method
                        }, checkpoint_location)

        util.seed_all((curr_seed * args.num_epochs) + epoch)
        start_time = time.time()

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

        model.eval()
        with torch.no_grad():
            total_train_loss, n, num_correct, num_total = 0, 0, 0, 0
            for batch_idx, (x, targetx) in enumerate(train_loader):
                x, targetx = x.to(device), targetx.to(device)
                output = model(x)
                train_loss = criterion(output, targetx)
                total_train_loss += train_loss
                n += 1
                _, prediction = torch.max(output.data, 1)
                num_correct += torch.sum(prediction == targetx.data)
                num_total += len(prediction)
            eval_train_loss = total_train_loss / n
            eval_train_acc = num_correct * 1.0 / num_total

            total_val_loss, n, num_correct, num_total = 0, 0, 0, 0
            for batch_idx2, (y, targety) in enumerate(validation_loader):
                y, targety = y.to(device), targety.to(device)
                output = model(y)
                val_loss = criterion(output, targety)
                total_val_loss += val_loss
                n += 1
                _, prediction = torch.max(output.data, 1)
                num_correct += torch.sum(prediction == targety.data)
                num_total += len(prediction)
            eval_val_loss = total_val_loss / n
            eval_val_acc = num_correct * 1.0 / num_total

        lr = ''
        for param_group in optimizer.param_groups:
            lr = param_group['lr']

        # Logging test results
        print(
            "    Epoch {}: LR {:1.5f}  |  train_acc {:1.5f}  |  val_acc {:1.5f}  |  train_loss {:1.5f}  |  val_loss {:1.5f}  |  time = {:1.5f}"
                .format(epoch, lr, eval_train_acc, eval_val_acc, eval_train_loss, eval_val_loss,
                        (time.time() - start_time)), flush=True
        )

        # Outputting data to CSV at end of epoch
        with open(outfile_path, mode='a') as out_file:
            writer = csv.DictWriter(out_file, fieldnames=fieldnames, lineterminator='\n')
            writer.writerow({'dataset': args.dataset,
                             'seed': curr_seed,
                             'epoch': epoch,
                             'time': (time.time() - start_time),
                             'actfun': args.actfun,
                             'sample_size': sample_size,
                             'hyper_params': hyper_params,
                             'model': args.model,
                             'batch_size': batch_size,
                             'alpha_primes': [],
                             'alphas': [],
                             'num_params': util.get_model_params(model),
                             'var_nparams': args.var_n_params,
                             'var_nsamples': args.var_n_samples,
                             'k': curr_k,
                             'p': curr_p,
                             'g': curr_g,
                             'perm_method': perm_method,
                             'gen_gap': float(eval_val_loss - eval_train_loss),
                             'resnet_ver': 0,
                             'resnet_width': 0,
                             'train_loss': float(eval_train_loss),
                             'val_loss': float(eval_val_loss),
                             'train_acc': float(eval_train_acc),
                             'val_acc': float(eval_val_acc)
                             })

        epoch += 1
