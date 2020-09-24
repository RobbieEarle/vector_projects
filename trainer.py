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


# -------------------- Loading Model
def load_model(model, dataset, actfun, k, p, g, num_params, perm_method, device, resnet_ver, resnet_width, verbose):

    model_params = []

    if dataset == 'mnist' or dataset == 'fashion_mnist':
        input_channels, input_dim, output_dim = 1, 28, 10
    elif dataset == 'cifar10' or dataset == 'svhn':
        input_channels, input_dim, output_dim = 3, 32, 10
    elif dataset == 'cifar100':
        input_channels, input_dim, output_dim = 3, 32, 100

    if model == 'nn' or model == 'mlp':
        if dataset == 'mnist' or dataset == 'fashion_mnist':
            input_dim = 784
        elif dataset == 'cifar10' or dataset == 'svhn':
            input_dim = 3072
        elif dataset == 'cifar100':
            input_dim = 3072
        model = models.CombinactMLP(actfun=actfun,
                                    input_dim=input_dim,
                                    output_dim=output_dim,
                                    k=k,
                                    p=p,
                                    g=g,
                                    num_params=num_params,
                                    permute_type=perm_method).to(device)
        model_params.append({'params': model.batch_norms.parameters(), 'weight_decay': 0})
        model_params.append({'params': model.linear_layers.parameters()})
        if actfun == 'combinact':
            model_params.append({'params': model.all_alpha_primes.parameters(), 'weight_decay': 0})

    elif model == 'cnn':
        model = models.CombinactCNN(actfun=actfun,
                                    num_input_channels=input_channels,
                                    input_dim=input_dim,
                                    num_outputs=output_dim,
                                    k=k,
                                    p=p,
                                    g=g,
                                    num_params=num_params,
                                    permute_type=perm_method).to(device)

        model_params.append({'params': model.conv_layers.parameters()})
        model_params.append({'params': model.pooling.parameters()})
        model_params.append({'params': model.batch_norms.parameters(), 'weight_decay': 0})
        model_params.append({'params': model.linear_layers.parameters()})
        if actfun == 'combinact':
            model_params.append({'params': model.all_alpha_primes.parameters(), 'weight_decay': 0})

    elif model == 'resnet':
        model = models.ResNet(resnet_ver=resnet_ver,
                              actfun=actfun,
                              num_input_channels=input_channels,
                              num_outputs=output_dim,
                              k=k,
                              p=p,
                              g=g,
                              permute_type=perm_method,
                              width=resnet_width,
                              verbose=verbose).to(device)

        model_params = model.parameters()

    return model, model_params


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

    if actfun == 'relu':
        curr_k = 1

    model, model_params = load_model(args.model, args.dataset, actfun, curr_k, curr_p, curr_g, num_params=num_params,
                                     perm_method=perm_method, device=device, resnet_ver=args.resnet_ver,
                                     resnet_width=args.resnet_width, verbose=args.verbose)

    util.seed_all(curr_seed)
    rng = np.random.RandomState(curr_seed)
    model.apply(util.weights_init)

    criterion = nn.CrossEntropyLoss()
    hyper_params = hp.get_hyper_params(args.model, args.dataset, actfun, rng=rng, exp=args.hyper_params, p=curr_p)

    num_epochs = args.num_epochs
    if args.overfit:
        num_epochs = 50
        hyper_params['cycle_peak'] = 0.35
    hyper_params['adam_wd'] *= args.wd

    if args.model == 'resnet':
        hyper_params['adam_beta_1'] = 0.9
        hyper_params['adam_beta_2'] = 0.99
        hyper_params['adam_eps'] = 1e-8
        hyper_params['adam_wd'] = 5e-4
        hyper_params['max_lr'] = 0.001
        hyper_params['cycle_peak'] = 0.4

    optimizer = optim.Adam(model_params,
                           lr=10 ** -6,
                           betas=(hyper_params['adam_beta_1'], hyper_params['adam_beta_2']),
                           eps=hyper_params['adam_eps'],
                           weight_decay=hyper_params['adam_wd']
                           )

    if args.model == 'resnet':
        scheduler = OneCycleLR(optimizer,
                               max_lr=hyper_params['max_lr'],
                               epochs=num_epochs,
                               steps_per_epoch=int(math.ceil(sample_size / batch_size)),
                               pct_start=hyper_params['cycle_peak'],
                               cycle_momentum=False
                               )
    else:
        num_batches = (sample_size / batch_size) * num_epochs
        scheduler = CyclicLR(optimizer,
                             base_lr=10 ** -8,
                             max_lr=hyper_params['max_lr'],
                             step_size_up=int(hyper_params['cycle_peak'] * num_batches),
                             step_size_down=int((1 - hyper_params['cycle_peak']) * num_batches),
                             cycle_momentum=False
                             )

    epoch = 1
    if args.model == 'resnet':
        resnet_ver = args.resnet_ver
        resnet_width = args.resnet_width
    else:
        resnet_ver = 0
        resnet_width = 0

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
                            util.get_model_params(model), sample_size, model.k, model.p, model.g,
                            perm_method, resnet_ver, resnet_width)

    # ---- Start Training
    while epoch <= num_epochs:

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

        alpha_primes = []
        alphas = []
        for i, layer_alpha_primes in enumerate(model.all_alpha_primes):
            curr_alpha_primes = torch.mean(layer_alpha_primes, dim=0)
            curr_alphas = F.softmax(curr_alpha_primes, dim=0).data.tolist()
            curr_alpha_primes = curr_alpha_primes.tolist()
            alpha_primes.append(curr_alpha_primes)
            alphas.append(curr_alphas)

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
        print(
            "    Epoch {}: LR {:1.5f}  |  train_acc {:1.5f}  |  val_acc {:1.5f}  |  train_loss {:1.5f}  |  val_loss {:1.5f}  |  time = {:1.5f}"
                .format(epoch, lr, eval_train_acc, eval_val_acc, eval_train_loss, eval_val_loss, (time.time() - start_time)), flush=True
        )

        # Outputting data to CSV at end of epoch
        with open(outfile_path, mode='a') as out_file:
            writer = csv.DictWriter(out_file, fieldnames=fieldnames, lineterminator='\n')
            writer.writerow({'dataset': args.dataset,
                             'seed': curr_seed,
                             'epoch': epoch,
                             'time': (time.time() - start_time),
                             'actfun': model.actfun,
                             'sample_size': sample_size,
                             'hyper_params': hyper_params,
                             'model': args.model,
                             'batch_size': batch_size,
                             'alpha_primes': alpha_primes,
                             'alphas': alphas,
                             'num_params': util.get_model_params(model),
                             'var_nparams': args.var_n_params,
                             'var_nsamples': args.var_n_samples,
                             'k': curr_k,
                             'p': curr_p,
                             'g': curr_g,
                             'perm_method': perm_method,
                             'gen_gap': float(eval_val_loss - eval_train_loss),
                             'resnet_ver': resnet_ver,
                             'resnet_width': resnet_width,
                             'train_loss': float(eval_train_loss),
                             'val_loss': float(eval_val_loss),
                             'train_acc': float(eval_train_acc),
                             'val_acc': float(eval_val_acc)
                             })

        epoch += 1
