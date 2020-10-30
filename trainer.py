import torch
import torch.utils.data
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ExponentialLR
from torch.optim.lr_scheduler import CyclicLR
from torch.optim.lr_scheduler import OneCycleLR
import torch.nn.functional as F

from torch_lr_finder import LRFinder
import matplotlib.pyplot as plt

import math
from models import mlp
from models import cnn
from models import preact_resnet
import util
import hyper_params_new as hp

import numpy as np
import csv
import time
import os


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
        elif dataset == 'iris':
            input_dim, output_dim = 4, 3
        model = mlp.MLP(actfun=actfun,
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
        model = cnn.CNN(actfun=actfun,
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
        model = preact_resnet.PreActResNet(resnet_ver=resnet_ver,
                                           actfun=actfun,
                                           in_channels=input_channels,
                                           out_channels=output_dim,
                                           k=k,
                                           p=p,
                                           g=g,
                                           permute_type=perm_method,
                                           width=resnet_width,
                                           verbose=verbose).to(device)

        model_params = model.parameters()

    return model, model_params


# -------------------- Setting Up & Running Training Function
def train(args, checkpoint, mid_checkpoint_location, final_checkpoint_location, best_checkpoint_location,
          actfun, curr_seed, outfile_path, filename, fieldnames, loaders, sample_size,
          batch_size, device, num_params, curr_k=2, curr_p=1, curr_g=1, perm_method='shuffle'):
    """
    Runs training session for a given randomized model
    :param args: arguments for this job
    :param checkpoint: current checkpoint
    :param checkpoint_location: output directory for checkpoints
    :param actfun: activation function currently being used
    :param curr_seed: seed being used by current job
    :param outfile_path: path to save outputs from training session
    :param fieldnames: column names for output file
    :param loaders: the train / eval loaders
    :param sample_size: number of training samples used in this experiment
    :param batch_size: number of samples per batch
    :param device: reference to CUDA device for GPU support
    :param num_params: number of parameters in the network
    :param curr_k: k value for this iteration
    :param curr_p: p value for this iteration
    :param curr_g: g value for this iteration
    :param perm_method: permutation strategy for our network
    :return:
    """

    if actfun == 'relu':
        curr_k = 1
        resnet_ver = args.resnet_ver
        resnet_width = args.resnet_width
    elif actfun == 'bin_all_max_min' or actfun == 'bin_all_max_sgm':
        resnet_ver = args.resnet_ver
        resnet_width = 2
    elif actfun == 'bin_all_max_min_sgm':
        resnet_ver = args.resnet_ver
        resnet_width = 1.5625
    else:
        resnet_ver = args.resnet_ver
        resnet_width = args.resnet_width + math.ceil(curr_k/2)
    if args.model != 'resnet':
        resnet_ver = 0
        resnet_width = 0

    actfuns_1d = ['relu', 'abs', 'swish', 'leaky_relu']
    if actfun in actfuns_1d:
        curr_k = 1

    model, model_params = load_model(args.model, args.dataset, actfun, curr_k, curr_p, curr_g, num_params=num_params,
                                     perm_method=perm_method, device=device, resnet_ver=resnet_ver,
                                     resnet_width=resnet_width, verbose=args.verbose)

    util.seed_all(curr_seed)
    rng = np.random.RandomState(curr_seed)
    model.apply(util.weights_init)

    print("=============================== Hyper params:")
    i = 0
    for name, param in model.named_parameters():
        # print(name, param.shape)
        if len(param.shape) == 4:
            print(param[:2, :2, :4, :4])
            break
        elif len(param.shape) == 3:
            print(param[:2, :2, :2])
        elif len(param.shape) == 2:
            print(param[:4, :4])
            break
        elif len(param.shape) == 1:
            print(param[:3])
        print()
        i += 1
        if i == 4:
            break
    print("===================================================================")

    criterion = nn.CrossEntropyLoss()
    hyper_params = hp.get_hyper_params(args.grid_id)

    num_epochs = args.num_epochs

    # if args.grid_id is None:
    #     if args.lr_init is not None:
    #         lr_init = args.lr_init
    #     elif args.model == 'mlp':
    #         lr_init = 0.01
    #     elif args.model == 'cnn':
    #         lr_init = 0.001
    #     elif args.model == 'resnet':
    #         lr_init = 0.001
    # else:
    lr_init = args.lr_init
    lr_gamma = args.lr_gamma
    rms_alpha = 0.99
    rms_momentum = 0
    wd = 1e-4
    # lr_init, lr_gamma, rms_alpha, rms_momentum = util.get_rms_hyperparams(args)

    # # optimizer = optim.RMSprop(model_params, lr=lr_init, weight_decay=wd, alpha=rms_alpha, momentum=rms_momentum)
    # optimizer = optim.RMSprop(model_params, lr=0.01, weight_decay=1e-3)
    # scheduler = ExponentialLR(optimizer, gamma=0.99)

    if args.lr_range:
        print("Running learning rate finder")
        optimizer = optim.Adam(model.parameters(), lr=1e-7, weight_decay=1e-4)
        lr_finder = LRFinder(model, optimizer, criterion, device=device)
        lr_finder.range_test(loaders['aug_train'], end_lr=100, num_iter=100, diverge_th=3)
        print("Plotting learning rate finder results")
        hf = plt.figure(figsize=(15, 9))
        ax = plt.axes()
        lr_finder.plot(skip_start=0, skip_end=1, log_lr=True, ax=ax)
        plt.tick_params(reset=True, color=(0.2, 0.2, 0.2))
        plt.tick_params(labelsize=14)
        ax.minorticks_on()
        ax.tick_params(direction="out")
        ax.set_ylim([None, 4.65])
        # Save figure
        figpth = os.path.join(args.save_path, filename) + '_lrfinder.png'
        plt.savefig(figpth)
        print("LR Finder results saved to {}".format(figpth))

    else:
        if args.optim == 'onecycle':
            lr_init = 10 ** -6
            optimizer = optim.Adam(model_params,
                                   lr=lr_init,
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
        # elif args.optim == 'rmsprop':
        #     if args.grid_id is None:
        #         if args.lr_init is not None:
        #             lr_init = args.lr_init
        #         elif args.model == 'mlp':
        #             lr_init = 0.01
        #         elif args.model == 'cnn':
        #             lr_init = 0.001
        #         elif args.model == 'resnet':
        #             lr_init = 0.001
        #     else:
        #         wd = 1e-4
        #         lr_init, lr_gamma, rms_alpha, rms_momentum = util.get_rms_hyperparams(args)
        #
        #     optimizer = optim.RMSprop(model_params, lr=lr_init, weight_decay=wd, alpha=rms_alpha, momentum=rms_momentum)
        #     scheduler = ExponentialLR(optimizer, gamma=lr_gamma)

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
                  "\nperm_method: {}".format(mid_checkpoint_location, checkpoint['curr_seed'],
                                             checkpoint['epoch'], checkpoint['actfun'],
                                             checkpoint['num_params'], checkpoint['sample_size'],
                                             checkpoint['p'], checkpoint['k'], checkpoint['g'],
                                             checkpoint['perm_method']))

        util.print_exp_settings(curr_seed, args.dataset, outfile_path, args.model, actfun, hyper_params,
                                util.get_model_params(model), sample_size, model.k, model.p, model.g,
                                perm_method, resnet_ver, resnet_width, args.optim, args.validation,
                                lr_init, lr_gamma, wd, rms_alpha, rms_momentum)

        best_val_acc = 0

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
                            }, mid_checkpoint_location)

            util.seed_all((curr_seed * args.num_epochs) + epoch)
            start_time = time.time()
            scaler = torch.cuda.amp.GradScaler()

            # ---- Training
            model.train()
            total_train_loss, n, num_correct, num_total = 0, 0, 0, 0
            for batch_idx, (x, targetx) in enumerate(loaders['aug_train']):
                # print(batch_idx)
                x, targetx = x.to(device), targetx.to(device)
                optimizer.zero_grad()
                with torch.cuda.amp.autocast():
                    output = model(x)
                    train_loss = criterion(output, targetx)
                total_train_loss += train_loss
                n += 1
                scaler.scale(train_loss).backward()
                scaler.step(optimizer)
                scaler.update()
                if args.optim == 'onecycle':
                    scheduler.step()
                _, prediction = torch.max(output.data, 1)
                num_correct += torch.sum(prediction == targetx.data)
                num_total += len(prediction)
                for param_group in optimizer.param_groups:
                    lr_temp = param_group['lr']
                print(lr_temp)
            epoch_aug_train_loss = total_train_loss / n
            epoch_aug_train_acc = num_correct * 1.0 / num_total

            alpha_primes = []
            alphas = []
            if model.actfun == 'combinact':
                for i, layer_alpha_primes in enumerate(model.all_alpha_primes):
                    curr_alpha_primes = torch.mean(layer_alpha_primes, dim=0)
                    curr_alphas = F.softmax(curr_alpha_primes, dim=0).data.tolist()
                    curr_alpha_primes = curr_alpha_primes.tolist()
                    alpha_primes.append(curr_alpha_primes)
                    alphas.append(curr_alphas)

            model.eval()
            with torch.no_grad():
                total_val_loss, n, num_correct, num_total = 0, 0, 0, 0
                for batch_idx, (y, targety) in enumerate(loaders['aug_eval']):
                    y, targety = y.to(device), targety.to(device)
                    output = model(y)
                    val_loss = criterion(output, targety)
                    total_val_loss += val_loss
                    n += 1
                    _, prediction = torch.max(output.data, 1)
                    num_correct += torch.sum(prediction == targety.data)
                    num_total += len(prediction)
                epoch_aug_val_loss = total_val_loss / n
                epoch_aug_val_acc = num_correct * 1.0 / num_total

                total_val_loss, n, num_correct, num_total = 0, 0, 0, 0
                for batch_idx, (y, targety) in enumerate(loaders['eval']):
                    y, targety = y.to(device), targety.to(device)
                    output = model(y)
                    val_loss = criterion(output, targety)
                    total_val_loss += val_loss
                    n += 1
                    _, prediction = torch.max(output.data, 1)
                    num_correct += torch.sum(prediction == targety.data)
                    num_total += len(prediction)
                epoch_val_loss = total_val_loss / n
                epoch_val_acc = num_correct * 1.0 / num_total
            lr = 0
            for param_group in optimizer.param_groups:
                lr = param_group['lr']
            print(
                "    Epoch {}: LR {:1.4f} ||| aug_train_acc {:1.4f} | val_acc {:1.4f}, aug {:1.4f} ||| "
                "aug_train_loss {:1.4f} | val_loss {:1.4f}, aug {:1.4f} ||| time = {:1.4f}"
                    .format(epoch, lr, epoch_aug_train_acc, epoch_val_acc, epoch_aug_val_acc,
                            epoch_aug_train_loss, epoch_val_loss, epoch_aug_val_loss, (time.time() - start_time)), flush=True
            )

            if args.hp_idx is None:
                hp_idx = -1
            else:
                hp_idx = args.hp_idx

            epoch_train_loss = 0
            epoch_train_acc = 0
            if epoch == num_epochs:
                with torch.no_grad():
                    total_train_loss, n, num_correct, num_total = 0, 0, 0, 0
                    for batch_idx, (x, targetx) in enumerate(loaders['aug_train']):
                        x, targetx = x.to(device), targetx.to(device)
                        output = model(x)
                        train_loss = criterion(output, targetx)
                        total_train_loss += train_loss
                        n += 1
                        _, prediction = torch.max(output.data, 1)
                        num_correct += torch.sum(prediction == targetx.data)
                        num_total += len(prediction)
                    epoch_aug_train_loss = total_train_loss / n
                    epoch_aug_train_acc = num_correct * 1.0 / num_total

                    total_train_loss, n, num_correct, num_total = 0, 0, 0, 0
                    for batch_idx, (x, targetx) in enumerate(loaders['train']):
                        x, targetx = x.to(device), targetx.to(device)
                        output = model(x)
                        train_loss = criterion(output, targetx)
                        total_train_loss += train_loss
                        n += 1
                        _, prediction = torch.max(output.data, 1)
                        num_correct += torch.sum(prediction == targetx.data)
                        num_total += len(prediction)
                    epoch_train_loss = total_val_loss / n
                    epoch_train_acc = num_correct * 1.0 / num_total

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
                                 'gen_gap': float(epoch_val_loss - epoch_train_loss),
                                 'aug_gen_gap': float(epoch_aug_val_loss - epoch_aug_train_loss),
                                 'resnet_ver': resnet_ver,
                                 'resnet_width': resnet_width,
                                 'epoch_train_loss': float(epoch_train_loss),
                                 'epoch_train_acc': float(epoch_train_acc),
                                 'epoch_aug_train_loss': float(epoch_aug_train_loss),
                                 'epoch_aug_train_acc': float(epoch_aug_train_acc),
                                 'epoch_val_loss': float(epoch_val_loss),
                                 'epoch_val_acc': float(epoch_val_acc),
                                 'epoch_aug_val_loss': float(epoch_aug_val_loss),
                                 'epoch_aug_val_acc': float(epoch_aug_val_acc),
                                 'hp_idx': hp_idx,
                                 'lr_init': lr_init,
                                 'lr_gamma': lr_gamma,
                                 'curr_lr': lr,
                                 'weight_decay': wd,
                                 'alpha': rms_alpha,
                                 'momentum': rms_momentum,
                                 'grid_id': args.grid_id
                                 })

            epoch += 1

            if args.optim == 'rmsprop':
                scheduler.step()

            if args.checkpoints:
                if epoch_val_acc > best_val_acc:
                    best_val_acc = epoch_val_acc
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
                                }, best_checkpoint_location)

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
                            }, final_checkpoint_location)
