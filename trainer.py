import torch
import torch.utils.data
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ExponentialLR
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.optim.lr_scheduler import CyclicLR
from torch.optim.lr_scheduler import OneCycleLR
import torch.nn.functional as F

from apex import amp

import math
from models import mlp
from models import cnn
from models import preact_resnet
import util
import hparams

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
                                           c=num_params,
                                           permute_type=perm_method,
                                           width=resnet_width,
                                           verbose=verbose).to(device)
        model_params = model.parameters()

    return model, model_params


# -------------------- Setting Up & Running Training Function
def train(args, checkpoint, mid_checkpoint_location, final_checkpoint_location, best_checkpoint_location,
          actfun, curr_seed, outfile_path, filename, fieldnames, curr_sample_size, device, num_params,
          curr_k=2, curr_p=1, curr_g=1, perm_method='shuffle', resnet_width=0):
    """
    Runs training session for a given randomized model
    :param args: arguments for this job
    :param checkpoint: current checkpoint
    :param checkpoint_location: output directory for checkpoints
    :param actfun: activation function currently being used
    :param curr_seed: seed being used by current job
    :param outfile_path: path to save outputs from training session
    :param fieldnames: column names for output file
    :param device: reference to CUDA device for GPU support
    :param num_params: number of parameters in the network
    :param curr_k: k value for this iteration
    :param curr_p: p value for this iteration
    :param curr_g: g value for this iteration
    :param perm_method: permutation strategy for our network
    :return:
    """

    resnet_ver = args.resnet_ver
    num_epochs = args.num_epochs

    actfuns_1d = ['relu', 'abs', 'prelu', 'swish', 'leaky_relu', 'tanh']
    if actfun in actfuns_1d:
        curr_k = 1
    kwargs = {'num_workers': 1, 'pin_memory': True} if torch.cuda.is_available() else {}

    curr_hparams = hparams.get_hparams(args.model, args.dataset, actfun, curr_seed,
                                       num_epochs, args.search, args.hp_idx)
    lr = curr_hparams['max_lr']

    criterion = nn.CrossEntropyLoss()
    model, model_params = load_model(args.model, args.dataset, actfun, curr_k, curr_p, curr_g, num_params=num_params,
                               perm_method=perm_method, device=device, resnet_ver=resnet_ver,
                               resnet_width=resnet_width, verbose=args.verbose)

    util.seed_all(curr_seed)
    model.apply(util.weights_init)

    util.seed_all(curr_seed)
    dataset = util.load_dataset(
        args,
        args.model,
        args.dataset,
        seed=curr_seed,
        validation=args.validation,
        batch_size=int(args.batch_size * args.bs_factor),
        train_sample_size=curr_sample_size,
        kwargs=kwargs)
    loaders = {
        'aug_train': dataset[0],
        'train': dataset[1],
        'aug_eval': dataset[2],
        'eval': dataset[3],
    }
    sample_size = dataset[4]
    batch_size = dataset[5]

    optimizer = optim.Adam(model_params,
                           betas=(curr_hparams['beta1'], curr_hparams['beta2']),
                           eps=curr_hparams['eps'],
                           weight_decay=curr_hparams['wd'])
    scheduler = OneCycleLR(optimizer,
                           max_lr=curr_hparams['max_lr'] * args.bs_factor * args.lr_factor,
                           epochs=num_epochs,
                           steps_per_epoch=int(math.floor(sample_size / batch_size)),
                           pct_start=curr_hparams['cycle_peak'],
                           cycle_momentum=False)

    if args.mix_pre_apex:
        model, optimizer = amp.initialize(model, optimizer, opt_level="O2")

    if args.distributed:
        pmodel = nn.DataParallel(model)
    else:
        pmodel = model
    pmodel = pmodel.cuda()

    epoch = 1
    seen_actfuns = set()
    is_preempted = False
    if checkpoint is not None:
        is_preempted = True
        if actfun not in checkpoint['seen_actfuns']:
            seen_actfuns = checkpoint['seen_actfuns']
        else:
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            scheduler.load_state_dict(checkpoint['scheduler'])
            if args.mix_pre_apex:
                amp.load_state_dict(checkpoint["amp"])
            epoch = checkpoint['epoch']
            seen_actfuns = checkpoint['seen_actfuns']
            model.to(device)
            print("*** LOADED CHECKPOINT ***"
                  "\n{}"
                  "\nSeed: {}"
                  "\nEpoch: {}"
                  "\nActfun: {}".format(mid_checkpoint_location, checkpoint['curr_seed'],
                                             checkpoint['epoch'], checkpoint['actfun']))
    seen_actfuns.add(actfun)

    util.print_exp_settings(curr_seed, args.dataset, outfile_path, args.model, actfun,
                            util.get_model_params(model), sample_size, batch_size, model.k, model.p, model.g,
                            perm_method, resnet_ver, resnet_width, args.optim, args.validation, curr_hparams)

    best_val_acc = 0

    # ---- Start Training
    while epoch <= num_epochs:

        if args.check_path != '':
            temp_path = os.path.join(args.check_path, "temp.pth")
            if args.mix_pre_apex:
                mix_pre_state = amp.state_dict()
            else:
                mix_pre_state = None
            torch.save({'state_dict': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'scheduler': scheduler.state_dict(),
                        'amp':mix_pre_state,
                        'curr_seed': curr_seed,
                        'epoch': epoch,
                        'actfun': actfun,
                        'seen_actfuns': seen_actfuns
                        }, temp_path)
            os.replace(temp_path, mid_checkpoint_location)

        util.seed_all((curr_seed * args.num_epochs) + epoch)
        start_time = time.time()
        if args.mix_pre:
            scaler = torch.cuda.amp.GradScaler()

        # ---- Training
        model.train()
        total_train_loss, n, num_correct, num_total = 0, 0, 0, 0
        for batch_idx, (x, targetx) in enumerate(loaders['aug_train']):
            x, targetx = x.to(device), targetx.to(device)
            if not args.split_batch:
                optimizer.zero_grad()
            if args.mix_pre:
                with torch.cuda.amp.autocast():
                    output = pmodel(x)
                    train_loss = criterion(output, targetx)
                total_train_loss += train_loss
                n += 1
                scaler.scale(train_loss).backward()
                scaler.step(optimizer)
                scaler.update()
            elif args.mix_pre_apex:
                output = pmodel(x)
                train_loss = criterion(output, targetx)
                total_train_loss += train_loss
                n += 1
                with amp.scale_loss(train_loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
                if args.split_batch and (batch_idx+1)%8==0:
                    train_loss = train_loss/8
                    optimizer.step()
                    optimizer.zero_grad()
                elif not args.split_batch:
                    optimizer.step()

            else:
                output = pmodel(x)
                train_loss = criterion(output, targetx)
                total_train_loss += train_loss
                n += 1
                train_loss.backward()
                optimizer.step()
            scheduler.step()
            _, prediction = torch.max(output.data, 1)
            num_correct += torch.sum(prediction == targetx.data)
            num_total += len(prediction)
            if batch_idx <= 2:
                print('    Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(x), len(loaders['aug_train'].dataset),
                    100. * batch_idx / len(loaders['aug_train']), train_loss.item()), flush=True
                    )
        epoch_aug_train_loss = total_train_loss / n
        epoch_aug_train_acc = num_correct * 1.0 / num_total

        model.eval()
        with torch.no_grad():
            total_val_loss, n, num_correct, num_total = 0, 0, 0, 0
            for batch_idx, (y, targety) in enumerate(loaders['aug_eval']):
                y, targety = y.to(device), targety.to(device)
                output = pmodel(y)
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
                output = pmodel(y)
                val_loss = criterion(output, targety)
                total_val_loss += val_loss
                n += 1
                _, prediction = torch.max(output.data, 1)
                num_correct += torch.sum(prediction == targety.data)
                num_total += len(prediction)
            epoch_val_loss = total_val_loss / n
            epoch_val_acc = num_correct * 1.0 / num_total
        lr_curr = 0
        for param_group in optimizer.param_groups:
            lr_curr = param_group['lr']
        print(
            "    Epoch {}: LR {:1.5f} ||| aug_train_acc {:1.4f} | val_acc {:1.4f}, aug {:1.4f} ||| "
            "aug_train_loss {:1.4f} | val_loss {:1.4f}, aug {:1.4f} ||| time = {:1.4f}, is_preempted = {}"
                .format(epoch, lr_curr, epoch_aug_train_acc, epoch_val_acc, epoch_aug_val_acc,
                        epoch_aug_train_loss, epoch_val_loss, epoch_aug_val_loss,
                         (time.time() - start_time), is_preempted), flush=True
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
                    output = pmodel(x)
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
                    output = pmodel(x)
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
                             'model': args.model,
                             'batch_size': batch_size,
                             'num_params': util.get_model_params(model),
                             'k': curr_k,
                             'p': curr_p,
                             'g': curr_g,
                             'perm_method': perm_method,
                             'resnet_ver': resnet_ver,
                             'resnet_type': args.resnet_type,
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
                             'curr_lr': lr_curr,
                             'hparams': curr_hparams,
                             'epochs': num_epochs,
                             'is_preempted': is_preempted
                             })

        epoch += 1
