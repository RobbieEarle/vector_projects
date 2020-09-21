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

    # ---- Initialization

    model_params = []
    if args.model == 'nn' or args.model == 'mlp':
        if args.dataset == 'mnist' or args.dataset == 'fashion_mnist':
            input_dim, output_dim = 784, 10
        elif args.dataset == 'cifar10' or args.dataset == 'svhn':
            input_dim, output_dim = 3072, 10
        elif args.dataset == 'cifar100':
            input_dim, output_dim = 3072, 100
        model = models.CombinactMLP(actfun=actfun, input_dim=input_dim, output_dim=output_dim,
                                    k=curr_k, p=curr_p, g=curr_g, reduce_actfuns=args.reduce_actfuns,
                                    num_params=num_params, permute_type=perm_method).to(device)

    elif args.model == 'cnn':
        if args.dataset == 'mnist' or args.dataset == 'fashion_mnist':
            input_channels, input_dim, output_dim = 1, 28, 10
        elif args.dataset == 'cifar10' or args.dataset == 'svhn':
            input_channels, input_dim, output_dim = 3, 32, 10
        elif args.dataset == 'cifar100':
            input_channels, input_dim, output_dim = 3, 32, 100

        model = models.CombinactCNN(actfun=actfun, num_input_channels=input_channels, input_dim=input_dim,
                                    num_outputs=output_dim, k=curr_k, p=curr_p, g=curr_g, num_params=num_params,
                                    reduce_actfuns=args.reduce_actfuns, permute_type=perm_method).to(device)

        model_params.append({'params': model.conv_layers.parameters()})
        model_params.append({'params': model.pooling.parameters()})

    elif args.model == 'resnet':
        if args.dataset == 'mnist' or args.dataset == 'fashion_mnist':
            input_channels, input_dim, output_dim = 1, 28, 10
        elif args.dataset == 'cifar10' or args.dataset == 'svhn':
            input_channels, input_dim, output_dim = 3, 32, 10
        elif args.dataset == 'cifar100':
            input_channels, input_dim, output_dim = 3, 32, 100

        model = models.ResNet(resnet_ver=args.resnet_ver, actfun=actfun,
                              num_input_channels=input_channels, num_outputs=output_dim, k=curr_k, p=curr_p, g=curr_g, reduce_actfuns=args.reduce_actfuns,
                              permute_type=perm_method, width=args.resnet_width, orig=args.resnet_orig).to(device)

        if not args.resnet_orig:
            model_params.append({'params': model.conv_layers.parameters()})
        else:
            model_params = model.parameters()

    if not args.resnet_orig:
        model_params.append({'params': model.batch_norms.parameters(), 'weight_decay': 0})
        model_params.append({'params': model.linear_layers.parameters()})
    if actfun == 'combinact':
        model_params.append({'params': model.all_alpha_primes.parameters(), 'weight_decay': 0})

    util.seed_all(curr_seed)
    rng = np.random.RandomState(curr_seed)
    model.apply(util.weights_init)

    hyper_params = hp.get_hyper_params(args.model, args.dataset, actfun, rng=rng, exp=args.hyper_params, p=curr_p)

    criterion = nn.CrossEntropyLoss()

    num_epochs = args.num_epochs
    if args.overfit:
        num_epochs = 50
        hyper_params['cycle_peak'] = 0.35

    hyper_params['adam_wd'] *= args.wd

    if not args.resnet_orig:
        optimizer = optim.Adam(model_params,
                               lr=10 ** -8,
                               betas=(hyper_params['adam_beta_1'], hyper_params['adam_beta_2']),
                               eps=hyper_params['adam_eps'],
                               weight_decay=hyper_params['adam_wd']
                               )
    else:
        optimizer = optim.Adam(model.parameters(), lr=0.000001, betas=(0.9, 0.99), weight_decay=5e-4)

    if args.resnet_orig:
        scheduler = OneCycleLR(optimizer,
                               max_lr=0.1,
                               epochs=num_epochs,
                               steps_per_epoch=int(math.ceil(sample_size / batch_size)),
                               pct_start=0.2,
                               cycle_momentum=False
                               )
    elif args.model == 'resnet':
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
        resnet_ver = None
        resnet_width = None

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

    if not args.resnet_orig:
        k_print = model.k
        p_print = model.p
        g_print = model.g
    else:
        k_print = 1
        p_print = 1
        g_print = 1
    util.print_exp_settings(curr_seed, args.dataset, outfile_path, args.model, actfun, hyper_params,
                            util.get_model_params(model), sample_size, k_print, p_print, g_print,
                            perm_method, resnet_ver, resnet_width, args.resnet_orig)

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
            "    Epoch {} Completed: gen_gap = {:1.6f}  |  train_loss = {:1.6f}  |  val_loss = {:1.6f}  |  accuracy = {:1.6f}  |  time = {}"
                .format(epoch, final_val_loss - final_train_loss, final_train_loss, final_val_loss, accuracy, (time.time() - start_time)), flush=True
        )

        alpha_primes = []
        alphas = []

        if not args.resnet_orig:
            for i, layer_alpha_primes in enumerate(model.all_alpha_primes):
                curr_alpha_primes = torch.mean(layer_alpha_primes, dim=0)
                curr_alphas = F.softmax(curr_alpha_primes, dim=0).data.tolist()
                curr_alpha_primes = curr_alpha_primes.tolist()
                alpha_primes.append(curr_alpha_primes)
                alphas.append(curr_alphas)

        eval_train_loss = 0
        eval_val_loss = 0
        eval_train_acc = 0
        eval_val_acc = 0
        if epoch == num_epochs:
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

        if not args.resnet_orig:
            print_actfun = args.actfun
        else:
            print_actfun = model.actfun

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
                             'actfun': print_actfun,
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
                             'gen_gap': float(final_val_loss - final_train_loss),
                             'resnet_ver': resnet_ver,
                             'resnet_width': resnet_width,
                             'eval_train_loss': float(eval_train_loss),
                             'eval_val_loss': float(eval_val_loss),
                             'eval_train_acc': float(eval_train_acc),
                             'eval_val_acc': float(eval_val_acc)
                             })

        epoch += 1
