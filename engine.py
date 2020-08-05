import torch
import torch.utils.data
import util
import argparse
import os
import datetime
import csv
import trainer


def setup_experiment(args, outfile_path):
    """
    Retrieves training / validation data, randomizes network structure and activation functions, creates model,
    creates new output file, sets hyperparameters for optimizer and scheduler during training, initializes training
    :param seed: seed for parameter randomization
    :param outfile_path: path to save outputs from experiment
    :param actfun: model architecture
    :return:
    """

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

    if args.actfun == 'all':
        all_actfuns = ['relu', 'abs', 'combinact', 'max', 'min', 'lse', 'lae', 'l2', 'linf', 'prod', 'signed_geomean',
                       'swishk', 'binary_ops_partition', 'binary_ops_all']
    elif args.actfun == 'new_only':
        all_actfuns = ['min', 'lse', 'lae', 'linf', 'prod', 'signed_geomean', 'swishk', 'binary_ops_partition',
                       'binary_ops_all']
    elif args.actfun == '1d':
        all_actfuns = ['relu', 'abs']
    else:
        all_actfuns = [args.actfun]

    if args.model == 'nn':
        if args.dataset == 'mnist' or args.dataset == 'fashion_mnist':
            if args.var_n_params:
                param_factors = [3.5, 3.23, 2.95, 2.65, 2.34, 2.01, 1.665]
                param_factors_1d = [1.45, 1.345, 1.24, 1.125, 1, 0.875, 0.735]
            else:
                param_factors = [1.665]
                param_factors_1d = [1.45]
            if args.actfun == 'binary_ops_partition':
                for i in range(len(param_factors)):
                    param_factors[i] *= 0.94
            elif args.actfun == 'binary_ops_all':
                for i in range(len(param_factors)):
                    if i >= 4:
                        param_factors[i] *= 0.71
                    else:
                        param_factors[i] *= 0.68

        else:
            if args.var_n_params:
                param_factors = [1.25, 1.13, 1.01, 0.885, 0.76, 0.635, 0.515]
                param_factors_1d = [0.6, 0.545, 0.49, 0.43, 0.375, 0.31, 0.255]
            else:
                param_factors = [1.25]
                param_factors_1d = [0.6]
            if args.actfun == 'binary_ops_partition':
                for i in range(len(param_factors)):
                    param_factors[i] *= 0.99
            elif args.actfun == 'binary_ops_all':
                for i in range(len(param_factors)):
                    param_factors[i] *= 0.92

    if args.model == 'cnn':
        if args.var_n_params:
            param_factors = [1.01, 0.925, 0.83, 0.72, 0.59, 0.41, 0.29]
            param_factors_1d = [0.36, 0.33, 0.295, 0.255, 0.21, 0.1475, 0.105]
        else:
            param_factors = [1.01]
            param_factors_1d = [0.36]

        if args.dataset == 'mnist' or args.dataset == 'fashion_mnist':
            for i in range(len(param_factors)):
                param_factors[i] *= 1.21
            for i in range(len(param_factors_1d)):
                param_factors_1d[i] *= 1.21

        if args.actfun == 'binary_ops_partition':
            for i in range(len(param_factors)):
                param_factors[i] *= 0.865
        elif args.actfun == 'binary_ops_all':
            for i in range(len(param_factors)):
                param_factors[i] *= 0.5

    if args.var_n_samples:
        train_samples = [50000, 45000, 40000, 35000, 30000, 25000, 20000, 15000, 10000, 5000]
    else:
        train_samples = [args.sample_size]

    # ---- Create new output file
    fieldnames = ['dataset', 'seed', 'epoch', 'train_loss', 'val_loss', 'acc', 'time', 'actfun',
                  'sample_size', 'hyper_params', 'model', 'batch_size', 'alpha_primes', 'alphas',
                  'num_params', 'var_nparams', 'var_nsamples']
    checkpoint_location = os.path.join(args.check_path, "cp_{}.pth".format(args.seed))
    checkpoint = None

    if os.path.exists(checkpoint_location):
        checkpoint = torch.load(checkpoint_location)
        all_actfuns = all_actfuns[all_actfuns.index(checkpoint['actfun']):]
        cp_param_factor = checkpoint['param_factor']
        cp_train_sample = checkpoint['train_sample']
        cp_seed = checkpoint['curr_seed']
    else:
        with open(outfile_path, mode='w') as out_file:
            writer = csv.DictWriter(out_file, fieldnames=fieldnames, lineterminator='\n')
            writer.writeheader()

    for actfun in all_actfuns:

        if actfun == 'relu' or actfun == 'abs':
            curr_param_factors = param_factors_1d
        else:
            curr_param_factors = param_factors

        curr_seed = (args.seed * len(param_factors) * len(train_samples))

        if checkpoint is not None:
            curr_param_factors = curr_param_factors[curr_param_factors.index(cp_param_factor):]
            curr_seed = cp_seed

        for i, pfact in enumerate(curr_param_factors):

            curr_train_samples = train_samples
            if checkpoint is not None:
                curr_train_samples = curr_train_samples[curr_train_samples.index(cp_train_sample):]

            for j, curr_sample_size in enumerate(curr_train_samples):
                # ---- Loading Dataset
                print()
                train_loader, validation_loader, sample_size, batch_size = util.load_dataset(args.dataset,
                                                                                             seed=curr_seed,
                                                                                             batch_size=args.batch_size,
                                                                                             sample_size=curr_sample_size,
                                                                                             kwargs=kwargs)
                # ---- Begin training model
                util.seed_all(curr_seed)
                trainer.train(args, actfun, curr_seed, outfile_path, checkpoint, fieldnames,
                              train_loader, validation_loader, sample_size, batch_size, device, pfact=pfact)
                print()

                curr_seed += 1
                checkpoint = None


# --------------------  Entry Point
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Higher order activation function testing')
    parser.add_argument('--seed', type=int, default=1, help='Job seed')
    parser.add_argument('--dataset', type=str, default='mnist', help='Dataset being used. mnist or cifar10')
    parser.add_argument('--model', type=str, default='nn', help='What type of model to use')
    parser.add_argument('--actfun', type=str, default='all')
    parser.add_argument('--save_path', type=str, default='', help='Where to save results')
    parser.add_argument('--check_path', type=str, default='', help='Where to save checkpoints')
    parser.add_argument('--sample_size', type=int, default=None, help='Training sample size')
    parser.add_argument('--batch_size', type=int, default=None, help='Batch size during training')
    parser.add_argument('--num_epochs', type=int, default=10, help='Number of training epochs')
    parser.add_argument('--var_n_params', action='store_true', help='When true, varies number of network parameters')
    parser.add_argument('--var_n_samples', action='store_true', help='When true, varies number of training samples')
    parser.add_argument('--reduce_actfuns', action='store_true', help='When true, does not use extra actfuns')
    args = parser.parse_args()

    out = os.path.join(
        args.save_path,
        '{}-{}-{}-{}-{}.csv'.format(
            datetime.date.today(),
            args.seed,
            args.dataset,
            args.model,
            args.actfun
        )
    )

    setup_experiment(args, out)
