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

    all_actfuns = util.get_actfuns(args.actfun)
    param_factors, param_factors_1d = util.get_param_factors(args)
    train_samples = util.get_train_samples(args)
    p_vals, k_vals = util.get_pk_vals(args)

    if args.var_perm_method:
        perm_methods = ['shuffle', 'roll', 'roll_grouped']
    else:
        perm_methods = ['shuffle']

    # =========================== Creating new output file
    fieldnames = ['dataset', 'seed', 'epoch', 'train_loss', 'val_loss', 'acc', 'time', 'actfun',
                  'sample_size', 'hyper_params', 'model', 'batch_size', 'alpha_primes', 'alphas',
                  'num_params', 'var_nparams', 'var_nsamples', 'k', 'p', 'perm_method']
    checkpoint_location = os.path.join(args.check_path, "cp_{}.pth".format(args.seed))
    checkpoint = None

    # =========================== Loading checkpoint
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

    # =========================== Training
    for actfun in all_actfuns:

        curr_param_factors = param_factors
        curr_p_vals = p_vals
        curr_k_vals = k_vals
        if actfun == 'relu' or actfun == 'abs':
            curr_param_factors = param_factors_1d
            curr_k_vals = [1]

        curr_seed = (args.seed * len(param_factors) * len(train_samples))
        if checkpoint is not None:
            curr_param_factors = curr_param_factors[curr_param_factors.index(cp_param_factor):]
            curr_seed = cp_seed

        for pfact in curr_param_factors:

            curr_train_samples = train_samples
            if checkpoint is not None:
                curr_train_samples = curr_train_samples[curr_train_samples.index(cp_train_sample):]

            for curr_sample_size in curr_train_samples:

                for p in curr_p_vals:

                    if args.model == 'nn':
                        curr_pfact_p = pfact - (0.3 * (p - 1)) ** (3 / p)
                        if p != 1:
                            curr_pfact_p = curr_pfact_p - 0.3
                    elif args.model == 'cnn':
                        curr_pfact_p = pfact - (0.001 * (p - 1)) ** (3 / (5*p))
                        if p == 5:
                            curr_pfact_p = curr_pfact_p - 0.155
                        elif p == 4:
                            curr_pfact_p = curr_pfact_p - 0.19
                        elif p != 1:
                            curr_pfact_p = curr_pfact_p - 0.23

                    for k in curr_k_vals:

                        if args.model == 'nn':
                            curr_pfact_pk = curr_pfact_p + ((k - 2) * 0.15) ** (2 / k)
                        elif args.model == 'cnn':
                            curr_pfact_pk = curr_pfact_p + ((k - 2) * 0.2) ** (3 / k)
                            if k == 3:
                                curr_pfact_pk += 0.1

                        for perm_method in perm_methods:

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
                                          train_loader, validation_loader, sample_size, batch_size, device,
                                          pfact=curr_pfact_pk, curr_p=p, curr_k=k, perm_method=perm_method)
                            print()

                            curr_seed += 1
                            checkpoint = None


# --------------------  Entry Point
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Higher order activation function testing')
    parser.add_argument('--seed', type=int, default=1, help='Job seed')
    parser.add_argument('--dataset', type=str, default='mnist', help='Dataset being used. mnist or cifar10')
    parser.add_argument('--model', type=str, default='nn', help='What type of model to use')
    parser.add_argument('--actfun', type=str, default='relu')
    parser.add_argument('--save_path', type=str, default='', help='Where to save results')
    parser.add_argument('--check_path', type=str, default='', help='Where to save checkpoints')
    parser.add_argument('--sample_size', type=int, default=None, help='Training sample size')
    parser.add_argument('--batch_size', type=int, default=None, help='Batch size during training')
    parser.add_argument('--num_epochs', type=int, default=10, help='Number of training epochs')
    parser.add_argument('--var_n_params', action='store_true', help='When true, varies number of network parameters')
    parser.add_argument('--var_n_samples', action='store_true', help='When true, varies number of training samples')
    parser.add_argument('--reduce_actfuns', action='store_true', help='When true, does not use extra actfuns')
    parser.add_argument('--var_k', action='store_true', help='When true, varies k hyper-param')
    parser.add_argument('--var_p', action='store_true', help='When true, varies k hyper-param')
    parser.add_argument('--var_perm_method', action='store_true', help='When true, varies permutation method')
    parser.add_argument('--overfit', action='store_false', help='When true, causes model to overfit')
    args = parser.parse_args()

    extras = util.get_extras(args)

    out = os.path.join(
        args.save_path,
        '{}-{}-{}-{}-{}{}.csv'.format(
            datetime.date.today(),
            args.seed,
            args.dataset,
            args.model,
            args.actfun,
            extras
        )
    )

    setup_experiment(args, out)
