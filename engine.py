import torch
import torch.utils.data
import util
import argparse
import os
import datetime
import csv
import trainer
import trainer_dawnnet


def retrieve_checkpoint(curr_entry, full_arr):
    if curr_entry in full_arr:
        return full_arr[full_arr.index(curr_entry):]
    else:
        return full_arr


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

    # =========================== Creating new output file
    fieldnames = ['dataset', 'seed', 'epoch', 'time', 'actfun',
                  'sample_size', 'hyper_params', 'model', 'batch_size', 'alpha_primes', 'alphas',
                  'num_params', 'var_nparams', 'var_nsamples', 'k', 'p', 'g', 'perm_method',
                  'gen_gap', 'resnet_ver', 'resnet_width', 'train_loss', 'val_loss',
                  'train_acc', 'val_acc']

    checkpoint_location = os.path.join(args.check_path, "cp_{}_{}_{}.pth".format(args.seed, args.model, args.dataset))
    checkpoint = None

    all_actfuns = util.get_actfuns(args.actfun)
    if os.path.exists(checkpoint_location):
        checkpoint = torch.load(checkpoint_location)
        all_actfuns = retrieve_checkpoint(checkpoint['actfun'], all_actfuns)
    else:
        with open(outfile_path, mode='w') as out_file:
            writer = csv.DictWriter(out_file, fieldnames=fieldnames, lineterminator='\n')
            writer.writeheader()

    # =========================== Training
    for actfun in all_actfuns:

        num_params = util.get_num_params(args, actfun)
        train_samples = util.get_train_samples(args)
        p_vals, k_vals, g_vals = util.get_pkg_vals(args)
        perm_methods = util.get_perm_methods(args.var_perm_method)
        curr_seed = (args.seed * len(num_params) * len(train_samples) * len(p_vals) * len(k_vals) * len(
            g_vals) * len(perm_methods))
        if checkpoint is not None:
            num_params = retrieve_checkpoint(checkpoint['num_params'], num_params)
            if train_samples[0] is not None:
                train_samples = retrieve_checkpoint(checkpoint['sample_size'], train_samples)
            p_vals = retrieve_checkpoint(checkpoint['p'], p_vals)
            k_vals = retrieve_checkpoint(checkpoint['k'], k_vals)
            perm_methods = retrieve_checkpoint(checkpoint['perm_method'], perm_methods)
            curr_seed = checkpoint['curr_seed']

        for curr_num_params in num_params:
            for curr_sample_size in train_samples:
                for p in p_vals:
                    for k in k_vals:
                        for g in g_vals:

                            if args.var_pg:
                                g = p

                            for perm_method in perm_methods:

                                util.seed_all(curr_seed)

                                # ---- Loading Dataset
                                print()
                                train_loader, validation_loader, sample_size, batch_size = util.load_dataset(args.model,
                                                                                                             args.dataset,
                                                                                                             seed=curr_seed,
                                                                                                             batch_size=args.batch_size,
                                                                                                             sample_size=curr_sample_size,
                                                                                                             kwargs=kwargs)

                                if args.model == 'dawnnet':
                                    trainer_dawnnet.train(args, checkpoint, checkpoint_location, actfun, curr_seed,
                                                  outfile_path,
                                                  fieldnames, train_loader, validation_loader, sample_size, batch_size,
                                                  device, num_params=curr_num_params, curr_p=p, curr_k=k, curr_g=g,
                                                  perm_method=perm_method)

                                # ---- Begin training model
                                else:
                                    trainer.train(args, checkpoint, checkpoint_location, actfun, curr_seed, outfile_path,
                                                  fieldnames, train_loader, validation_loader, sample_size, batch_size,
                                                  device, num_params=curr_num_params, curr_p=p, curr_k=k, curr_g=g,
                                                  perm_method=perm_method)
                                print()

                                checkpoint = None
                                curr_seed += 1


# --------------------  Entry Point
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Higher order activation function testing')
    parser.add_argument('--seed', type=int, default=1, help='Job seed')
    parser.add_argument('--p', type=int, default=1, help='Default p value for model')
    parser.add_argument('--k', type=int, default=2, help='Default k value for model')
    parser.add_argument('--g', type=int, default=1, help='Default g value for model')
    parser.add_argument('--num_params', type=int, default=0, help='Adjust number of model params')
    parser.add_argument('--resnet_ver', type=int, default=34, help='Which version of ResNet to use')
    parser.add_argument('--resnet_width', type=int, default=1, help='How wide to make our ResNet layers')
    parser.add_argument('--model', type=str, default='resnet', help='cnn, mlp, resnet')  # cnn
    parser.add_argument('--dataset', type=str, default='cifar10', help='mnist, cifar10, cifar100')  # mnist
    parser.add_argument('--actfun', type=str, default='relu')  # all
    parser.add_argument('--save_path', type=str, default='', help='Where to save results')
    parser.add_argument('--check_path', type=str, default='', help='Where to save checkpoints')
    parser.add_argument('--sample_size', type=int, default=None, help='Training sample size')
    parser.add_argument('--batch_size', type=int, default=None, help='Batch size during training')
    parser.add_argument('--num_epochs', type=int, default=35, help='Number of training epochs')
    parser.add_argument('--wd', type=float, default=1, help='Weight decay multiplier')
    parser.add_argument('--hyper_params', type=str, default='', help='Which hyper param settings to use')
    parser.add_argument('--resnet_orig', action='store_true', help='')

    parser.add_argument('--var_n_params', action='store_true', help='When true, varies number of network parameters')
    parser.add_argument('--var_n_params_log', action='store_true', help='Varies number of network params on log scale')
    parser.add_argument('--var_n_samples', action='store_true', help='When true, varies number of training samples')
    parser.add_argument('--reduce_actfuns', action='store_true', help='When true, does not use extra actfuns')
    parser.add_argument('--var_p', action='store_true', help='When true, varies p hyper-param')
    parser.add_argument('--var_k', action='store_true', help='When true, varies k hyper-param')
    parser.add_argument('--var_g', action='store_true', help='When true, varies g hyper-param')
    parser.add_argument('--var_pg', action='store_true', help='When true, varies p and g hyper-params')
    parser.add_argument('--var_perm_method', action='store_true', help='When true, varies permutation method')
    parser.add_argument('--overfit', action='store_true', help='When true, causes model to overfit')
    parser.add_argument('--p_param_eff', action='store_true', help='When true, varies p and number params')

    parser.add_argument('--bin_redo', action='store_true', help='')
    parser.add_argument('--bin_peff_redo', action='store_true', help='')
    parser.add_argument('--nparam_redo', action='store_true', help='')
    args = parser.parse_args()

    extras = util.get_extras(args)

    if args.model == 'resnet':
        model = "{}-{}-{}".format(args.model, args.resnet_ver, args.resnet_width)
    else:
        model = args.model

    out = os.path.join(
        args.save_path,
        '{}-{}-{}-{}_{}{}.csv'.format(
            datetime.date.today(),
            args.seed,
            args.dataset,
            model,
            args.actfun,
            extras
        )
    )

    setup_experiment(args, out)
