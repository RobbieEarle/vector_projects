import torch
import torch.utils.data
import util
import argparse
import os
import datetime
import csv
import trainer

_WIDTH_FACTORS = {
    "0.5": {
        "3/2": 0.4,
        "1": 0.5,
        "1/2": 0.75
    },
    "1": {
        "3/2": 0.75,
        "1": 1,
        "1/2": 1.5
    },
    "2": {
        "3/2": 1.5,
        "1": 2,
        "1/2": 3
    },
    "4": {
        "3/2": 3,
        "1": 4,
        "1/2": 6
    }
}


def retrieve_checkpoint(curr_entry, full_arr):
    if curr_entry in full_arr:
        return full_arr[full_arr.index(curr_entry):]
    else:
        return full_arr


def setup_experiment(args):
    """
    Retrieves training / validation data, randomizes network structure and activation functions, creates model,
    creates new output file, sets hyperparameters for optimizer and scheduler during training, initializes training
    :param args: args passed in for current experiment
    :param outfile_path: path to save outputs from experiment
    :return:
    """

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    torch.cuda.set_device(device)

    skip_actfun = False
    if args.actfun_idx is not None:
        all_actfuns = ['max', 'relu', 'swish', 'bin_all_max_min', 'ail_or',
                       'ail_xnor', 'ail_all_or_and', 'ail_all_or_xnor',
                       'ail_all_or_and_xnor', 'ail_part_or_xnor',
                       'ail_part_or_and_xnor']
        actfun = all_actfuns[args.actfun_idx]

        if not args.balanced:
            resnet_width = _WIDTH_FACTORS[args.resnet_type]["1"]
        else:
            if actfun in ['relu', 'swish', 'bin_all_max_min', 'ail_all_or_and', 'ail_all_or_xnor']:
                resnet_width = _WIDTH_FACTORS[args.resnet_type]["1"]
                skip_actfun = True
            elif actfun in ['ail_all_or_and_xnor']:
                resnet_width = _WIDTH_FACTORS[args.resnet_type]["3/2"]
            else:
                resnet_width = _WIDTH_FACTORS[args.resnet_type]["1/2"]
    else:
        actfun = args.actfun
        resnet_width = args.resnet_width

    if not (args.skip_actfuns and skip_actfun):
        # =========================== Creating new output file
        if args.one_shot and args.search:
            fieldnames = ['hp_idx', 'hyperparam_set', 'seed', 'lr', 'loss']
        else:
            fieldnames = ['dataset', 'seed', 'epoch', 'time', 'actfun',
                          'sample_size', 'model', 'batch_size', 'alpha_primes', 'alphas',
                          'num_params', 'var_nparams', 'var_nsamples', 'k', 'p', 'g', 'perm_method',
                          'gen_gap', 'aug_gen_gap', 'resnet_ver', 'resnet_type', 'resnet_width',
                          'epoch_train_loss', 'epoch_train_acc', 'epoch_aug_train_loss',
                          'epoch_aug_train_acc', 'epoch_val_loss', 'epoch_val_acc', 'epoch_aug_val_loss',
                          'epoch_aug_val_acc', 'hp_idx', 'curr_lr', 'found_lr', 'hparams', 'epochs',
                          'is_preempted']

        if args.model == 'resnet':
            model = "{}-{}-{}".format(args.model, args.resnet_ver, args.resnet_width)
        else:
            model = args.model
        filename = '{}{}'.format(args.seed, args.label)

        outfile_path = os.path.join(args.save_path, filename) + '.csv'
        mid_checkpoint_path = os.path.join(args.check_path, "checkpoint_latest") + '.pth'
        checkpoint = None
        if os.path.exists(mid_checkpoint_path):
            checkpoint = torch.load(mid_checkpoint_path)
            checkpoint_valid = (actfun == checkpoint['actfun'] or actfun not in checkpoint['seen_actfuns'])
            if args.c is not None:
                checkpoint_valid = checkpoint_valid and checkpoint['c'] <= args.c

        if checkpoint is None or checkpoint_valid:
            if not os.path.exists(outfile_path):
                with open(outfile_path, mode='w') as out_file:
                    writer = csv.DictWriter(out_file, fieldnames=fieldnames, lineterminator='\n')
                    writer.writeheader()

            # =========================== Training
            num_params = util.get_num_params(args)
            train_samples = util.get_train_samples(args)
            p_vals, k_vals, g_vals = util.get_pkg_vals(args)
            perm_methods = util.get_perm_methods(args)
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

                                    filename = '{}-{}-{}-{}-{}-{}-{}-{}{}'.format(args.seed,
                                                                                  args.dataset,
                                                                                  model,
                                                                                  actfun,
                                                                                  p, k, g, perm_method,
                                                                                  args.label
                                                                                  )
                                    final_checkpoint_path = os.path.join(args.save_path, filename) + '_final.pth'
                                    best_checkpoint_path = os.path.join(args.save_path, filename) + '_best.pth'

                                    # ---- Begin training model
                                    trainer.train(args,
                                                  checkpoint,
                                                  mid_checkpoint_path,
                                                  final_checkpoint_path,
                                                  best_checkpoint_path,
                                                  actfun,
                                                  curr_seed,
                                                  outfile_path,
                                                  filename,
                                                  fieldnames,
                                                  curr_sample_size,
                                                  device,
                                                  num_params=curr_num_params,
                                                  curr_p=p,
                                                  curr_k=k,
                                                  curr_g=g,
                                                  perm_method=perm_method,
                                                  resnet_width=resnet_width)
                                    print()

                                    checkpoint = None
                                    curr_seed += 1


# --------------------  Entry Point
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Higher order activation function testing')
    parser.add_argument('--seed', type=int, default=0, help='Job seed')
    parser.add_argument('--p', type=int, default=1, help='Default p value for model')
    parser.add_argument('--k', type=int, default=2, help='Default k value for model')
    parser.add_argument('--g', type=int, default=1, help='Default g value for model')
    parser.add_argument('--c', type=int, default=None, help='Default c value for model')
    parser.add_argument('--num_params', type=int, default=0, help='Adjust number of model params')
    parser.add_argument('--resnet_ver', type=int, default=34, help='Which version of ResNet to use')
    parser.add_argument('--resnet_type', type=str, default='wrn50_custom', help='Resnet type')
    parser.add_argument('--resnet_width', type=float, default=2, help='How wide to make our ResNet layers')
    parser.add_argument('--model', type=str, default='mlp', help='cnn, mlp, resnet')  # cnn
    parser.add_argument('--dataset', type=str, default='cifar100', help='mnist, cifar10, cifar100')  # mnist
    parser.add_argument('--actfun', type=str, default='max')
    parser.add_argument('--actfun_idx', type=int, default=None)
    parser.add_argument('--optim', type=str, default='onecycle')  # all
    parser.add_argument('--save_path', type=str, default='', help='Where to save results')
    parser.add_argument('--check_path', type=str, default='', help='Where to save checkpoints')
    parser.add_argument('--sample_size', type=int, default=None, help='Training sample size')
    parser.add_argument('--batch_size', type=int, default=None, help='Batch size during training')
    parser.add_argument('--num_epochs', type=int, default=10, help='Number of training epochs')
    parser.add_argument('--wd', type=float, default=1, help='Weight decay multiplier')
    parser.add_argument('--hyper_params', type=str, default='', help='Which hyper param settings to use')
    parser.add_argument('--perm_method', type=str, default='shuffle', help='Which permuation method to use')  # shuffle
    parser.add_argument('--label', type=str, default='', help='Label to differentiate different jobs')
    parser.add_argument('--validation', action='store_true', help='When true, uses validation set instead of test set')
    parser.add_argument('--lr_init', type=float, default=1e-4, help='Initial learning rate value')
    parser.add_argument('--lr_gamma', type=float, default=0.95, help='Weight decay multiplier')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='Weight decay')
    parser.add_argument('--max_lr', type=float, default=1e-5, help='Maximum LR during one cycle schedule')
    parser.add_argument('--momentum', type=float, default=0, help='Maximum LR during one cycle schedule')
    parser.add_argument('--checkpoints', action='store_true', help='When true, stores permanent checkpoints')
    parser.add_argument('--aug', action='store_true', help='When true, uses training set augmentations')
    parser.add_argument('--distributed', action='store_true', help='When true, uses distributed compute across GPUs')

    parser.add_argument('--var_n_params', type=str, default='', help='varies number of network parameters')
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
    parser.add_argument('--verbose', action='store_true', help='')
    parser.add_argument('--double_val', action='store_true', help='')
    parser.add_argument('--hp_idx', type=int, default=None, help='')
    parser.add_argument('--grid_id', type=int, default=5, help='')
    parser.add_argument('--lr_range', action='store_true', help='')
    parser.add_argument('--mix_pre', action='store_true', help='')
    parser.add_argument('--mix_pre_apex', action='store_true', help='')
    parser.add_argument('--cycle_mom', action='store_true', help='')
    parser.add_argument('--one_shot', action='store_true', help='')
    parser.add_argument('--search', action='store_true', help='')
    parser.add_argument('--skip_actfuns', action='store_true', help='When true, skip redundant actfuns')
    parser.add_argument('--balanced', action='store_true', help='When true num params are adjusted to be equal')
    parser.add_argument('--bs_factor', type=float, default=1.0, help='Batch size reduction factor')
    parser.add_argument('--lr_factor', type=float, default=1.0, help='Learning rate reduction factor')

    args = parser.parse_args()

    setup_experiment(args)
