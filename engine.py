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
            elif actfun in ['ail_all_or_and_xnor']:
                resnet_width = _WIDTH_FACTORS[args.resnet_type]["3/2"]
            else:
                resnet_width = _WIDTH_FACTORS[args.resnet_type]["1/2"]
    else:
        actfun = args.actfun
        resnet_width = args.resnet_width

    # =========================== Creating new output file
    fieldnames = ['dataset', 'seed', 'epoch', 'time', 'actfun',
                  'sample_size', 'model', 'batch_size', 'alpha_primes', 'alphas',
                  'num_params', 'var_nparams', 'var_nsamples', 'k', 'p', 'g', 'perm_method',
                  'gen_gap', 'aug_gen_gap', 'resnet_ver', 'resnet_type', 'resnet_width',
                  'epoch_train_loss', 'epoch_train_acc', 'epoch_aug_train_loss',
                  'epoch_aug_train_acc', 'epoch_val_loss', 'epoch_val_acc', 'epoch_aug_val_loss',
                  'epoch_aug_val_acc', 'hp_idx', 'curr_lr', 'found_lr', 'hparams', 'epochs']

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

    if checkpoint is None or checkpoint_valid:
        if not os.path.exists(outfile_path):
            with open(outfile_path, mode='w') as out_file:
                writer = csv.DictWriter(out_file, fieldnames=fieldnames, lineterminator='\n')
                writer.writeheader()

        # =========================== Training
        num_params = util.get_num_params(args)
        train_samples = args.sample_size
        p, k, g = 1, 2, 1
        perm_method = args.perm_method

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
                      args.seed,
                      outfile_path,
                      filename,
                      fieldnames,
                      sample_size,
                      device,
                      num_params=num_params,
                      curr_p=p,
                      curr_k=k,
                      curr_g=g,
                      perm_method=perm_method,
                      resnet_width=resnet_width)


# --------------------  Entry Point
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Higher order activation function testing')
    parser.add_argument('--seed', type=int, default=0, help='Job seed')
    parser.add_argument('--p', type=int, default=1, help='Default p value for model')
    parser.add_argument('--k', type=int, default=2, help='Default k value for model')
    parser.add_argument('--g', type=int, default=1, help='Default g value for model')
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
    parser.add_argument('--perm_method', type=str, default='shuffle', help='Which permuation method to use')  # shuffle
    parser.add_argument('--label', type=str, default='', help='Label to differentiate different jobs')
    parser.add_argument('--validation', action='store_true', help='When true, uses validation set instead of test set')
    parser.add_argument('--aug', action='store_true', help='When true, uses training set augmentations')
    parser.add_argument('--distributed', action='store_true', help='When true, uses distributed compute across GPUs')

    parser.add_argument('--verbose', action='store_true', help='')
    parser.add_argument('--hp_idx', type=int, default=None, help='')
    parser.add_argument('--grid_id', type=int, default=5, help='')
    parser.add_argument('--mix_pre', action='store_true', help='')
    parser.add_argument('--mix_pre_apex', action='store_true', help='')
    parser.add_argument('--search', action='store_true', help='')
    parser.add_argument('--skip_actfuns', action='store_true', help='When true, skip redundant actfuns')
    parser.add_argument('--balanced', action='store_true', help='When true num params are adjusted to be equal')
    parser.add_argument('--bs_factor', type=float, default=1.0, help='Batch size reduction factor')
    parser.add_argument('--lr_factor', type=float, default=1.0, help='Learning rate reduction factor')

    args = parser.parse_args()

    setup_experiment(args)
