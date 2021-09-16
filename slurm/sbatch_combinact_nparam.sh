#!/bin/bash
#SBATCH -p p100,t4v1,t4v2
#SBATCH --gres=gpu:1                        # request GPU(s)
#SBATCH --qos=normal
#SBATCH -c 24                                # number of CPU cores
#SBATCH --mem=128G                            # memory per node
#SBATCH --time=700:00:00                     # max walltime, hh:mm:ss
#SBATCH --array=0-10%11                    # array value
#SBATCH --output=logs_new/wrn_50_nparam3/%a-%N-%j    # %N for node name, %j for jobID
#SBATCH --job-name=wrn_50_nparam3

source ~/.bashrc
source activate ~/venvs/combinact

SEED="$1"
RESNET_WIDTH="$2"
ACTFUN_IDX="$SLURM_ARRAY_TASK_ID"

SAVE_PATH=~/vector_projects/outputs/wrn_50_nparam2
CHECK_PATH="/checkpoint/$USER/${SLURM_JOB_ID}"
touch $CHECK_PATH

# Debugging outputs
pwd
which conda
python --version
pip freeze

echo ""
python -c "import torch; print('torch version = {}'.format(torch.__version__))"
python -c "import torch.cuda; print('cuda = {}'.format(torch.cuda.is_available()))"
python -c "import scipy; print('scipy version = {}'.format(scipy.__version__))"
python -c "import sklearn; print('sklearn version = {}'.format(sklearn.__version__))"
python -c "import matplotlib; print('matplotlib version = {}'.format(matplotlib.__version__))"
echo ""

echo "SAVE_PATH=$SAVE_PATH"
echo "SEED=$SEED"

python engine.py --seed $SEED --save_path $SAVE_PATH --check_path $CHECK_PATH --model resnet --batch_size 128 --actfun_idx $ACTFUN_IDX --resnet_width $RESNET_WIDTH --optim onecycle --num_epochs 100 --dataset cifar100 --aug --mix_pre_apex --c 80
python engine.py --seed $SEED --save_path $SAVE_PATH --check_path $CHECK_PATH --model resnet --batch_size 128 --actfun_idx $ACTFUN_IDX --resnet_width $RESNET_WIDTH --optim onecycle --num_epochs 100 --dataset cifar100 --aug --mix_pre_apex --c 25
python engine.py --seed $SEED --save_path $SAVE_PATH --check_path $CHECK_PATH --model resnet --batch_size 128 --actfun_idx $ACTFUN_IDX --resnet_width $RESNET_WIDTH --optim onecycle --num_epochs 100 --dataset cifar100 --aug --mix_pre_apex --c 8
python engine.py --seed $SEED --save_path $SAVE_PATH --check_path $CHECK_PATH --model resnet --batch_size 128 --actfun_idx $ACTFUN_IDX --resnet_width $RESNET_WIDTH --optim onecycle --num_epochs 100 --dataset cifar100 --aug --mix_pre_apex --c 3
