#!/bin/bash
#SBATCH -p t4v2
#SBATCH --exclude=gpu102
#SBATCH --gres=gpu:1                        # request GPU(s)
#SBATCH --qos=normal
#SBATCH -c 4                                # number of CPU cores
#SBATCH --mem=8G                            # memory per node
#SBATCH --time=30:00:00                     # max walltime, hh:mm:ss
#SBATCH --array=0%1                         # array value
#SBATCH --output=logs/e8_resnet_rs5/%a-%N-%j    # %N for node name, %j for jobID
#SBATCH --job-name=e8_resnet_rs5

source ~/.bashrc
source activate ~/venvs/combinact

SAVE_PATH="$1"
ACTFUN="$2"
HP_IDX="$3"
SEED="$SLURM_ARRAY_TASK_ID"

touch /checkpoint/robearle/${SLURM_JOB_ID}
CHECK_DIR=/checkpoint/robearle/${SLURM_JOB_ID}

# Debugging outputs
pwd
which conda
python --version
pip freeze

echo ""
python -c "import torch; print('torch version = {}'.format(torch.__version__))"
python -c "import torch.cuda; print('cuda = {}'.format(torch.cuda.is_available()))"
python -c "import scipy; print('scipy version = {}'.format(scipy.__version__))"
echo ""

echo "SAVE_PATH=$SAVE_PATH"
echo "SEED=$SEED"

python engine.py --seed 0 --save_path $SAVE_PATH --check_path $CHECK_DIR --model resnet --dataset mnist --actfun $ACTFUN --resnet_ver 34 --resnet_width 2 --num_epochs 50 --hp_idx $HP_IDX --label $HP_IDX
python engine.py --seed 0 --save_path $SAVE_PATH --check_path $CHECK_DIR --model resnet --dataset cifar10 --actfun $ACTFUN --resnet_ver 34 --resnet_width 2 --num_epochs 50 --hp_idx $HP_IDX --label $HP_IDX
python engine.py --seed 0 --save_path $SAVE_PATH --check_path $CHECK_DIR --model resnet --dataset cifar100 --actfun $ACTFUN --resnet_ver 34 --resnet_width 2 --num_epochs 50 --hp_idx $HP_IDX --label $HP_IDX

