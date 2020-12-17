#!/bin/bash
#SBATCH -p t4v2
#SBATCH --exclude=gpu102
#SBATCH --exclude=gpu115
#SBATCH --gres=gpu:1                        # request GPU(s)
#SBATCH --qos=normal
#SBATCH -c 4                                # number of CPU cores
#SBATCH --mem=8G                            # memory per node
#SBATCH --time=40:00:00                     # max walltime, hh:mm:ss
#SBATCH --array=0-119%10                     # array value
#SBATCH --output=logs_new/hparam_rs3/100e/%a-%N-%j    # %N for node name, %j for jobID
#SBATCH --job-name=hparam_rs3

source ~/.bashrc
source activate ~/venvs/combinact

SAVE_PATH="$1"
EPOCHS="$2"
ACTFUN="$3"
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
python -c "import sklearn; print('sklearn version = {}'.format(sklearn.__version__))"
python -c "import matplotlib; print('matplotlib version = {}'.format(matplotlib.__version__))"
echo ""

echo "SAVE_PATH=$SAVE_PATH"
echo "SEED=$SEED"

python engine.py --seed $SEED --save_path $SAVE_PATH --check_path $CHECK_DIR --model mlp --optim onecycle --num_epochs $EPOCHS --dataset cifar100 --actfun $ACTFUN --aug --validation --label _$EPOCHS
python engine.py --seed $SEED --save_path $SAVE_PATH --check_path $CHECK_DIR --model cnn --optim onecycle --num_epochs $EPOCHS --dataset cifar100 --actfun $ACTFUN --aug --validation --label _$EPOCHS
