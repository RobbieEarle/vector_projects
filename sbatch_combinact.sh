#!/bin/bash
#SBATCH -p t4v2
#SBATCH --exclude=gpu102
#SBATCH --exclude=gpu115
#SBATCH --gres=gpu:1                        # request GPU(s)
#SBATCH --qos=normal
#SBATCH -c 4                                # number of CPU cores
#SBATCH --mem=8G                            # memory per node
#SBATCH --array=0-99%5                    # array value
#SBATCH --output=logs_new/wrn_50_rs1/%a-%N-%j    # %N for node name, %j for jobID
#SBATCH --job-name=wrn_50_rs1

source ~/.bashrc
source activate ~/venvs/combinact

ACTFUN="$1"
RN_WIDTH="$2"
SEED="$SLURM_ARRAY_TASK_ID"

SAVE_PATH=~/vector_projects/outputs/wrn_50_rs1
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

python engine.py --seed $SEED --save_path $SAVE_PATH --check_path $CHECK_DIR --model resnet --resnet_width $RN_WIDTH --optim onecycle --num_epochs 100 --dataset cifar10 --actfun $ACTFUN --aug --validation --search --mix_pre_apex
python engine.py --seed $SEED --save_path $SAVE_PATH --check_path $CHECK_DIR --model resnet --resnet_width $RN_WIDTH --optim onecycle --num_epochs 100 --dataset cifar100 --actfun $ACTFUN --aug --validation --search --mix_pre_apex
