#!/bin/bash
#SBATCH -p t4v2
#SBATCH --exclude=gpu102
#SBATCH --exclude=gpu115
#SBATCH --gres=gpu:1                        # request GPU(s)
#SBATCH --qos=normal
#SBATCH -c 4                                # number of CPU cores
#SBATCH --mem=8G                            # memory per node
#SBATCH --time=40:00:00                     # max walltime, hh:mm:ss
#SBATCH --array=0-19%20                    # array value
#SBATCH --output=logs_new/p1_mlp_cnn_10/%a-%N-%j    # %N for node name, %j for jobID
#SBATCH --job-name=p1_mlp_cnn_10

source ~/.bashrc
source activate ~/venvs/combinact

SAVE_PATH="$1"
MODEL="$2"
DATASET="$3"
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

python engine.py --seed $SEED --save_path $SAVE_PATH --check_path $CHECK_DIR --model $MODEL --optim onecycle --num_epochs 10 --dataset $DATASET --actfun max --aug
python engine.py --seed $SEED --save_path $SAVE_PATH --check_path $CHECK_DIR --model $MODEL --optim onecycle --num_epochs 10 --dataset $DATASET --actfun relu --aug
python engine.py --seed $SEED --save_path $SAVE_PATH --check_path $CHECK_DIR --model $MODEL --optim onecycle --num_epochs 10 --dataset $DATASET --actfun bin_all_max_min --aug
python engine.py --seed $SEED --save_path $SAVE_PATH --check_path $CHECK_DIR --model $MODEL --optim onecycle --num_epochs 10 --dataset $DATASET --actfun ail_xnor --aug
python engine.py --seed $SEED --save_path $SAVE_PATH --check_path $CHECK_DIR --model $MODEL --optim onecycle --num_epochs 10 --dataset $DATASET --actfun ail_or --aug
python engine.py --seed $SEED --save_path $SAVE_PATH --check_path $CHECK_DIR --model $MODEL --optim onecycle --num_epochs 10 --dataset $DATASET --actfun ail_all_or_and --aug
python engine.py --seed $SEED --save_path $SAVE_PATH --check_path $CHECK_DIR --model $MODEL --optim onecycle --num_epochs 10 --dataset $DATASET --actfun ail_all_or_xnor --aug
python engine.py --seed $SEED --save_path $SAVE_PATH --check_path $CHECK_DIR --model $MODEL --optim onecycle --num_epochs 10 --dataset $DATASET --actfun ail_all_or_and_xnor --aug
python engine.py --seed $SEED --save_path $SAVE_PATH --check_path $CHECK_DIR --model $MODEL --optim onecycle --num_epochs 10 --dataset $DATASET --actfun ail_part_or_xnor --aug
python engine.py --seed $SEED --save_path $SAVE_PATH --check_path $CHECK_DIR --model $MODEL --optim onecycle --num_epochs 10 --dataset $DATASET --actfun ail_part_or_and_xnor --aug
