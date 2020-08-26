#!/bin/bash
#SBATCH -p p100                # partition - should be gpu on MaRS (q), and either p100 or t4 on Vaughan (vremote1)
#SBATCH --exclude=gpu053
#SBATCH --gres=gpu:1           # request GPU(s)
#SBATCH --qos=nopreemption
#SBATCH -c 4                   # number of CPU cores
#SBATCH --mem=8G               # memory per node
#SBATCH --time=30:00:00        # max walltime, hh:mm:ss
#SBATCH --array=0-31%8        # array value
#SBATCH --output=logs/bin2_nparam/%a-%N-%j    # %N for node name, %j for jobID
#SBATCH --job-name=bin2_nparam

source ~/.bashrc
source activate ~/venvs/combinact

SAVE_PATH="$1"
SEED="$SLURM_ARRAY_TASK_ID"

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

python engine.py --seed $SEED --save_path $SAVE_PATH --model mlp --dataset mnist --actfun bin --bin_redo
python engine.py --seed $SEED --save_path $SAVE_PATH --model mlp --dataset cifar10 --actfun bin --bin_redo
python engine.py --seed $SEED --save_path $SAVE_PATH --model mlp --dataset cifar100 --actfun bin --bin_redo
python engine.py --seed $SEED --save_path $SAVE_PATH --model cnn --dataset mnist --actfun bin --bin_redo
python engine.py --seed $SEED --save_path $SAVE_PATH --model cnn --dataset cifar10 --actfun bin --bin_redo
python engine.py --seed $SEED --save_path $SAVE_PATH --model cnn --dataset cifar100 --actfun bin --bin_redo
