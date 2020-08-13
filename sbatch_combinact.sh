#!/bin/bash
#SBATCH -p p100                # partition - should be gpu on MaRS (q), and either p100 or t4 on Vaughan (vremote1)
#SBATCH --exclude=gpu053
#SBATCH --gres=gpu:1           # request GPU(s)
#SBATCH -c 4                   # number of CPU cores
#SBATCH --mem=8G               # memory per node
#SBATCH --time=20:00:00        # max walltime, hh:mm:ss
#SBATCH --array=0-49%10        # array value
#SBATCH --output=logs/e3_pk_nn_cifar/%a-%N-%j    # %N for node name, %j for jobID
#SBATCH --job-name=e3_pk_nn_nn_cifar

source ~/.bashrc
source activate ~/venvs/combinact

SAVE_PATH="$1"
SEED="$SLURM_ARRAY_TASK_ID"

#touch /checkpoint/robearle/${SLURM_JOB_ID}
#CHECK_DIR=/checkpoint/robearle/${SLURM_JOB_ID}

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

python engine.py --seed $SEED --save_path $SAVE_PATH --actfun all_pk --dataset cifar10 --model nn --var_k
python engine.py --seed $SEED --save_path $SAVE_PATH --actfun all_pk --dataset cifar10 --model nn --var_p
python engine.py --seed $SEED --save_path $SAVE_PATH --actfun all_pk --dataset cifar100 --model nn --var_k
python engine.py --seed $SEED --save_path $SAVE_PATH --actfun all_pk --dataset cifar100 --model nn --var_p
