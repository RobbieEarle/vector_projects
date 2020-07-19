#!/bin/bash
#SBATCH -p p100                # partition - should be gpu on MaRS (q), and either p100 or t4 on Vaughan (vremote1)
#SBATCH --exclude=gpu053
#SBATCH --gres=gpu:1           # request GPU(s)
#SBATCH -c 4                   # number of CPU cores
#SBATCH --mem=8G               # memory per node
#SBATCH --time=20:00:00        # max walltime, hh:mm:ss
#SBATCH --array=0-50%10        # array value
#SBATCH --output=logs/cmbnct_nparam2b/%a-%N-%j    # %N for node name, %j for jobID
#SBATCH --job-name=cmbnct_nparam2b

source ~/.bashrc
source activate ~/venvs/combinact

SAVE_PATH="$1"
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
echo ""

echo "SAVE_PATH=$SAVE_PATH"
echo "SEED=$SEED"

python train.py --seed $SEED --save_path $SAVE_PATH --model cnn --actfun combinact --var_n_params --reduce_actfuns
