#!/bin/bash
#SBATCH -p p100                # partition - should be gpu on MaRS (q), and either p100 or t4 on Vaughan (vremote1)
#SBATCH --gres=gpu:1           # request GPU(s)
#SBATCH -c 4                   # number of CPU cores
#SBATCH --mem=8G               # memory per node
#SBATCH --time=12:00:00        # max walltime, hh:mm:ss
#SBATCH --array=0-499%10       # array value
#SBATCH --output=logs/combinact7/%a-%N-%j    # %N for node name, %j for jobID
#SBATCH --job-name=combinact7

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
echo ""

echo "SAVE_PATH=$SAVE_PATH"
echo "SEED=$SEED"

python combinact.py "0" "$SEED" "$SAVE_PATH"
python combinact.py "1" "$SEED" "$SAVE_PATH"
python combinact.py "2" "$SEED" "$SAVE_PATH"
python combinact.py "3" "$SEED" "$SAVE_PATH"
python combinact.py "4" "$SEED" "$SAVE_PATH"
python combinact.py "5" "$SEED" "$SAVE_PATH"
