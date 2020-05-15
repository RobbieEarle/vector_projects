#!/bin/bash
#SBATCH -p p100                # partition - should be gpu on MaRS (q), and either p100 or t4 on Vaughan (vremote1)
#SBATCH --gres=gpu:1           # request GPU(s)
#SBATCH -c 4                   # number of CPU cores
#SBATCH --mem=8G               # memory per node
#SBATCH --time=12:00:00        # max walltime, hh:mm:ss
#SBATCH --array=0-499%10       # array value
#SBATCH --output=logs/combinact/%a-%N-%j    # %N for node name, %j for jobID
#SBATCH --job-name=combinact

source ~/.bashrc
source activate ~/venvs/combinact

SAVE_PATH="$1"
MODEL_TYPE="$2"
PERMUTE_TYPE="$3"
ALPHA_DIST="$4"
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
echo "MODEL_TYPE=$MODEL_TYPE"
echo "PERMUTE_TYPE=$PERMUTE_TYPE"
echo "ALPHA_DIST=$ALPHA_DIST"
echo "SEED=$SEED"

python combinact.py "0" "$SEED" "$SAVE_PATH" "$MODEL_TYPE" "$PERMUTE_TYPE" "$ALPHA_DIST"
python combinact.py "1" "$SEED" "$SAVE_PATH" "$MODEL_TYPE" "$PERMUTE_TYPE" "$ALPHA_DIST"
python combinact.py "2" "$SEED" "$SAVE_PATH" "$MODEL_TYPE" "$PERMUTE_TYPE" "$ALPHA_DIST"
python combinact.py "3" "$SEED" "$SAVE_PATH" "$MODEL_TYPE" "$PERMUTE_TYPE" "$ALPHA_DIST"
python combinact.py "4" "$SEED" "$SAVE_PATH" "$MODEL_TYPE" "$PERMUTE_TYPE" "$ALPHA_DIST"
python combinact.py "5" "$SEED" "$SAVE_PATH" "$MODEL_TYPE" "$PERMUTE_TYPE" "$ALPHA_DIST"
