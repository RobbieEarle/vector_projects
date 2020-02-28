#!/bin/bash
#SBATCH -p p100,max12hours     # partition - should be gpu on MaRS (q), and either p100 or t4 on Vaughan (vremote1)
#SBATCH --gres=gpu:1           # request GPU(s)
#SBATCH -c 4                   # number of CPU cores
#SBATCH --mem=8G               # memory per node
#SBATCH --time=12:00:00        # max walltime, hh:mm:ss
#SBATCH --array=0-999%10       # array value
#SBATCH --output=logs/combinact/%a-%N-%j    # %N for node name, %j for jobID
#SBATCH --job-name=combinact_rw

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
​
echo ""

echo "SAVE_PATH=$SAVE_PATH"
echo "SEED=$SEED"

~/utilities/log_gpu_cpu_stats -l 0.5 -n 600 "logs/combinact/${SLURM_ARRAY_TASK_ID}_${SLURM_NODEID}_${SLURM_ARRAY_JOB_ID}_compute_usage.csv"

python combinact_rand_search.py "relu" "$SEED" "$SAVE_PATH"
python combinact_rand_search.py "max" "$SEED" "$SAVE_PATH"
python combinact_rand_search.py "signed_geomean" "$SEED" "$SAVE_PATH"
python combinact_rand_search.py "swish2" "$SEED" "$SAVE_PATH"
python combinact_rand_search.py "l2" "$SEED" "$SAVE_PATH"
python combinact_rand_search.py "linf" "$SEED" "$SAVE_PATH"
python combinact_rand_search.py "zclse-approx" "$SEED" "$SAVE_PATH"
