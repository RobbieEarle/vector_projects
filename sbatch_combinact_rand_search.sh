#!/bin/bash
#SBATCH -p p100,max12hour,t4   # partition - should be gpu on MaRS (q), and either p100 or t4 on Vaughan (vremote1)
#SBATCH --gres=gpu:1           # request GPU(s)
#SBATCH -c 4                   # number of CPU cores
#SBATCH --mem=8G               # memory per node
#SBATCH --time=12:00:00        # max walltime, hh:mm:ss
#SBATCH --array=0-999%10       # array value
#SBATCH --output=logs/combinact/%a-%N-%j    # %N for node name, %j for jobID
#SBATCH --job-name=combinact_rw

source ~/.bashrc
source activate ~/venvs/combinact

NUM_ITERATIONS="$1"
SAVE_PATH="$2"
SEED="$SLURM_ARRAY_TASK_ID"

# Debugging outputs
pwd
which conda
python --version
pip freeze

python combinact_rand_search.py "relu" "$SEED" "$SAVE_PATH"
python combinact_rand_search.py "max" "$SEED" "$SAVE_PATH"
python combinact_rand_search.py "signed_geomean" "$SEED" "$SAVE_PATH"
python combinact_rand_search.py "swish2" "$SEED" "$SAVE_PATH"
python combinact_rand_search.py "l2" "$SEED" "$SAVE_PATH"
python combinact_rand_search.py "linf" "$SEED" "$SAVE_PATH"
python combinact_rand_search.py "zclse-approx" "$SEED" "$SAVE_PATH"
