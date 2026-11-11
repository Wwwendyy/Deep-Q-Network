#!/bin/bash
#SBATCH --account=e32706
#SBATCH --partition=gengpu,gengpu-long
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --job-name=dqn-cp-grid
#SBATCH --array=0-7%1
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err

set -euo pipefail

module purge
module load anaconda3/2022.05
source /software/anaconda3/2022.05/etc/profile.d/conda.sh
unset PYTHONPATH PYTHONHOME
export PYTHONNOUSERSITE=1

cd "$SLURM_SUBMIT_DIR"
conda activate "$SLURM_SUBMIT_DIR/.conda_env"

learning_rates=(0.00002 0.00002 0.00005 0.00005 0.0001 0.0001 0.0002 0.0002)
epsilon_decays=(50000 150000 50000 150000 50000 150000 50000 150000)
idx="${SLURM_ARRAY_TASK_ID}"
lr="${learning_rates[$idx]}"
decay="${epsilon_decays[$idx]}"
out_dir="runs/cartpole_grid/lr${lr}_eps${decay}_seed123"

mkdir -p "$out_dir"
echo "Host: $(hostname)"
echo "Config: learning_rate=$lr epsilon_decay_steps=$decay seed=123"
nvidia-smi

"$CONDA_PREFIX/bin/python" -u dqn_assignment_validated.py \
  --env CartPole-v1 \
  --episodes 5000 \
  --gamma 0.95 \
  --learning-rate "$lr" \
  --batch-size 64 \
  --buffer-size 50000 \
  --learning-starts 1000 \
  --train-every 1 \
  --target-update 1000 \
  --epsilon-end 0.05 \
  --epsilon-decay-steps "$decay" \
  --validation-every 250 \
  --validation-episodes 20 \
  --validation-seed 100000 \
  --device cuda \
  --out-dir "$out_dir" \
  --seed 123
