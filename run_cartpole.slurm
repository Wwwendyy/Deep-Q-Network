#!/bin/bash
#SBATCH --account=e32706
#SBATCH --partition=gengpu,gengpu-long
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --job-name=dqn-cartpole
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

module purge
module load anaconda3/2022.05
source /software/anaconda3/2022.05/etc/profile.d/conda.sh

unset PYTHONPATH PYTHONHOME
export PYTHONNOUSERSITE=1

cd "$SLURM_SUBMIT_DIR"
conda activate "$SLURM_SUBMIT_DIR/.conda_env"

echo "Host: $(hostname)"
nvidia-smi

"$CONDA_PREFIX/bin/python" -u dqn_assignment.py \
  --env CartPole-v1 \
  --episodes 5000 \
  --gamma 0.95 \
  --learning-rate 0.00005 \
  --batch-size 64 \
  --buffer-size 50000 \
  --learning-starts 1000 \
  --train-every 1 \
  --target-update 1000 \
  --epsilon-end 0.05 \
  --epsilon-decay-steps 150000 \
  --device cuda \
  --out-dir runs/cartpole_retry \
  --seed 123