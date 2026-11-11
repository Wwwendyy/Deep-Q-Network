#!/bin/bash
#SBATCH --account=e32706
#SBATCH --partition=gengpu,gengpu-long
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --job-name=dqn-evaluate
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

ENV_NAME="${1:-CartPole-v1}"
MODEL_PATH="${2:-runs/cartpole/model.pt}"
SEED="${3:-42}"
OUT_DIR="${4:-$(dirname "$MODEL_PATH")/eval_seed_${SEED}}"

"$CONDA_PREFIX/bin/python" -u dqn_assignment_validated.py \
  --env "$ENV_NAME" \
  --evaluate-only "$MODEL_PATH" \
  --eval-episodes 500 \
  --seed "$SEED" \
  --device cuda \
  --out-dir "$OUT_DIR"