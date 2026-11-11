#!/bin/bash
#SBATCH --account=e32706
#SBATCH --partition=gengpu,gengpu-long
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=48:00:00
#SBATCH --job-name=dqn-mspacman
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

cd /projects/e32706/way5959/MLDS_hw2

module purge
module load anaconda3/2022.05
source /software/anaconda3/2022.05/etc/profile.d/conda.sh
unset PYTHONPATH PYTHONHOME
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1

conda activate "$SLURM_SUBMIT_DIR/.conda_env"
PYTHON="$CONDA_PREFIX/bin/python"

mkdir -p runs/mspacman_grid logs

echo "Host: $(hostname)"
echo "Job: ${SLURM_JOB_ID}"
nvidia-smi

# Start training only if the GPU and MsPacman setup pass these checks.
"$PYTHON" - <<'PY'
import torch
from dqn_assignment_validated import make_env, pack_obs

assert torch.cuda.is_available(), "PyTorch cannot access the allocated GPU."
env = make_env("MsPacman-v5")
obs, _ = env.reset(seed=42)
assert env.action_space.n == 9, f"Expected 9 actions, got {env.action_space.n}"
packed = pack_obs(obs, is_atari=True)
assert packed.shape == (88, 80), f"Unexpected image shape: {packed.shape}"
print(f"Preflight passed: actions={env.action_space.n}, image={packed.shape}, "
      f"GPU={torch.cuda.get_device_name(0)}")
env.close()
PY

"$PYTHON" -u dqn_assignment_validated.py \
  --env MsPacman-v5 \
  --episodes 5000 \
  --gamma 0.99 \
  --learning-rate 0.0001 \
  --batch-size 32 \
  --buffer-size 50000 \
  --learning-starts 10000 \
  --target-update 1000 \
  --epsilon-end 0.1 \
  --epsilon-decay-steps 100000 \
  --checkpoint-every 100 \
  --device cuda \
  --out-dir runs/mspacman_grid/seed42 \
  --seed 42