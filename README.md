# MLDS Assignment 2: Deep Q-Network

## Repository information

- **GitHub username:** Wwwendyy
- **Repository name:** Deep-Q-Network
- **Public repository URL:** https://github.com/Wwwendyy/Deep-Q-Network
- **Environment:** Northwestern Quest, one GPU per training job

## Overview

This project implements Deep Q-Network (DQN) agents for CartPole-v1 and MsPacman-v5. It uses neural Q-functions, experience replay, a target network, Huber loss, and gradient clipping. The Atari observation is preprocessed into an 88 × 80 grayscale image using the assignment's specified crop, downsampling, grayscale conversion, contrast-pixel removal, and int8 normalization.

The CartPole agent uses a multilayer perceptron. The MsPacman agent uses a convolutional neural network and predicts values for all nine actions. Image transitions are stored as uint8 in the replay buffer to reduce memory use.

## Results

| Environment | Training | Evaluation episodes | Mean reward | Standard deviation |
|---|---:|---:|---:|---:|
| CartPole-v1 | 5,000-episode hyperparameter sweep; selected checkpoint from validation | 500 | 500.00 | 0.00 |
| MsPacman-v5 | 5,000 episodes | 500 | 1,188.88 | 358.11 |

### CartPole-v1

The required discount factor was 0.95. I compared eight configurations using four learning rates (2e-5, 5e-5, 1e-4, 2e-4) and two epsilon-decay lengths (50,000 and 150,000 environment steps). Other settings were held fixed: batch size 64, replay capacity 50,000, 1,000 learning-start steps, and target-network update every 1,000 gradient updates. Several configurations reached the 500-point ceiling on the 20-episode validation set. I selected the 2e-5 learning rate and 50,000-step epsilon decay, then evaluated the selected checkpoint on 500 separate seeds. Every evaluation episode reached the 500-step limit.

A CartPole episode gives one reward per environment step. A return of 500 therefore means the pole remained balanced until the environment's 500-step time limit. The reported 500 ± 0.00 is the mean and sample standard deviation across the 500 evaluation episodes.

**Selected configuration:** gamma=0.95, learning rate=2e-5, batch size=64, replay capacity=50,000, learning starts=1,000, target update=1,000, epsilon from 1.0 to 0.05 over 50,000 steps, seed=123.

![CartPole training rewards and 100-episode moving average](report_figures/cartpole_training_rewards.png)

![CartPole mean maximum Q-value by episode](report_figures/cartpole_q_values.png)

![CartPole greedy-policy evaluation over 500 episodes](report_figures/cartpole_evaluation_histogram.png)

### MsPacman-v5

The required discount factor was 0.99. The agent received the assignment-prescribed 88 × 80 preprocessed image and used nine actions. Training ran for 5,000 episodes with learning rate 1e-4, batch size 32, replay capacity 50,000, 10,000 learning-start steps, target-network update every 1,000 gradient updates, and epsilon decaying from 1.0 to 0.1 over 100,000 environment steps.

The mean reward over the first 100 training episodes was 300.3; over the final 100 episodes it was 1,153.9. This is clear evidence of improvement during training. On 500 greedy evaluation episodes using a separate seed range, the mean reward was 1,188.88 with standard deviation 358.11. The standard deviation is large, so performance is variable across episodes; the evaluation nevertheless confirms that the learned policy obtains substantial scores.

![MsPacman training rewards and 100-episode moving average](report_figures/mspacman_training_rewards.png)

![MsPacman mean maximum Q-value by episode](report_figures/mspacman_q_values.png)

![MsPacman greedy-policy evaluation over 500 episodes](report_figures/mspacman_evaluation_histogram.png)

## Non-standard diagnostic plot

The replay-sample-age plot tracks how old sampled transitions are relative to the current environment step. It helps interpret replay-buffer behavior: a larger age means the update is learning from older experience, while a smaller age means samples are more recent. The plot is included as an additional diagnostic rather than a required score plot.

![Replay-buffer sample age during CartPole training](report_figures/cartpole_replay_sample_age.png)

## Edge cases and correctness checks

The implementation and validation run address these cases:

- **Gymnasium API variations:** reset and step helpers support the common return signatures, including the separate terminated and truncated flags.
- **Termination versus time-limit truncation:** a true terminal transition does not bootstrap from the next state. A time-limit truncation ends the episode but still bootstraps, as it is not necessarily a terminal state in the underlying MDP.
- **Replay warm-up:** learning is skipped until the replay buffer contains at least the larger of the batch size and configured learning-start threshold.
- **MsPacman observation contract:** synthetic input checks the crop/downsampling output shape (88 × 80 × 1), int8 type, value range, and contrast-removal pixel.
- **MsPacman action space:** training checks that the environment exposes exactly nine actions.
- **Reproducibility:** resetting CartPole with the same seed and applying the same ten-action sequence must reproduce the same state/reward/termination trace.

Run the validation experiments with:

\`\`\`bash
./.conda_env/bin/python dqn_assignment_validated.py --validate --seed 2026
\`\`\`

The run writes \`runs/validation/validation.json\` and \`runs/validation/policy_sanity_check.png\`. With seed 2026, the preprocessing output was (88, 80, 1) with dtype int8; the fixed-action trace was reproducible; the terminal Bellman target was 1.0 and the truncation target was 4.8 for reward=1, gamma=0.95, and next-state value=4. Across 40 common seeds, the random policy achieved 20.825 ± 12.830 reward, while a fixed angle-based heuristic achieved 500.0 ± 0.0. This is a controlled environment/target sanity check; the learned DQN policies are evaluated separately above.

![Controlled CartPole policy sanity check](report_figures/policy_sanity_check.png)

## Reproducing the experiments

Run these commands from the repository root on Quest. Create \`logs/\` before submitting jobs because Slurm opens its output files before the job script starts.

\`\`\`bash
module purge
module load anaconda3/2022.05
source /software/anaconda3/2022.05/etc/profile.d/conda.sh
unset PYTHONPATH PYTHONHOME
export PYTHONNOUSERSITE=1
conda activate "$PWD/.conda_env"
\`\`\`

Run the correctness/behavior validation:

\`\`\`bash
./.conda_env/bin/python dqn_assignment_validated.py --validate --seed 2026
\`\`\`

Submit the CartPole grid and MsPacman training jobs:

\`\`\`bash
mkdir -p logs
sbatch run_cartpole_grid.slurm
sbatch run_mspacman.slurm
\`\`\`

The CartPole grid summarizes validation results with:

\`\`\`bash
./.conda_env/bin/python summarize_cartpole_grid.py
\`\`\`

For final evaluation, \`run_evaluation.slurm\` should call \`dqn_assignment_validated.py\` so it can load checkpoints containing validation configuration fields. Example calls:

\`\`\`bash
sbatch run_evaluation.slurm CartPole-v1 \\
  runs/cartpole_grid/lr0.00002_eps50000_seed123/best_eval_model.pt \\
  200000 runs/cartpole_grid/final_eval_lr2e-5_eps50000

sbatch run_evaluation.slurm MsPacman-v5 \\
  runs/mspacman_grid/seed42/best_model.pt \\
  200000 runs/mspacman_grid/seed42/eval500
\`\`\`

The evaluation command runs 500 greedy episodes. The evaluation function uses the provided seed as the first seed and increments it for each episode.

## Additional experiments with three more months

With three additional months, I would test performance across multiple independent training seeds and report confidence intervals, not only one selected run. I would run controlled ablations for replay capacity, learning-start length, epsilon schedule, target-update interval, and Double DQN targets. I would also compare uniform replay with prioritized replay and compare the current CNN with a smaller/larger Atari network. For MsPacman, I would evaluate learning curves at fixed environment-step budgets and measure score distributions on multiple held-out seed sets. These experiments would help separate improvements caused by a hyperparameter choice from ordinary run-to-run variation.

## Figure collection

The README embeds figures from \`report_figures/\`. Copy the generated PNG files there before committing the report:

\`\`\`bash
mkdir -p report_figures
cp runs/cartpole_grid/lr0.00002_eps50000_seed123/episode_rewards.png report_figures/cartpole_training_rewards.png
cp runs/cartpole_grid/lr0.00002_eps50000_seed123/q_values.png report_figures/cartpole_q_values.png
cp runs/cartpole_grid/lr0.00002_eps50000_seed123/replay_sample_age.png report_figures/cartpole_replay_sample_age.png
cp runs/cartpole_grid/final_eval_lr2e-5_eps50000/evaluation_histogram.png report_figures/cartpole_evaluation_histogram.png
cp runs/mspacman_grid/seed42/episode_rewards.png report_figures/mspacman_training_rewards.png
cp runs/mspacman_grid/seed42/q_values.png report_figures/mspacman_q_values.png
cp runs/mspacman_grid/seed42/eval500/evaluation_histogram.png report_figures/mspacman_evaluation_histogram.png
cp runs/validation/policy_sanity_check.png report_figures/policy_sanity_check.png
\`\`\`

## Submission checklist

- Replace the GitHub username, repository name, and URL at the top.
- Commit the source code, Slurm scripts, README, and the PNG files in \`report_figures/\`.
- Export this report to the required PDF format and submit the PDF to Canvas.
