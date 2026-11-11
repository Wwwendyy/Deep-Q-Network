#!/usr/bin/env python3
"""Rank CartPole grid runs by their best fixed-seed greedy validation mean."""
import csv
from pathlib import Path

root = Path("runs/cartpole_grid")
results = []
for csv_path in root.glob("*/validation.csv"):
    with csv_path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        continue
    best = max(rows, key=lambda r: float(r["greedy_mean_reward"]))
    model = csv_path.parent / "best_eval_model.pt"
    results.append((float(best["greedy_mean_reward"]), float(best["greedy_std_reward"]),
                    int(best["episode"]), model, csv_path.parent))

if not results:
    raise SystemExit(f"No completed validation.csv files found under {root}")

results.sort(reverse=True, key=lambda r: r[0])
print("Rank  ValMean  ValStd  Episode  Checkpoint")
for rank, (mean, std, episode, model, run_dir) in enumerate(results, 1):
    print(f"{rank:>4}  {mean:>7.2f}  {std:>6.2f}  {episode:>7}  {model}")
print("\nUse the top checkpoint for final evaluation on a new seed range.")
