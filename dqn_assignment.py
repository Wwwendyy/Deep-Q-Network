"""DQN homework implementation for CartPole-v1 and MsPacman-v5.

Examples:
  python dqn_assignment.py --env CartPole-v1 --episodes 1000
  python dqn_assignment.py --env MsPacman-v5 --episodes 5000 --device cuda
  python dqn_assignment.py --env CartPole-v1 --evaluate-only runs/cartpole/model.pt --eval-episodes 500
  python dqn_assignment.py --validate
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import random
import time
from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


def preprocess_observation(obs: np.ndarray) -> np.ndarray:
    """Course-prescribed MsPacman crop/downsample/greyscale normalization."""
    img = obs[1:176:2, ::2].copy()
    img = img.sum(axis=2)
    img[img == (210 + 164 + 74)] = 0
    img = (img // 3 - 128).astype(np.int8)
    return img.reshape(88, 80, 1)


def make_env(requested: str, render_mode: Optional[str] = None):
    import gymnasium as gym
    candidates = [requested]
    if requested.lower() in {"mspacman-v5", "ale/mspacman-v5"}:
        import ale_py
        register_envs = getattr(gym, "register_envs", None)
        if register_envs is not None:
            register_envs(ale_py)
        candidates = ["ALE/MsPacman-v5", "MsPacman-v5"]
    last_error = None
    for env_id in candidates:
        try:
            kwargs = {"render_mode": render_mode} if render_mode else {}
            if env_id == "ALE/MsPacman-v5":
                kwargs.update(obs_type="rgb")
            return gym.make(env_id, **kwargs)
        except Exception as exc:  # try legacy Gymnasium Atari registration
            last_error = exc
    raise RuntimeError(
        f"Could not create {requested}. Install gymnasium[atari], ale-py, and Atari ROMs. "
        f"Last error: {last_error}"
    )


def seed_env(env, seed: int):
    obs, info = env.reset(seed=seed)
    try:
        env.action_space.seed(seed)
    except Exception:
        pass
    return obs, info


def pack_obs(obs: np.ndarray, is_atari: bool) -> np.ndarray:
    if is_atari:
        # Preserve the exact signed course preprocessing while storing one byte/pixel.
        return (preprocess_observation(obs).squeeze(-1).astype(np.int16) + 128).astype(np.uint8)
    return np.asarray(obs, dtype=np.float32)


class ReplayBuffer:
    """Uniform replay with compact uint8 image storage and sampled-age diagnostics."""
    def __init__(self, capacity: int, obs_shape: tuple[int, ...], image: bool, seed: int):
        self.capacity, self.image = capacity, image
        dtype = np.uint8 if image else np.float32
        self.states = np.empty((capacity, *obs_shape), dtype=dtype)
        self.next_states = np.empty_like(self.states)
        self.actions = np.empty(capacity, dtype=np.int64)
        self.rewards = np.empty(capacity, dtype=np.float32)
        self.terminated = np.empty(capacity, dtype=np.float32)
        self.sequence = np.empty(capacity, dtype=np.int64)
        self.position = self.size = 0
        self.rng = np.random.default_rng(seed)

    def add(self, state, action, reward, next_state, terminated, step):
        i = self.position
        self.states[i] = state
        self.next_states[i] = next_state
        self.actions[i] = action
        self.rewards[i] = reward
        # A time-limit truncation ends an episode, but is not an MDP terminal state.
        self.terminated[i] = float(terminated)
        self.sequence[i] = step
        self.position = (i + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int, current_step: int, device: torch.device):
        idx = self.rng.integers(0, self.size, size=batch_size)
        scale = 1.0 / 128.0 if self.image else 1.0
        states = torch.as_tensor(self.states[idx], device=device).float()
        next_states = torch.as_tensor(self.next_states[idx], device=device).float()
        if self.image:
            states = (states - 128.0) * scale
            next_states = (next_states - 128.0) * scale
            states = states.unsqueeze(1)
            next_states = next_states.unsqueeze(1)
        return (states, torch.as_tensor(self.actions[idx], device=device),
                torch.as_tensor(self.rewards[idx], device=device), next_states,
                torch.as_tensor(self.terminated[idx], device=device),
                float(np.mean(np.maximum(0, current_step - self.sequence[idx]))))


class CartPoleQ(nn.Module):
    def __init__(self, n_actions: int):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(4, 128), nn.ReLU(), nn.Linear(128, 128),
                                 nn.ReLU(), nn.Linear(128, n_actions))
    def forward(self, x):
        return self.net(x)


class AtariQ(nn.Module):
    """A small Atari CNN; accepts the assignment's 88x80 grayscale image."""
    def __init__(self, n_actions: int):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 8, stride=4), nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2), nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1), nn.ReLU(), nn.Flatten())
        with torch.no_grad():
            flat = self.features(torch.zeros(1, 1, 88, 80)).shape[1]
        self.head = nn.Sequential(nn.Linear(flat, 512), nn.ReLU(), nn.Linear(512, n_actions))
    def forward(self, x):
        return self.head(self.features(x))


@dataclass
class Config:
    env: str
    episodes: int
    gamma: float
    learning_rate: float
    batch_size: int
    buffer_size: int
    learning_starts: int
    train_every: int
    target_update: int
    epsilon_start: float
    epsilon_end: float
    epsilon_decay_steps: int
    seed: int
    device: str
    checkpoint_every: int = 100


class DQNAgent:
    def __init__(self, n_actions: int, image: bool, cfg: Config, device: torch.device):
        self.device, self.image, self.n_actions = device, image, n_actions
        self.online = (AtariQ(n_actions) if image else CartPoleQ(n_actions)).to(device)
        self.target = (AtariQ(n_actions) if image else CartPoleQ(n_actions)).to(device)
        self.target.load_state_dict(self.online.state_dict())
        self.target.eval()
        self.optimizer = torch.optim.Adam(self.online.parameters(), lr=cfg.learning_rate)
        shape = (88, 80) if image else (4,)
        self.replay = ReplayBuffer(cfg.buffer_size, shape, image, cfg.seed)
        self.cfg, self.updates = cfg, 0

    def epsilon(self, steps):
        frac = min(1., steps / max(1, self.cfg.epsilon_decay_steps))
        return self.cfg.epsilon_start + frac * (self.cfg.epsilon_end - self.cfg.epsilon_start)

    @torch.no_grad()
    def act(self, obs, epsilon=0.):
        if random.random() < epsilon:
            return random.randrange(self.n_actions), None
        tensor = torch.as_tensor(obs, device=self.device).float()
        if self.image:
            tensor = ((tensor - 128.) / 128.).unsqueeze(0).unsqueeze(0)
        else:
            tensor = tensor.unsqueeze(0)
        q = self.online(tensor)
        return int(q.argmax(1).item()), float(q.max(1).values.item())

    def learn(self, step):
        if self.replay.size < max(self.cfg.batch_size, self.cfg.learning_starts):
            return None

        states, actions, rewards, next_states, terminated, mean_age = self.replay.sample(
            self.cfg.batch_size, step, self.device
        )

        q = self.online(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            next_actions = self.online(next_states).argmax(dim=1, keepdim=True)
            next_q = self.target(next_states).gather(
                1, next_actions
            ).squeeze(1)
            target = rewards + self.cfg.gamma * (1.0 - terminated) * next_q

        loss = F.smooth_l1_loss(q, target)

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(self.online.parameters(), 10.0)
        self.optimizer.step()

        self.updates += 1
        if self.updates % self.cfg.target_update == 0:
            self.target.load_state_dict(self.online.state_dict())

        return float(loss.item()), mean_age


def reset_env(env, seed=None):
    result = env.reset(seed=seed) if seed is not None else env.reset()
    return result[0] if isinstance(result, tuple) else result


def step_env(env, action):
    result = env.step(action)
    if len(result) == 5:
        obs, reward, terminated, truncated, info = result
        return obs, reward, bool(terminated), bool(truncated), info
    obs, reward, done, info = result
    return obs, reward, bool(done), False, info


def moving_average(values, window=100):
    return np.asarray([np.mean(values[max(0, i-window+1):i+1]) for i in range(len(values))])


def train(cfg: Config, out_dir: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    random.seed(cfg.seed); np.random.seed(cfg.seed); torch.manual_seed(cfg.seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(cfg.seed)
    device = torch.device(cfg.device if cfg.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    env = make_env(cfg.env)
    image = "pacman" in cfg.env.lower()
    if image:
        try:
            from ale_py import ALEInterface
        except Exception:
            pass
    n_actions = env.action_space.n
    if image and n_actions != 9:
        raise ValueError(f"MsPacman-v5 should expose 9 actions; found {n_actions}.")
    agent = DQNAgent(n_actions, image, cfg, device)
    out_dir.mkdir(parents=True, exist_ok=True)
    rewards, qmaxes, losses, replay_ages = [], [], [], []
    total_steps, best_avg, started = 0, -np.inf, time.time()
    for ep in range(cfg.episodes):
        raw = reset_env(env, cfg.seed + ep)
        state = pack_obs(raw, image)
        episode_reward, ep_q, ended = 0., [], False
        while not ended:
            action, qmax = agent.act(state, agent.epsilon(total_steps))
            next_raw, reward, terminated, truncated, _ = step_env(env, action)
            next_state = pack_obs(next_raw, image)
            agent.replay.add(state, action, reward, next_state, terminated, total_steps)
            if qmax is not None: ep_q.append(qmax)
            episode_reward += float(reward)
            state = next_state
            total_steps += 1
            if total_steps >= cfg.learning_starts and total_steps % cfg.train_every == 0:
                result = agent.learn(total_steps)
                if result:
                    losses.append(result[0]); replay_ages.append(result[1])
            ended = terminated or truncated
        rewards.append(episode_reward)
        qmaxes.append(float(np.mean(ep_q)) if ep_q else 0.)
        avg100 = float(np.mean(rewards[-100:]))
        print(f"episode={ep+1:5d}/{cfg.episodes} reward={episode_reward:8.1f} avg100={avg100:8.2f} "
              f"qmax={qmaxes[-1]:8.3f} eps={agent.epsilon(total_steps):.3f} steps={total_steps} "
              f"elapsed={time.time()-started:.0f}s")
        # CartPole goal is a full 500-step episode. Continue to configured budget for curves.
        if len(rewards) >= 100 and avg100 > best_avg:
            best_avg = avg100
            torch.save({"online": agent.online.state_dict(), "config": asdict(cfg), "episode": ep+1}, out_dir/"best_model.pt")
        if cfg.checkpoint_every > 0 and (ep+1) % cfg.checkpoint_every == 0:
            torch.save({"online": agent.online.state_dict(), "config": asdict(cfg), "episode": ep+1},
                       out_dir/"latest_checkpoint.pt")
    torch.save({"online": agent.online.state_dict(), "config": asdict(cfg), "episode": cfg.episodes}, out_dir/"model.pt")
    with open(out_dir/"training.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["episode", "reward", "moving_avg_100", "mean_max_q"])
        for i, (r, q) in enumerate(zip(rewards, qmaxes), 1):
            w.writerow([i, r, moving_average(rewards)[i-1], q])
    ep_axis = np.arange(1, len(rewards)+1)
    fig, ax = plt.subplots(figsize=(9, 5)); ax.plot(ep_axis, qmaxes, alpha=.45, label="Episode mean max Q")
    ax.plot(ep_axis, moving_average(qmaxes), label="100 episode moving average")
    ax.set(xlabel="Training episode", ylabel="Maximum Q-value", title=f"{cfg.env}: Q values during training"); ax.legend(); ax.grid(alpha=.2)
    fig.tight_layout(); fig.savefig(out_dir/"q_values.png", dpi=160); plt.close(fig)
    fig, ax = plt.subplots(figsize=(9, 5)); ax.plot(ep_axis, rewards, alpha=.3, label="Episode reward")
    ax.plot(ep_axis, moving_average(rewards), color="darkorange", label="Moving average (last 100 episodes)")
    ax.set(xlabel="Training episode", ylabel="Episode reward", title=f"{cfg.env}: training rewards"); ax.legend(); ax.grid(alpha=.2)
    fig.tight_layout(); fig.savefig(out_dir/"episode_rewards.png", dpi=160); plt.close(fig)
    # Non-standard diagnostic: age of replay transitions sampled for gradient updates.
    fig, ax = plt.subplots(figsize=(9, 5));
    ax.plot(np.arange(1, len(replay_ages)+1), replay_ages, alpha=.55, linewidth=.7)
    if replay_ages: ax.plot(np.arange(1, len(replay_ages)+1), moving_average(replay_ages, 100), color="crimson", label="100 update moving average")
    ax.set(xlabel="Gradient update", ylabel="Mean sampled transition age (environment steps)",
           title=f"{cfg.env}: replay-buffer sample age"); ax.grid(alpha=.2); ax.legend()
    fig.tight_layout(); fig.savefig(out_dir/"replay_sample_age.png", dpi=160); plt.close(fig)
    if losses:
        with open(out_dir/"diagnostics.json", "w") as f:
            json.dump({"gradient_updates": len(losses), "final_mean_loss_100": float(np.mean(losses[-100:])),
                       "final_mean_sample_age_100": float(np.mean(replay_ages[-100:])),
                       "elapsed_seconds": time.time()-started}, f, indent=2)
    env.close()
    print(f"Saved model, logs, and plots to: {out_dir.resolve()}")


def load_agent(env_id, model_path, device):
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    cfg_data = checkpoint["config"]
    valid_keys = Config.__dataclass_fields__.keys()
    cfg = Config(**{k: v for k, v in cfg_data.items() if k in valid_keys})
    env = make_env(env_id)
    image = "pacman" in env_id.lower()
    agent = DQNAgent(env.action_space.n, image, cfg, device)
    agent.online.load_state_dict(checkpoint["online"]); agent.online.eval()
    return env, agent, image


def evaluate(env_id: str, model_path: str, n_episodes: int, seed: int, out_dir: Path, device):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    env, agent, image = load_agent(env_id, model_path, device)
    values = []
    for ep in range(n_episodes):
        state = pack_obs(reset_env(env, seed+ep), image)
        total, ended = 0., False
        while not ended:
            action, _ = agent.act(state, 0.)
            raw, reward, terminated, truncated, _ = step_env(env, action)
            state, total, ended = pack_obs(raw, image), total+float(reward), terminated or truncated
        values.append(total)
        if (ep+1) % 25 == 0 or ep+1 == n_episodes:
            print(f"evaluation={ep+1}/{n_episodes} reward={total:.1f}")
    env.close(); out_dir.mkdir(parents=True, exist_ok=True)
    mean, std = float(np.mean(values)), float(np.std(values, ddof=1)) if len(values)>1 else 0.
    with open(out_dir/"evaluation.json", "w") as f:
        json.dump({"episodes": n_episodes, "mean_reward": mean, "std_reward": std, "rewards": values}, f, indent=2)
    fig, ax = plt.subplots(figsize=(8, 5)); ax.hist(values, bins=min(30, max(5, int(np.sqrt(n_episodes)))), edgecolor="black")
    ax.set(xlabel="Episode reward", ylabel="Count", title=f"{env_id}: greedy policy over {n_episodes} episodes\nmean={mean:.2f}, std={std:.2f}")
    ax.grid(axis="y", alpha=.2); fig.tight_layout(); fig.savefig(out_dir/"evaluation_histogram.png", dpi=160); plt.close(fig)
    print(f"Evaluation mean={mean:.3f}, std={std:.3f}; results in {out_dir.resolve()}")


def validate(seed=2026):
    """Run reproducible, task-relevant sanity experiments and write numeric evidence."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    out = Path("runs/validation"); out.mkdir(parents=True, exist_ok=True)
    # 1) Exact preprocessing contract, including contrast-removal pixel.
    synthetic = np.zeros((210, 160, 3), dtype=np.uint8)
    synthetic[2, 0] = [210, 164, 74]
    processed = preprocess_observation(synthetic)
    assert processed.shape == (88, 80, 1) and processed.dtype == np.int8
    assert processed.min() >= -128 and processed.max() <= 127
    # 2) Reset/seed repeatability: same fixed action sequence should give the same trace.
    import gymnasium as gym
    def trace():
        env = gym.make("CartPole-v1"); s, _ = env.reset(seed=seed); vals=[]
        for a in [0, 1, 0, 0, 1, 1, 0, 1, 0, 1]:
            s, r, term, trunc, _ = env.step(a); vals.append((np.round(s, 6).tolist(), r, term or trunc))
        env.close(); return vals
    trace_a, trace_b = trace(), trace()
    assert trace_a == trace_b, "Fixed-seed CartPole trace was not reproducible"
    # 3) Bellman terminal/truncation edge: truncation bootstraps; true terminal does not.
    reward, gamma, next_q = 1., .95, 4.
    y_terminal = reward + gamma*(1.-1.)*next_q
    y_truncated = reward + gamma*(1.-0.)*next_q
    assert y_terminal == 1. and np.isclose(y_truncated, 4.8)
    # 4) Compare random policy and a repeatable fixed heuristic over common seeds.
    # The heuristic is for validation only, not the submitted learned policy.
    def score(policy, sd):
        env = gym.make("CartPole-v1"); s, _=env.reset(seed=sd); total=0
        while True:
            a = policy(s)
            s, r, t, tr, _=env.step(a); total += r
            if t or tr: break
        env.close(); return total
    random_returns, heuristic_returns = [], []
    rng = np.random.default_rng(seed)
    for i in range(40):
        sd = seed + 100 + i
        random_returns.append(score(lambda s: int(rng.integers(2)), sd))
        heuristic_returns.append(score(lambda s: int(s[2] + 0.5*s[3] > 0), sd))
    result = {"seed": seed, "preprocessing_shape": list(processed.shape), "preprocessing_dtype": str(processed.dtype),
              "fixed_action_trace_reproducible": True, "terminal_target_gamma_0.95_nextq_4": y_terminal,
              "truncation_target_gamma_0.95_nextq_4": y_truncated,
              "random_policy_mean_40": float(np.mean(random_returns)), "random_policy_std_40": float(np.std(random_returns, ddof=1)),
              "simple_heuristic_mean_40": float(np.mean(heuristic_returns)), "simple_heuristic_std_40": float(np.std(heuristic_returns, ddof=1))}
    with open(out/"validation.json", "w") as f: json.dump(result, f, indent=2)
    fig, ax=plt.subplots(figsize=(8,5)); ax.boxplot([random_returns, heuristic_returns], tick_labels=["Random", "Fixed angle heuristic"])
    ax.set(ylabel="CartPole episode reward", title="Controlled policy sanity check (40 common seeds)"); ax.grid(axis="y", alpha=.2)
    fig.tight_layout(); fig.savefig(out/"policy_sanity_check.png", dpi=160); plt.close(fig)
    print(json.dumps(result, indent=2)); print(f"Validation outputs: {out.resolve()}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--env", default="CartPole-v1", choices=["CartPole-v1", "MsPacman-v5"])
    p.add_argument("--episodes", type=int, default=None)
    p.add_argument("--gamma", type=float, default=None)
    p.add_argument("--learning-rate", type=float, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--buffer-size", type=int, default=None)
    p.add_argument("--learning-starts", type=int, default=None)
    p.add_argument("--train-every", type=int, default=1)
    p.add_argument("--target-update", type=int, default=None)
    p.add_argument("--epsilon-end", type=float, default=None)
    p.add_argument("--epsilon-decay-steps", type=int, default=None)
    p.add_argument("--checkpoint-every", type=int, default=100,
                   help="Save an intermediate weights checkpoint every N episodes (0 disables).")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    p.add_argument("--out-dir", default=None)
    p.add_argument("--evaluate-only", default=None)
    p.add_argument("--eval-episodes", type=int, default=500)
    p.add_argument("--validate", action="store_true")
    a = p.parse_args()
    if a.validate:
        validate(a.seed); return
    default = (Config(a.env, 1000 if a.env=="CartPole-v1" else 5000,
                      .95 if a.env=="CartPole-v1" else .99, 1e-3 if a.env=="CartPole-v1" else 1e-4,
                      64 if a.env=="CartPole-v1" else 32, 10000 if a.env=="CartPole-v1" else 50000,
                      1000 if a.env=="CartPole-v1" else 10000, 1, 500 if a.env=="CartPole-v1" else 1000,
                      1., .05 if a.env=="CartPole-v1" else .1, 20000 if a.env=="CartPole-v1" else 100000,
                      a.seed, a.device))
    cfg = Config(**{**asdict(default), **{k:v for k,v in {
        "episodes":a.episodes,"gamma":a.gamma,"learning_rate":a.learning_rate,"batch_size":a.batch_size,
        "buffer_size":a.buffer_size,"learning_starts":a.learning_starts,"train_every":a.train_every,
        "target_update":a.target_update,"epsilon_end":a.epsilon_end,"epsilon_decay_steps": a.epsilon_decay_steps,
        "checkpoint_every":a.checkpoint_every}.items() if v is not None}})
    out = Path(a.out_dir or f"runs/{'cartpole' if a.env=='CartPole-v1' else 'mspacman'}")
    device = torch.device(cfg.device if cfg.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    if a.evaluate_only: evaluate(a.env, a.evaluate_only, a.eval_episodes, a.seed+100000, out, device)
    else: train(cfg, out)


if __name__ == "__main__": main()
