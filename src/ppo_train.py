"""PPO training loop with sparse progress logging (Section 5)."""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Optional

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv

from .config import CFG, LOGS_DIR, METRICS_DIR, MODELS_DIR
from .env import make_env_factory
from .utils import get_device


class SparseEpisodeLoggingCallback(BaseCallback):
    """Collect episode stats and print a compact progress line every print_freq steps."""

    def __init__(self, print_freq: int = 40_000):
        super().__init__()
        self.print_freq = int(print_freq)
        self.episode_rewards = []
        self.episode_lengths = []
        self.last_print_step = 0

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        for info in infos:
            if "episode" in info:
                self.episode_rewards.append(float(info["episode"]["r"]))
                self.episode_lengths.append(int(info["episode"]["l"]))

        current_step = int(self.num_timesteps)
        if current_step - self.last_print_step >= self.print_freq:
            self.last_print_step = current_step
            if self.episode_rewards:
                recent_rewards = self.episode_rewards[-20:]
                recent_lengths = self.episode_lengths[-20:]
                print(
                    f"[PPO progress] "
                    f"timesteps={current_step:>7d} | "
                    f"episodes={len(self.episode_rewards):>4d} | "
                    f"mean_reward_last20={float(np.mean(recent_rewards)):>8.3f} | "
                    f"mean_len_last20={float(np.mean(recent_lengths)):>6.1f}"
                )
            else:
                print(
                    f"[PPO progress] timesteps={current_step:>7d} | "
                    "episodes=0 | waiting for completed episodes"
                )
        return True


def build_ppo_model(
    ndvi_field: np.ndarray,
    seed: int,
    tensorboard_log: Optional[Path] = None,
):
    """Build a PPO model on top of a fresh monitored VecEnv."""
    vec_env = DummyVecEnv([make_env_factory(ndvi_field, seed=seed, monitor=True)])
    model = PPO(
        policy="CnnPolicy",
        env=vec_env,
        learning_rate=CFG.learning_rate,
        gamma=CFG.gamma,
        n_steps=CFG.ppo_n_steps,
        batch_size=CFG.ppo_batch_size,
        n_epochs=CFG.ppo_n_epochs,
        verbose=0,
        device=get_device(),
        tensorboard_log=str(tensorboard_log) if tensorboard_log else None,
    )
    return model


def train_ppo(
    ndvi_field: np.ndarray,
    seed: int,
    total_timesteps: Optional[int] = None,
    model_path: Optional[Path] = None,
    print_freq: int = 40_000,
):
    """Train PPO and persist the model + metrics. Returns the trained model."""
    total_timesteps = total_timesteps or CFG.total_timesteps
    model_path = model_path or (MODELS_DIR / "ppo_ndvi_drone_final")

    model = build_ppo_model(ndvi_field, seed=seed, tensorboard_log=LOGS_DIR / "tb")
    logger_cb = SparseEpisodeLoggingCallback(print_freq=print_freq)

    print("=" * 72)
    print("Section 5 | PPO Training")
    print("=" * 72)
    print(f"Total timesteps : {total_timesteps}")
    print(f"Progress print frequency : {logger_cb.print_freq}")
    print(f"Policy                   : CnnPolicy")
    print(f"Device                   : {model.device}")
    print("-" * 72)

    t0 = time.time()
    model.learn(
        total_timesteps=total_timesteps,
        callback=logger_cb,
        progress_bar=False,
    )
    dt = time.time() - t0

    model.save(str(model_path))

    print("-" * 72)
    print("Training finished.")
    print(f"Training time            : {dt:.2f} s")
    print(f"Logged episodes          : {len(logger_cb.episode_rewards)}")
    print(f"Saved PPO model          : {model_path}.zip")

    # Persist metrics
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    out = METRICS_DIR / "section5_ppo_training_metrics.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(
            {
                "episode_rewards": logger_cb.episode_rewards,
                "episode_lengths": logger_cb.episode_lengths,
                "training_time_sec": dt,
                "total_timesteps": total_timesteps,
                "print_frequency": logger_cb.print_freq,
            },
            f,
            indent=2,
        )
    print(f"Saved training metrics   : {out}")
    return model


def load_ppo_model(model_path: Optional[Path] = None):
    """Load a saved PPO model from disk."""
    model_path = model_path or (MODELS_DIR / "ppo_ndvi_drone_final")
    # SB3 accepts path with or without .zip extension
    return PPO.load(str(model_path), device=get_device())
