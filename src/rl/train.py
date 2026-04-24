# PPO TRAINING SCRIPT

from __future__ import annotations

import time
import json

import numpy as np

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback

from src.config import CFG, SEED, DEVICE, MODELS_DIR, METRICS_DIR
from src.env.ndvi_env import NDVIDroneEnv


# CALLBACK

class SparseEpisodeLoggingCallback(BaseCallback):

    def __init__(self, print_freq: int = 40_000):
        super().__init__()
        self.print_freq = print_freq
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

            if len(self.episode_rewards) > 0:
                r = np.mean(self.episode_rewards[-20:])
                l = np.mean(self.episode_lengths[-20:])
                print(f"[PPO] steps={current_step} | reward={r:.2f} | len={l:.1f}")

        return True


# ENV FACTORY

def make_env(ndvi_field):

    def _init():
        env = NDVIDroneEnv(
            ndvi_field=ndvi_field,
            patch_size=CFG.patch_size,
            max_steps=CFG.max_steps_single,
            action_step_px=CFG.action_step_px,
            spawn_margin=CFG.spawn_margin,
        )
        env = Monitor(env)
        env.reset(seed=SEED)
        return env

    return _init


# TRAIN FUNCTION

def train_ppo(ndvi_field):

    vec_env = DummyVecEnv([make_env(ndvi_field)])

    model = PPO(
        policy="CnnPolicy",
        env=vec_env,
        learning_rate=CFG.learning_rate,
        gamma=CFG.gamma,
        n_steps=CFG.ppo_n_steps,
        batch_size=CFG.ppo_batch_size,
        n_epochs=CFG.ppo_n_epochs,
        verbose=0,
        device=DEVICE,
    )

    callback = SparseEpisodeLoggingCallback()

    print("Training PPO...")

    start = time.time()

    model.learn(
        total_timesteps=CFG.total_timesteps,
        callback=callback,
        progress_bar=False,
    )

    end = time.time()

    print(f"Training finished in {end - start:.2f}s")

    # SAVE MODEL

    model_path = MODELS_DIR / "ppo_ndvi_drone_final"
    model.save(str(model_path))

    print(f"Model saved at: {model_path}.zip")

    # SAVE METRICS

    metrics = {
        "episode_rewards": callback.episode_rewards,
        "episode_lengths": callback.episode_lengths,
        "training_time": end - start,
    }

    metrics_path = METRICS_DIR / "ppo_training_metrics.json"

    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"Metrics saved at: {metrics_path}")

    return model