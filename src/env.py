"""Single-agent local-observation environment (Sections 2-4 of the notebook)."""
from __future__ import annotations

from typing import Optional

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from .config import CFG


class NDVIDroneEnv(gym.Env):
    """Single-agent environment over a VARI-derived utility field.

    Observation:
        Local patch of shape (1, patch_size, patch_size), dtype uint8.

    Action space:
        0 = up, 1 = right, 2 = down, 3 = left.

    Reward:
        First-visit utility value (phi(c_k) if cell not yet visited, else 0).

    Termination:
        Fixed time horizon via truncation at max_steps.
    """

    metadata = {"render_modes": ["rgb_array"], "render_fps": 5}

    def __init__(
        self,
        ndvi_field: np.ndarray,
        patch_size: int = 128,
        max_steps: int = 300,
        action_step_px: int = 1,
        spawn_margin: Optional[int] = None,
    ):
        super().__init__()
        assert ndvi_field.ndim == 2, "ndvi_field must be a 2D array."
        self.ndvi_field = ndvi_field.astype(np.float32)
        self.H, self.W = self.ndvi_field.shape
        self.patch_size = int(patch_size)
        self.max_steps = int(max_steps)
        self.action_step_px = int(action_step_px)
        self.pad = self.patch_size // 2

        if spawn_margin is None:
            spawn_margin = self.pad
        self.spawn_margin = int(spawn_margin)

        # Keep spawn margin valid even for smaller images.
        self.y_low = min(max(self.spawn_margin, 0), self.H - 1)
        self.y_high = max(min(self.H - self.spawn_margin, self.H), self.y_low + 1)
        self.x_low = min(max(self.spawn_margin, 0), self.W - 1)
        self.x_high = max(min(self.W - self.spawn_margin, self.W), self.x_low + 1)

        self.ndvi_padded = np.pad(
            self.ndvi_field,
            pad_width=self.pad,
            mode="constant",
            constant_values=0.0,
        )

        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=(1, self.patch_size, self.patch_size),
            dtype=np.uint8,
        )

        self.agent_x: Optional[int] = None
        self.agent_y: Optional[int] = None
        self.visited: Optional[np.ndarray] = None
        self.step_count = 0
        self.last_spawn_seed = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _random_spawn(self) -> None:
        self.agent_y = int(self.np_random.integers(self.y_low, self.y_high))
        self.agent_x = int(self.np_random.integers(self.x_low, self.x_high))

    def _get_obs(self) -> np.ndarray:
        y, x = self.agent_y, self.agent_x
        yp = y + self.pad
        xp = x + self.pad
        p = self.patch_size
        patch = self.ndvi_padded[
            yp - p // 2 : yp - p // 2 + p,
            xp - p // 2 : xp - p // 2 + p,
        ]
        patch_uint8 = np.clip(patch * 255.0, 0, 255).astype(np.uint8)
        return patch_uint8[None, ...]

    # ------------------------------------------------------------------
    # Gymnasium API
    # ------------------------------------------------------------------
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.last_spawn_seed = seed
        self.step_count = 0
        self.visited = np.zeros((self.H, self.W), dtype=bool)
        self._random_spawn()
        self.visited[self.agent_y, self.agent_x] = True

        obs = self._get_obs()
        info = {
            "agent_pos": (self.agent_y, self.agent_x),
            "spawn_seed": seed,
            "spawn_is_random": True,
            "spawn_margin": self.spawn_margin,
            "step_count": self.step_count,
            "utility_value": float(self.ndvi_field[self.agent_y, self.agent_x]),
        }
        return obs, info

    def step(self, action):
        action = int(action)
        if action == 0:    # up
            self.agent_y -= self.action_step_px
        elif action == 1:  # right
            self.agent_x += self.action_step_px
        elif action == 2:  # down
            self.agent_y += self.action_step_px
        elif action == 3:  # left
            self.agent_x -= self.action_step_px
        else:
            raise ValueError(f"Invalid action: {action}")

        self.agent_y = int(np.clip(self.agent_y, 0, self.H - 1))
        self.agent_x = int(np.clip(self.agent_x, 0, self.W - 1))

        y, x = self.agent_y, self.agent_x
        first_visit = not self.visited[y, x]
        reward = float(self.ndvi_field[y, x]) if first_visit else 0.0
        self.visited[y, x] = True

        self.step_count += 1
        terminated = False
        truncated = self.step_count >= self.max_steps

        obs = self._get_obs()
        info = {
            "agent_pos": (y, x),
            "step_count": self.step_count,
            "first_visit": first_visit,
            "utility_value": float(self.ndvi_field[y, x]),
            "unique_visited": int(self.visited.sum()),
        }
        return obs, reward, terminated, truncated, info

    def render(self):
        import matplotlib.pyplot as plt
        import matplotlib.patches as patches

        fig, ax = plt.subplots(figsize=(6, 6), dpi=120)
        ax.imshow(self.ndvi_field, cmap="viridis", vmin=0.0, vmax=1.0)
        ax.axis("off")
        ax.scatter(
            [self.agent_x], [self.agent_y],
            s=45, c="white", edgecolors="black",
            linewidths=1.0, zorder=4,
        )
        if CFG.show_patch_windows:
            half = self.patch_size // 2
            rect = patches.Rectangle(
                (self.agent_x - half, self.agent_y - half),
                self.patch_size, self.patch_size,
                linewidth=1.8, edgecolor="red", facecolor="none", zorder=3,
            )
            ax.add_patch(rect)
        fig.canvas.draw()
        rgba = np.asarray(fig.canvas.buffer_rgba())
        rgb_frame = rgba[:, :, :3].copy()
        plt.close(fig)
        return rgb_frame


# ---------------------------------------------------------------------------
# Vectorized-env factory for Stable-Baselines3 (Sections 4-5)
# ---------------------------------------------------------------------------
def make_env_factory(
    ndvi_field: np.ndarray,
    seed: Optional[int] = None,
    monitor: bool = False,
):
    """Return a zero-arg factory for DummyVecEnv."""
    def _init():
        env_local = NDVIDroneEnv(
            ndvi_field=ndvi_field,
            patch_size=CFG.patch_size,
            max_steps=CFG.max_steps_single,
            action_step_px=CFG.action_step_px,
            spawn_margin=CFG.spawn_margin,
        )
        if monitor:
            from stable_baselines3.common.monitor import Monitor
            env_local = Monitor(env_local)
        env_local.reset(seed=seed)
        return env_local
    return _init
