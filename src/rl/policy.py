# PPO POLICY INTERFACE

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
from stable_baselines3 import PPO

from src.config import MODELS_DIR

# LOAD MODEL

def load_ppo_model(model_name: str = "ppo_ndvi_drone_final") -> PPO:
    """
    Load trained PPO model from models directory.

    NOTE:
    - No training here
    - Only inference usage
    """

    model_path = MODELS_DIR / model_name

    if not model_path.with_suffix(".zip").exists():
        raise FileNotFoundError(
            f"PPO model not found at: {model_path}.zip\n"
            "Train model first or place pretrained model."
        )

    model = PPO.load(str(model_path))

    return model


# ACTION INFERENCE

def predict_action(
    model: PPO,
    obs: np.ndarray,
    deterministic: bool = True,
) -> int:
    """
    Predict action from observation.

    Input:
        obs: (1, P, P) uint8

    Output:
        action: int in {0,1,2,3}
    """

    action, _ = model.predict(obs, deterministic=deterministic)

    return int(action)


# ACTION → DIRECTION MAPPING

def action_to_direction(action: int) -> np.ndarray:
    """
    Convert discrete action to 2D direction vector.

    Mapping:
        0 = up    → (-1, 0)
        1 = right → (0, 1)
        2 = down  → (1, 0)
        3 = left  → (0, -1)
    """

    if action == 0:
        return np.array([-1, 0], dtype=np.float32)

    elif action == 1:
        return np.array([0, 1], dtype=np.float32)

    elif action == 2:
        return np.array([1, 0], dtype=np.float32)

    elif action == 3:
        return np.array([0, -1], dtype=np.float32)

    else:
        raise ValueError(f"Invalid action: {action}")