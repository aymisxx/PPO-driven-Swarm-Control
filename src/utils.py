"""Shared utilities: reproducibility seeding, device detection, geometry helpers."""
from __future__ import annotations

import itertools
import random
from typing import Iterable, List, Sequence, Tuple

import numpy as np
import torch


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
def seed_everything(seed: int) -> None:
    """Seed python/random, numpy, torch, and stable-baselines3.

    Does not eliminate all GPU nondeterminism but gives a controlled baseline.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    try:
        from stable_baselines3.common.utils import set_random_seed
        set_random_seed(seed)
    except Exception:
        # SB3 optional at import time for pure eval/vis tasks.
        pass

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------
# Geometry helpers used by every swarm controller
# ---------------------------------------------------------------------------
Position = Tuple[int, int]  # (y, x)


def compute_distances(positions: Sequence[Position]) -> List[float]:
    """All unordered pairwise Euclidean distances between agents."""
    return [
        float(np.linalg.norm(np.array(positions[i]) - np.array(positions[j])))
        for i, j in itertools.combinations(range(len(positions)), 2)
    ]


def get_neighbors(positions: Sequence[Position], i: int, comm_radius: float) -> List[int]:
    """Indices of agents within comm_radius of agent i."""
    pi = np.array(positions[i])
    return [
        j
        for j, pj in enumerate(positions)
        if i != j and np.linalg.norm(pi - np.array(pj)) <= comm_radius
    ]


def action_to_direction(action: int) -> np.ndarray:
    """Map discrete action -> unit direction vector (y, x order).

    0 = up, 1 = right, 2 = down, 3 = left.
    """
    table = {
        0: np.array([-1, 0]),
        1: np.array([0, 1]),
        2: np.array([1, 0]),
        3: np.array([0, -1]),
    }
    if action not in table:
        raise ValueError(f"Invalid action: {action}")
    return table[action]
