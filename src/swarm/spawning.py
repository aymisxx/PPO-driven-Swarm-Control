"""Controlled random spawning strategies.

Two strategies are used in the notebook:

1. ``fixed_cluster_spawn`` — Sections 8, 9, 10, 14
   A seeded cluster shape placed at the image center. The *shape* is fixed
   by a dedicated RNG (default BASE_SEED=12345), but the spawn location is
   the geometric center of the image. Used to compare controllers from an
   identical initial condition.

2. ``random_spread_spawn`` — Section 15 (final GIF)
   Random center anywhere in the middle half of the image, with 2x spread
   around it. Uses bare ``np.random`` so each run differs — this is the
   demonstration that the full hybrid controller works from arbitrary starts.
"""
from __future__ import annotations

from typing import List, Tuple

import numpy as np

from ..config import CFG


Position = Tuple[int, int]


def fixed_cluster_spawn(
    ndvi_shape: Tuple[int, int],
    num_agents: int = CFG.num_agents,
    min_sep: int = 40,
    cluster_radius: int = 60,
    base_seed: int = 12345,
) -> List[Position]:
    """Seeded cluster shape, centered on the image (matches notebook Section 8)."""
    H, W = ndvi_shape
    shape_rng = np.random.default_rng(base_seed)

    relative_positions: List[Tuple[float, float]] = []
    for _ in range(num_agents):
        while True:
            dy = shape_rng.uniform(-cluster_radius, cluster_radius)
            dx = shape_rng.uniform(-cluster_radius, cluster_radius)
            valid = True
            for (py, px) in relative_positions:
                if np.linalg.norm([dy - py, dx - px]) < min_sep:
                    valid = False
                    break
            if valid:
                relative_positions.append((dy, dx))
                break

    center_y = H // 2
    center_x = W // 2
    return [
        (
            int(np.clip(center_y + dy, 0, H - 1)),
            int(np.clip(center_x + dx, 0, W - 1)),
        )
        for (dy, dx) in relative_positions
    ]


def random_spread_spawn(
    ndvi_shape: Tuple[int, int],
    num_agents: int = CFG.num_agents,
    spread: float = 2.0,
    box: float = 60.0,
) -> List[Position]:
    """Random center + 2x spread spawn used for the final GIF (Section 15).

    Uses bare ``np.random`` (not a seeded Generator) so each call gives a
    different starting configuration. Do NOT seed globally right before
    calling this if you want run-to-run variation.
    """
    H, W = ndvi_shape
    center_y = np.random.randint(H // 4, 3 * H // 4)
    center_x = np.random.randint(W // 4, 3 * W // 4)

    return [
        (
            int(np.clip(center_y + spread * np.random.uniform(-box, box), 0, H - 1)),
            int(np.clip(center_x + spread * np.random.uniform(-box, box), 0, W - 1)),
        )
        for _ in range(num_agents)
    ]
