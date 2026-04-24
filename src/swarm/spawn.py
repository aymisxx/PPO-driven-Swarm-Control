# SWARM SPAWNING

from __future__ import annotations

import numpy as np


def sample_cluster_positions(
    num_agents: int,
    H: int,
    W: int,
    cluster_radius: float,
    min_sep: float,
    rng: np.random.Generator,
    center: tuple[int, int] | None = None,
    max_attempts: int = 10_000,
):
    """
    Controlled random cluster spawning for multi-agent initialization.

    Implements:
    - cluster-based initialization
    - minimum separation constraint
    - rejection sampling
    - optional fixed or random center

    Args:
        num_agents: number of agents
        H, W: field dimensions
        cluster_radius: max deviation from center
        min_sep: minimum pairwise distance
        rng: numpy random generator
        center:
            None → random center
            (y,x) → fixed center (for controlled experiments)
        max_attempts: rejection sampling limit

    Returns:
        positions: list of (y, x)
    """

    # Center selection

    if center is None:
        center_y = int(rng.integers(H // 4, 3 * H // 4))
        center_x = int(rng.integers(W // 4, 3 * W // 4))
    else:
        center_y, center_x = int(center[0]), int(center[1])

    positions: list[tuple[int, int]] = []

    # Rejection sampling

    for i in range(num_agents):

        for _ in range(max_attempts):

            dy = rng.uniform(-cluster_radius, cluster_radius)
            dx = rng.uniform(-cluster_radius, cluster_radius)

            y = int(np.clip(center_y + dy, 0, H - 1))
            x = int(np.clip(center_x + dx, 0, W - 1))

            valid = True

            for (py, px) in positions:
                dist = np.linalg.norm([y - py, x - px])
                if dist < min_sep:
                    valid = False
                    break

            if valid:
                positions.append((y, x))
                break

        else:
            raise RuntimeError(
                f"Failed to sample valid cluster positions "
                f"(agent {i}, attempts exceeded)"
            )

    return positions


# OPTIONAL UTILITY (DEBUG / METRICS)

def compute_pairwise_distances(positions):
    """
    Compute all pairwise distances (for diagnostics)
    """
    dists = []

    for i in range(len(positions)):
        for j in range(i + 1, len(positions)):
            pi = np.array(positions[i])
            pj = np.array(positions[j])
            dists.append(np.linalg.norm(pi - pj))

    return dists