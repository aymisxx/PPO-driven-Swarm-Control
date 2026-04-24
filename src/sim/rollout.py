# SWARM ROLLOUT ENGINE

from __future__ import annotations

import numpy as np

from src.swarm.spawn import sample_cluster_positions
from src.swarm.apf import compute_k_rep_eff
from src.swarm.controller import compute_all_controls
from src.swarm.roles import (
    initialize_roles,
    update_roles,
)
from src.rl.policy import predict_action, action_to_direction


def run_swarm_rollout(
    ndvi_field: np.ndarray,
    model,
    num_agents: int,
    cluster_radius: float,
    min_sep: float,
    max_steps: int,
    R_rep: float,
    k_rep: float,
    R_comm: float,
    k_cons: float,
    role_switch_prob: float = 0.05,
    rng: np.random.Generator | None = None,
):
    """
    Full hybrid swarm rollout (pipeline-consistent)

    IMPORTANT:
    Motion is DISCRETE GRID-BASED, not continuous.
    """

    if rng is None:
        rng = np.random.default_rng()

    H, W = ndvi_field.shape


    # SPAWN

    positions = sample_cluster_positions(
        num_agents=num_agents,
        H=H,
        W=W,
        cluster_radius=cluster_radius,
        min_sep=min_sep,
        rng=rng,
    )

    trajectories = [[p] for p in positions]


    # STATE

    visited = np.zeros_like(ndvi_field, dtype=bool)
    visited_once = np.zeros_like(ndvi_field, dtype=bool)

    roles = initialize_roles(num_agents)
    role_history = [[] for _ in range(num_agents)]

    # Pre-pad once (performance)
    pad = 64
    padded = np.pad(ndvi_field, pad, mode="constant")


    # ROLLOUT LOOP

    for step in range(max_steps):

        directions = []


        # PPO DIRECTIONS

        for (y, x) in positions:

            yp = y + pad
            xp = x + pad

            patch = padded[
                yp - pad : yp + pad,
                xp - pad : xp + pad,
            ]

            obs = (patch * 255.0).clip(0, 255).astype(np.uint8)[None, ...]

            action = predict_action(model, obs, deterministic=False)
            d = action_to_direction(action)

            directions.append(d)

        directions = np.array(directions)


        # APF scaling

        k_rep_eff = compute_k_rep_eff(k_rep, R_rep, positions)


        # HYBRID CONTROLLER

        controls = compute_all_controls(
            positions,
            directions,
            roles,
            R_rep,
            k_rep_eff,
            R_comm,
            k_cons,
        )


        # UPDATE POSITIONS

        new_positions = []

        for i, (y, x) in enumerate(positions):

            u = controls[i]

            # 🔥 CRITICAL FIX: DISCRETE MOTION
            dy = int(np.sign(u[0]))
            dx = int(np.sign(u[1]))

            y_new = int(np.clip(y + dy, 0, H - 1))
            x_new = int(np.clip(x + dx, 0, W - 1))

            # NDVI tracking
            if not visited_once[y_new, x_new]:
                visited_once[y_new, x_new] = True

            visited[y_new, x_new] = True

            new_positions.append((y_new, x_new))
            trajectories[i].append((y_new, x_new))

            # store role at this step
            role_history[i].append(roles[i])

        positions = new_positions


        
        # ROLE SWITCHING

        
        roles = update_roles(roles, role_switch_prob)

    return {
        "trajectories": trajectories,
        "role_history": role_history,
        "visited": visited,
    }