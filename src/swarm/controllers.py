"""Swarm rollout controllers (Sections 8, 9, 10, 14, 15).

Each rollout function returns a dict with:
    trajectories        : list of lists of (y, x) tuples (len = num_agents)
    role_history        : list of lists of role strings (only for full hybrid)
    visited_global      : HxW bool array of all touched cells
    visited_once        : HxW bool array of first-visit cells
    ndvi_gain           : sum of phi over first visits
    min_dist_over_time  : per-step min pairwise distance
    pairwise_distances  : flat list of all pairwise distances over all steps
    close_encounters    : count of (d < safe_dist) pairs summed over steps
    edge_counts         : per-step total graph-edge count (consensus variants)

The physics constants match the notebook byte-for-byte:
    R_REP = 60, K_REP = 1.75, R_COMM = 120, K_CONS = 0.35, SAFE_DIST = 20.
The PPO direction is sampled stochastically (deterministic=False) for all
swarm variants, matching the notebook.
"""
from __future__ import annotations

import itertools
import random
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..config import CFG, ROLE_SWITCH_PROB, ROLE_WEIGHTS
from ..env import NDVIDroneEnv
from ..utils import action_to_direction, compute_distances, get_neighbors
from .roles import ROLES, transition_role


Position = Tuple[int, int]
SAFE_DIST = 20  # collision-safety threshold used in Sections 8-14


# ---------------------------------------------------------------------------
# Helpers shared across controllers
# ---------------------------------------------------------------------------
def _ppo_direction(
    ppo_model,
    ndvi_field: np.ndarray,
    visited_global: np.ndarray,
    y: int,
    x: int,
    deterministic: bool = False,
) -> np.ndarray:
    """Query the trained PPO policy at a point and return a unit direction.

    Mirrors the notebook's use of a throwaway NDVIDroneEnv to grab the
    local observation at (y, x) before calling predict().
    """
    temp_env = NDVIDroneEnv(
        ndvi_field=ndvi_field,
        patch_size=CFG.patch_size,
        max_steps=1,
        action_step_px=CFG.action_step_px,
        spawn_margin=CFG.spawn_margin,
    )
    temp_env.agent_y = y
    temp_env.agent_x = x
    temp_env.visited = visited_global
    obs = temp_env._get_obs()
    action, _ = ppo_model.predict(obs, deterministic=deterministic)
    return action_to_direction(int(action))


def _effective_k_rep(pairwise: Sequence[float], r_rep: float, k_rep: float) -> float:
    d_min = min(pairwise) if pairwise else 1.0
    if d_min < r_rep:
        return k_rep * (r_rep / (d_min + 1e-6))
    return k_rep


def _repulsion_force(
    positions: Sequence[Position],
    i: int,
    r_rep: float,
    k_rep_eff: float,
) -> np.ndarray:
    y, x = positions[i]
    F = np.zeros(2)
    for j, (yj, xj) in enumerate(positions):
        if i == j:
            continue
        diff = np.array([y - yj, x - xj])
        dist = float(np.linalg.norm(diff))
        if r_rep > dist > 1e-6:
            psi = max(0.0, (1.0 / dist) - (1.0 / r_rep))
            F = F + k_rep_eff * psi * (diff / (dist + 1e-6))
    return F


def _empty_rollout_buffers(
    ndvi_field: np.ndarray,
    initial_positions: Sequence[Position],
    track_roles: bool = False,
):
    trajectories = [[p] for p in initial_positions]
    visited_global = np.zeros_like(ndvi_field, dtype=bool)
    visited_once = np.zeros_like(ndvi_field, dtype=bool)
    role_history = [[] for _ in initial_positions] if track_roles else None
    return trajectories, visited_global, visited_once, role_history


def _summarize(result: Dict, ndvi_shape: Tuple[int, int]) -> None:
    """Populate scalar metrics into the result dict in-place."""
    H, W = ndvi_shape
    unique_visited = int(result["visited_global"].sum())
    unique_once = int(result["visited_once"].sum())
    result["unique_visited"] = unique_visited
    result["unique_once"] = unique_once
    result["coverage_ratio"] = unique_visited / (H * W)
    result["redundancy"] = 1.0 - (unique_once / max(1, unique_visited))
    if result["pairwise_distances"]:
        result["mean_pairwise_distance"] = float(np.mean(result["pairwise_distances"]))
    else:
        result["mean_pairwise_distance"] = 0.0
    result["min_distance"] = (
        float(min(result["min_dist_over_time"])) if result["min_dist_over_time"] else 0.0
    )


# ---------------------------------------------------------------------------
# Section 8 — Naive multi-agent PPO (no coordination)
# ---------------------------------------------------------------------------
def rollout_naive_ppo(
    ppo_model,
    ndvi_field: np.ndarray,
    initial_positions: Sequence[Position],
    max_steps: int = CFG.max_steps_swarm,
) -> Dict:
    """Pure policy replication. No APF, no consensus, no roles."""
    H, W = ndvi_field.shape
    trajectories, visited_global, visited_once, _ = _empty_rollout_buffers(
        ndvi_field, initial_positions, track_roles=False
    )

    positions = list(initial_positions)
    ndvi_gain = 0.0
    min_dist_over_time: List[float] = []
    pairwise_distances: List[float] = []
    close_encounters = 0

    for _ in range(max_steps):
        new_positions: List[Position] = []
        for i, (y, x) in enumerate(positions):
            d = _ppo_direction(ppo_model, ndvi_field, visited_global, y, x)
            y_new = int(np.clip(y + d[0], 0, H - 1))
            x_new = int(np.clip(x + d[1], 0, W - 1))

            if not visited_once[y_new, x_new]:
                ndvi_gain += float(ndvi_field[y_new, x_new])
                visited_once[y_new, x_new] = True

            new_positions.append((y_new, x_new))
            trajectories[i].append((y_new, x_new))
            visited_global[y_new, x_new] = True

        positions = new_positions
        dists = compute_distances(positions)
        if dists:
            min_dist_over_time.append(min(dists))
            pairwise_distances.extend(dists)
            close_encounters += sum(d < SAFE_DIST for d in dists)

    result = {
        "trajectories": trajectories,
        "role_history": None,
        "visited_global": visited_global,
        "visited_once": visited_once,
        "ndvi_gain": ndvi_gain,
        "min_dist_over_time": min_dist_over_time,
        "pairwise_distances": pairwise_distances,
        "close_encounters": close_encounters,
        "edge_counts": [],
    }
    _summarize(result, ndvi_field.shape)
    return result


# ---------------------------------------------------------------------------
# Section 9 — PPO + repulsive APF
# ---------------------------------------------------------------------------
def rollout_ppo_repulsion(
    ppo_model,
    ndvi_field: np.ndarray,
    initial_positions: Sequence[Position],
    max_steps: int = CFG.max_steps_swarm,
    r_rep: float = CFG.repulsion_radius,
    k_rep: float = CFG.k_rep,
) -> Dict:
    """PPO direction combined with pairwise repulsion only."""
    H, W = ndvi_field.shape
    trajectories, visited_global, visited_once, _ = _empty_rollout_buffers(
        ndvi_field, initial_positions, track_roles=False
    )

    positions = list(initial_positions)
    ndvi_gain = 0.0
    min_dist_over_time: List[float] = []
    pairwise_distances: List[float] = []
    close_encounters = 0

    for _ in range(max_steps):
        dists = compute_distances(positions)
        k_rep_eff = _effective_k_rep(dists, r_rep, k_rep)
        new_positions: List[Position] = []

        for i, (y, x) in enumerate(positions):
            d_ppo = _ppo_direction(ppo_model, ndvi_field, visited_global, y, x)
            F_rep = _repulsion_force(positions, i, r_rep, k_rep_eff)
            u = d_ppo + F_rep
            norm = float(np.linalg.norm(u))
            if norm > 1e-6:
                u = u / norm

            y_new = int(np.clip(y + u[0], 0, H - 1))
            x_new = int(np.clip(x + u[1], 0, W - 1))

            if not visited_once[y_new, x_new]:
                ndvi_gain += float(ndvi_field[y_new, x_new])
                visited_once[y_new, x_new] = True

            new_positions.append((y_new, x_new))
            trajectories[i].append((y_new, x_new))
            visited_global[y_new, x_new] = True

        positions = new_positions
        dists_after = compute_distances(positions)
        if dists_after:
            min_dist_over_time.append(min(dists_after))
            pairwise_distances.extend(dists_after)
            close_encounters += sum(d < SAFE_DIST for d in dists_after)

    result = {
        "trajectories": trajectories,
        "role_history": None,
        "visited_global": visited_global,
        "visited_once": visited_once,
        "ndvi_gain": ndvi_gain,
        "min_dist_over_time": min_dist_over_time,
        "pairwise_distances": pairwise_distances,
        "close_encounters": close_encounters,
        "edge_counts": [],
    }
    _summarize(result, ndvi_field.shape)
    return result


# ---------------------------------------------------------------------------
# Section 10 — PPO + repulsion + graph-based consensus
# ---------------------------------------------------------------------------
def rollout_ppo_repulsion_consensus(
    ppo_model,
    ndvi_field: np.ndarray,
    initial_positions: Sequence[Position],
    max_steps: int = CFG.max_steps_swarm,
    r_rep: float = CFG.repulsion_radius,
    k_rep: float = CFG.k_rep,
    r_comm: float = CFG.comm_radius,
    k_cons: float = CFG.k_cons,
) -> Dict:
    H, W = ndvi_field.shape
    trajectories, visited_global, visited_once, _ = _empty_rollout_buffers(
        ndvi_field, initial_positions, track_roles=False
    )

    positions = list(initial_positions)
    ndvi_gain = 0.0
    min_dist_over_time: List[float] = []
    pairwise_distances: List[float] = []
    close_encounters = 0
    edge_counts: List[int] = []

    for _ in range(max_steps):
        # 1. Query PPO directions for every agent first (matches notebook).
        directions = np.array([
            _ppo_direction(ppo_model, ndvi_field, visited_global, y, x)
            for (y, x) in positions
        ])

        # 2. Compute adaptive repulsion gain from current spacing.
        dists = compute_distances(positions)
        k_rep_eff = _effective_k_rep(dists, r_rep, k_rep)

        edge_count = 0
        new_positions: List[Position] = []
        for i, (y, x) in enumerate(positions):
            d_ppo = directions[i]
            F_rep = _repulsion_force(positions, i, r_rep, k_rep_eff)

            neighbors = get_neighbors(positions, i, r_comm)
            edge_count += len(neighbors)
            u_cons = np.zeros(2)
            for j in neighbors:
                u_cons = u_cons + (directions[j] - directions[i])
            u_cons = k_cons * u_cons

            u = d_ppo + F_rep + u_cons
            norm = float(np.linalg.norm(u))
            if norm > 1e-6:
                u = u / norm

            y_new = int(np.clip(y + u[0], 0, H - 1))
            x_new = int(np.clip(x + u[1], 0, W - 1))

            if not visited_once[y_new, x_new]:
                ndvi_gain += float(ndvi_field[y_new, x_new])
                visited_once[y_new, x_new] = True

            new_positions.append((y_new, x_new))
            trajectories[i].append((y_new, x_new))
            visited_global[y_new, x_new] = True

        positions = new_positions
        edge_counts.append(edge_count)
        dists_after = compute_distances(positions)
        if dists_after:
            min_dist_over_time.append(min(dists_after))
            pairwise_distances.extend(dists_after)
            close_encounters += sum(d < SAFE_DIST for d in dists_after)

    result = {
        "trajectories": trajectories,
        "role_history": None,
        "visited_global": visited_global,
        "visited_once": visited_once,
        "ndvi_gain": ndvi_gain,
        "min_dist_over_time": min_dist_over_time,
        "pairwise_distances": pairwise_distances,
        "close_encounters": close_encounters,
        "edge_counts": edge_counts,
    }
    _summarize(result, ndvi_field.shape)
    return result


# ---------------------------------------------------------------------------
# Section 14/15 — Full hybrid (roles + repulsion + consensus)
# ---------------------------------------------------------------------------
def rollout_full_hybrid(
    ppo_model,
    ndvi_field: np.ndarray,
    initial_positions: Sequence[Position],
    max_steps: int = CFG.max_steps_swarm,
    r_rep: float = CFG.repulsion_radius,
    k_rep: float = CFG.k_rep,
    r_comm: float = CFG.comm_radius,
    k_cons: float = CFG.k_cons,
    role_switch_prob: float = ROLE_SWITCH_PROB,
    step_callback: Optional[Callable[[int, Dict], None]] = None,
) -> Dict:
    """Full hybrid controller with stochastic role switching.

    step_callback, if provided, is invoked at the end of every step as
    step_callback(step_idx, live_state_dict) — used by the GIF renderer
    to draw frames without duplicating the rollout loop.
    """
    H, W = ndvi_field.shape
    trajectories, visited_global, visited_once, role_history = _empty_rollout_buffers(
        ndvi_field, initial_positions, track_roles=True
    )

    positions = list(initial_positions)
    roles = [random.choice(ROLES) for _ in range(len(positions))]

    ndvi_gain = 0.0
    min_dist_over_time: List[float] = []
    pairwise_distances: List[float] = []
    close_encounters = 0
    edge_counts: List[int] = []

    for step_idx in range(max_steps):
        # 1. PPO directions (stochastic, matches notebook).
        directions = np.array([
            _ppo_direction(ppo_model, ndvi_field, visited_global, y, x)
            for (y, x) in positions
        ])

        # 2. Adaptive repulsion gain.
        dists = compute_distances(positions)
        k_rep_eff = _effective_k_rep(dists, r_rep, k_rep)

        edge_count = 0
        new_positions: List[Position] = []

        for i, (y, x) in enumerate(positions):
            role = roles[i]
            role_history[i].append(role)
            w = ROLE_WEIGHTS[role]

            d_ppo = directions[i] * w["ppo"]
            F_rep = _repulsion_force(positions, i, r_rep, k_rep_eff) * w["rep"]

            neighbors = get_neighbors(positions, i, r_comm)
            edge_count += len(neighbors)
            u_cons = np.zeros(2)
            for j in neighbors:
                u_cons = u_cons + (directions[j] - directions[i])
            u_cons = u_cons * (w["cons"] * k_cons)

            u = d_ppo + F_rep + u_cons
            norm = float(np.linalg.norm(u))
            if norm > 1e-6:
                u = u / norm

            y_new = int(np.clip(y + u[0], 0, H - 1))
            x_new = int(np.clip(x + u[1], 0, W - 1))

            if not visited_once[y_new, x_new]:
                ndvi_gain += float(ndvi_field[y_new, x_new])
                visited_once[y_new, x_new] = True

            new_positions.append((y_new, x_new))
            trajectories[i].append((y_new, x_new))
            visited_global[y_new, x_new] = True

            # Stochastic role switch (matches notebook: uses role pre-update).
            if random.random() < role_switch_prob:
                roles[i] = transition_role(role)

        positions = new_positions
        edge_counts.append(edge_count)

        dists_after = compute_distances(positions)
        if dists_after:
            min_dist_over_time.append(min(dists_after))
            pairwise_distances.extend(dists_after)
            close_encounters += sum(d < SAFE_DIST for d in dists_after)

        if step_callback is not None:
            step_callback(
                step_idx,
                {
                    "positions": positions,
                    "roles": roles,
                    "trajectories": trajectories,
                    "role_history": role_history,
                    "visited_global": visited_global,
                    "edge_count": edge_count,
                },
            )

    result = {
        "trajectories": trajectories,
        "role_history": role_history,
        "final_roles": roles,
        "visited_global": visited_global,
        "visited_once": visited_once,
        "ndvi_gain": ndvi_gain,
        "min_dist_over_time": min_dist_over_time,
        "pairwise_distances": pairwise_distances,
        "close_encounters": close_encounters,
        "edge_counts": edge_counts,
    }
    _summarize(result, ndvi_field.shape)
    return result
