"""Single-agent evaluation and PPO-vs-random diagnostic (Sections 6, 7)."""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

import numpy as np

from .config import CFG, METRICS_DIR
from .env import NDVIDroneEnv


def _run_single_agent_rollout(
    ppo_model,
    ndvi_field: np.ndarray,
    episode_seed: Optional[int],
    policy_kind: str = "ppo",
    deterministic: bool = True,
) -> Dict:
    env = NDVIDroneEnv(
        ndvi_field=ndvi_field,
        patch_size=CFG.patch_size,
        max_steps=CFG.max_steps_single,
        action_step_px=CFG.action_step_px,
        spawn_margin=CFG.spawn_margin,
    )
    obs, _ = env.reset(seed=episode_seed)
    if episode_seed is not None:
        env.action_space.seed(episode_seed + 12345)

    trajectory = [(int(env.agent_y), int(env.agent_x))]
    rewards, utilities, first_visit_flags, actions = [], [], [], []

    for _ in range(CFG.max_steps_single):
        if policy_kind == "ppo":
            action, _ = ppo_model.predict(obs, deterministic=deterministic)
            action = int(action)
        elif policy_kind == "random":
            action = int(env.action_space.sample())
        else:
            raise ValueError(f"Unknown policy_kind: {policy_kind}")

        obs, reward, terminated, truncated, info = env.step(action)
        trajectory.append((int(env.agent_y), int(env.agent_x)))
        rewards.append(float(reward))
        utilities.append(float(info["utility_value"]))
        first_visit_flags.append(bool(info["first_visit"]))
        actions.append(action)
        if terminated or truncated:
            break

    traj = np.array(trajectory, dtype=np.int32)
    rew = np.array(rewards, dtype=np.float32)
    utils = np.array(utilities, dtype=np.float32)
    fv = np.array(first_visit_flags, dtype=bool)

    path_length = float(np.sum(np.linalg.norm(np.diff(traj[:, ::-1], axis=0), axis=1)))
    net_disp = float(np.linalg.norm(traj[-1, ::-1] - traj[0, ::-1]))
    path_eff = float(net_disp / path_length) if path_length > 0 else 0.0

    return {
        "policy": policy_kind,
        "trajectory": traj,
        "rewards": rew,
        "utilities": utils,
        "first_visit": fv,
        "actions": actions,
        "unique_visited": int(env.visited.sum()),
        "total_reward": float(rew.sum()),
        "mean_utility": float(utils.mean()) if len(utils) else 0.0,
        "end_utility": float(utils[-1]) if len(utils) else 0.0,
        "path_length_px": path_length,
        "net_displacement_px": net_disp,
        "path_efficiency": path_eff,
    }


def evaluate_single_agent(
    ppo_model,
    ndvi_field: np.ndarray,
    seed: Optional[int] = None,
    save_metrics: bool = True,
) -> Dict:
    """Run one deterministic PPO rollout from a random spawn (Section 6)."""
    rollout = _run_single_agent_rollout(
        ppo_model=ppo_model,
        ndvi_field=ndvi_field,
        episode_seed=seed,
        policy_kind="ppo",
        deterministic=True,
    )

    H, W = ndvi_field.shape
    rollout["coverage_ratio"] = rollout["unique_visited"] / (H * W)

    if save_metrics:
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        METRICS_DIR.mkdir(parents=True, exist_ok=True)
        out = METRICS_DIR / f"section6_single_agent_ppo_eval_metrics_{run_id}.json"
        payload = {
            "run_id": run_id,
            "spawn_mode": "fresh_random_unseeded" if seed is None else "seeded",
            "steps_executed": int(len(rollout["rewards"])),
            "start_position_yx": rollout["trajectory"][0].tolist(),
            "end_position_yx": rollout["trajectory"][-1].tolist(),
            "total_first_visit_reward": rollout["total_reward"],
            "unique_visited_cells": rollout["unique_visited"],
            "coverage_ratio": rollout["coverage_ratio"],
            "first_visit_steps": int(rollout["first_visit"].sum()),
            "path_length_px": rollout["path_length_px"],
            "net_displacement_px": rollout["net_displacement_px"],
            "path_efficiency": rollout["path_efficiency"],
        }
        with open(out, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        rollout["metrics_path"] = str(out)
    return rollout


def ppo_vs_random_diagnostic(
    ppo_model,
    ndvi_field: np.ndarray,
    num_episodes: int = 20,
    base_seed: Optional[int] = None,
    save_metrics: bool = True,
) -> Dict:
    """Paired PPO/Random comparison from matched random starts (Section 7)."""
    if base_seed is None:
        base_seed = int(np.random.default_rng().integers(0, 2_000_000_000))

    ppo_list, rand_list = [], []
    for idx in range(num_episodes):
        ep_seed = base_seed + idx
        ppo_list.append(_run_single_agent_rollout(ppo_model, ndvi_field, ep_seed, "ppo", False))
        rand_list.append(_run_single_agent_rollout(ppo_model, ndvi_field, ep_seed, "random", False))

    def _mean(lst, key):
        return float(np.mean([x[key] for x in lst]))

    summary = {
        "base_seed": int(base_seed),
        "num_episodes": int(num_episodes),
        "ppo": {
            "mean_total_reward": _mean(ppo_list, "total_reward"),
            "mean_unique_visited": _mean(ppo_list, "unique_visited"),
            "mean_end_utility": _mean(ppo_list, "end_utility"),
            "mean_path_efficiency": _mean(ppo_list, "path_efficiency"),
        },
        "random": {
            "mean_total_reward": _mean(rand_list, "total_reward"),
            "mean_unique_visited": _mean(rand_list, "unique_visited"),
            "mean_end_utility": _mean(rand_list, "end_utility"),
            "mean_path_efficiency": _mean(rand_list, "path_efficiency"),
        },
    }

    if save_metrics:
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        METRICS_DIR.mkdir(parents=True, exist_ok=True)
        out = METRICS_DIR / f"section7_ppo_vs_random_{run_id}.json"
        with open(out, "w", encoding="utf-8") as f:
            json.dump({**summary, "run_id": run_id}, f, indent=2)
        summary["metrics_path"] = str(out)

    return summary
