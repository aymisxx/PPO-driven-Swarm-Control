"""Visualization helpers: trajectory plots, role-colored overlays, and GIF export.

The GIF rendering mirrors Section 15 byte-for-byte:
- 60 seconds at 10 fps -> 600 frames target
- step_skip = max(1, max_steps // total_frames)
- trajectory colored by each agent's *current* role
- dotted white edges between agents within comm_radius
"""
from __future__ import annotations

import itertools
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import imageio.v2 as imageio
import matplotlib.pyplot as plt
import numpy as np

from .config import CFG, GIFS_DIR, PLOTS_DIR, ROLE_COLORS_SIMPLE


Position = Tuple[int, int]


# ---------------------------------------------------------------------------
# Static plots
# ---------------------------------------------------------------------------
def plot_trajectories(
    ndvi_field: np.ndarray,
    trajectories: Sequence[Sequence[Position]],
    title: str,
    out_path: Optional[Path] = None,
    show: bool = False,
) -> Path:
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(ndvi_field, cmap="viridis")
    for traj in trajectories:
        t = np.array(traj)
        ax.plot(t[:, 1], t[:, 0], linewidth=1.5, alpha=0.8)
        ax.scatter(
            t[0, 1], t[0, 0],
            s=70, c="white", edgecolors="black",
            linewidths=1.2, marker="o", zorder=5,
        )
        ax.scatter(
            t[-1, 1], t[-1, 0],
            s=90, c="red", marker="x", linewidths=2.0, zorder=6,
        )
    ax.set_title(title)
    ax.invert_yaxis()
    plt.tight_layout()

    if out_path is None:
        PLOTS_DIR.mkdir(parents=True, exist_ok=True)
        out_path = PLOTS_DIR / "swarm_trajectories.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)
    return out_path


def plot_role_colored_trajectories(
    ndvi_field: np.ndarray,
    trajectories: Sequence[Sequence[Position]],
    role_history: Sequence[Sequence[str]],
    title: str = "Full Hybrid Swarm with Role Switching",
    out_path: Optional[Path] = None,
    show: bool = False,
) -> Path:
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(ndvi_field, cmap="viridis")
    for traj, roles_seq in zip(trajectories, role_history):
        t = np.array(traj)
        for k in range(len(t) - 1):
            role = roles_seq[k] if k < len(roles_seq) else roles_seq[-1]
            ax.plot(
                [t[k, 1], t[k + 1, 1]],
                [t[k, 0], t[k + 1, 0]],
                color=ROLE_COLORS_SIMPLE[role],
                linewidth=2,
            )
    ax.set_title(title)
    ax.invert_yaxis()
    plt.tight_layout()

    if out_path is None:
        PLOTS_DIR.mkdir(parents=True, exist_ok=True)
        out_path = PLOTS_DIR / "section14_hybrid_role_trajectories.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)
    return out_path


def plot_role_population(
    role_history: Sequence[Sequence[str]],
    out_path: Optional[Path] = None,
    show: bool = False,
) -> Path:
    from collections import Counter

    num_agents = len(role_history)
    T = len(role_history[0]) if num_agents else 0
    roles = list(ROLE_COLORS_SIMPLE.keys())
    counts = np.zeros((T, len(roles)), dtype=int)
    for t in range(T):
        c = Counter(role_history[i][t] for i in range(num_agents))
        counts[t] = [c[r] for r in roles]

    fig, ax = plt.subplots(figsize=(7, 4))
    for i, r in enumerate(roles):
        ax.plot(counts[:, i], label=r, color=ROLE_COLORS_SIMPLE[r])
    ax.set_title("Role Population Over Time")
    ax.set_xlabel("Time Step")
    ax.set_ylabel("Number of Agents")
    ax.legend()
    plt.tight_layout()

    if out_path is None:
        PLOTS_DIR.mkdir(parents=True, exist_ok=True)
        out_path = PLOTS_DIR / "section14_role_population.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)
    return out_path


# ---------------------------------------------------------------------------
# GIF rendering (Section 15)
# ---------------------------------------------------------------------------
def _render_frame_section15(
    ndvi_field: np.ndarray,
    trajectories: Sequence[Sequence[Position]],
    positions: Sequence[Position],
    roles: Sequence[str],
    step: int,
    comm_radius: float,
) -> np.ndarray:
    """Exact frame composition from Section 15 of the notebook."""
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(ndvi_field, cmap="viridis")

    # Trajectories colored by each agent's *current* role.
    for i, traj in enumerate(trajectories):
        t = np.array(traj)
        role = roles[i]
        color = ROLE_COLORS_SIMPLE[role]
        ax.plot(t[:, 1], t[:, 0], color=color, linewidth=2)
        ax.scatter(t[-1, 1], t[-1, 0], c=color, s=40)

    # Dotted graph edges.
    num_agents = len(positions)
    for i, j in itertools.combinations(range(num_agents), 2):
        pi = positions[i]
        pj = positions[j]
        if np.linalg.norm(np.array(pi) - np.array(pj)) <= comm_radius:
            ax.plot(
                [pi[1], pj[1]],
                [pi[0], pj[0]],
                linestyle="dotted",
                color="white",
                linewidth=1,
            )

    ax.set_title(f"t = {step}")
    ax.axis("off")
    fig.canvas.draw()
    frame = np.array(fig.canvas.renderer.buffer_rgba())[:, :, :3]
    plt.close(fig)
    return frame


def render_hybrid_rollout_gif(
    ppo_model,
    ndvi_field: np.ndarray,
    initial_positions: Sequence[Position],
    out_path: Optional[Path] = None,
    target_seconds: int = CFG.gif_seconds_target,
    fps: int = 10,
    max_steps: int = CFG.max_steps_swarm,
    comm_radius: float = CFG.comm_radius,
) -> Tuple[Path, Dict]:
    """Run the full hybrid rollout and save a 60-second GIF.

    Mirrors Section 15 exactly, including the step_skip derivation
    ``STEP_SKIP = max(1, MAX_STEPS // (target_seconds * fps))``.

    Returns (gif_path, rollout_result_dict).
    """
    from .swarm.controllers import rollout_full_hybrid  # local import avoids cycles

    total_frames = target_seconds * fps
    step_skip = max(1, max_steps // total_frames)
    frames: List[np.ndarray] = []

    def _cb(step_idx: int, live: Dict) -> None:
        if step_idx % step_skip == 0:
            frames.append(
                _render_frame_section15(
                    ndvi_field=ndvi_field,
                    trajectories=live["trajectories"],
                    positions=live["positions"],
                    roles=live["roles"],
                    step=step_idx,
                    comm_radius=comm_radius,
                )
            )

    result = rollout_full_hybrid(
        ppo_model=ppo_model,
        ndvi_field=ndvi_field,
        initial_positions=initial_positions,
        max_steps=max_steps,
        step_callback=_cb,
    )

    GIFS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = out_path or (GIFS_DIR / "final_hybrid_swarm.gif")
    imageio.mimsave(str(out_path), frames, fps=fps)

    return out_path, result
