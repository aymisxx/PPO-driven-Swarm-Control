# GIF GENERATION

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import itertools
import imageio

from src.swarm.roles import ROLE_COLORS


# FRAME RENDER

def render_frame(
    ndvi_field,
    positions,
    trajectories,
    roles,
    role_history,
    R_comm,
    step,
):
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(ndvi_field, cmap="viridis")


    # Trajectories (Segment-wise coloring)

    for i, traj in enumerate(trajectories):

        traj = np.array(traj)
        role_seq = role_history[i]

        if len(traj) < 2:
            continue

        # Draw trajectory segments with role-based coloring
        for k in range(len(traj) - 1):

            if k < len(role_seq):
                role = role_seq[k]
            else:
                role = role_seq[-1]

            color = ROLE_COLORS[role]

            ax.plot(
                [traj[k, 1], traj[k + 1, 1]],
                [traj[k, 0], traj[k + 1, 0]],
                color=color,
                linewidth=2,
                alpha=0.9,
            )

        # Current agent position
        y, x = traj[-1]

        # Current role (safe indexing)
        if step < len(role_seq):
            current_role = role_seq[step]
        else:
            current_role = role_seq[-1]

        ax.scatter(
            x,
            y,
            c=ROLE_COLORS[current_role],
            s=40,
            edgecolors="black",
            linewidths=0.8,
            zorder=5,
        )

    # Graph edges (dotted)

    for i, j in itertools.combinations(range(len(positions)), 2):

        pi = positions[i]
        pj = positions[j]

        if np.linalg.norm(np.array(pi) - np.array(pj)) <= R_comm:

            ax.plot(
                [pi[1], pj[1]],
                [pi[0], pj[0]],
                linestyle="dotted",
                color="white",
                linewidth=1,
                alpha=0.6,
                zorder=3,
            )

    ax.set_title(f"t = {step}")
    ax.axis("off")

    fig.canvas.draw()
    frame = np.array(fig.canvas.renderer.buffer_rgba())[:, :, :3]

    plt.close(fig)
    return frame


# MAIN GIF FUNCTION


def generate_swarm_gif(
    ndvi_field,
    rollout_data,
    R_comm,
    output_path,
    duration_seconds: int = 60,
    fps: int = 10,
):
    trajectories = rollout_data["trajectories"]
    role_history = rollout_data["role_history"]

    max_steps = len(trajectories[0])

    total_frames = duration_seconds * fps
    step_skip = max(1, max_steps // total_frames)

    frames = []

    for step in range(0, max_steps, step_skip):

        # Safe positions
        positions = [
            traj[step] if step < len(traj) else traj[-1]
            for traj in trajectories
        ]

        # Safe roles
        roles = [
            role_history[i][step] if step < len(role_history[i])
            else role_history[i][-1]
            for i in range(len(trajectories))
        ]

        frame = render_frame(
            ndvi_field,
            positions,
            trajectories,
            roles,
            role_history,   # 🔥 PASS FULL HISTORY
            R_comm,
            step,
        )

        frames.append(frame)

    imageio.mimsave(output_path, frames, fps=fps)

    return output_path