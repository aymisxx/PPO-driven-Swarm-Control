"""Top-level script: produce the final hybrid-swarm 60-second GIF.

Usage (from the repo root, after `pip install -r requirements.txt`):

    python run_hybrid_rollout.py

Behavior:
    1. Sets up project directories.
    2. Loads (or rebuilds) the VARI utility field from data/field_satellite.jpg.
    3. Loads the trained PPO model from models/ppo_ndvi_drone_final.zip.
       (Use --train to train it from scratch — ~5 min on a modern GPU.)
    4. Spawns 8 agents with a random-center + 2x spread initialization
       (matches Section 15 of the reproducibility notebook).
    5. Runs the full hybrid controller
           u_i = w_ppo(role) * u_ppo_i
               + w_rep(role) * F_rep_i
               + w_cons(role) * u_cons_i
       with stochastic role switching over 400 steps.
    6. Writes a 60-second GIF to results/gifs/final_hybrid_swarm.gif.

All physics constants (R_rep=60, R_comm=120, K_rep=1.75, K_cons=0.35,
SPREAD=2.0, role-switch probability=0.05, etc.) are identical to the
notebook. The spawn RNG is intentionally unseeded so each invocation
produces a different starting configuration.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Path handling: this script lives at the repo root; src/ is a sibling.
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.config import (
    CFG,
    DATA_DIR,
    GIFS_DIR,
    MODELS_DIR,
    SEED,
    ensure_dirs,
    save_config_snapshot,
)
from src.evaluation import evaluate_single_agent  # noqa: F401 (available for --eval)
from src.ppo_train import load_ppo_model, train_ppo
from src.swarm.spawning import random_spread_spawn
from src.utility_field import build_utility_field, load_utility_field
from src.utils import get_device, seed_everything
from src.visualization import render_hybrid_rollout_gif


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the full hybrid swarm rollout.")
    parser.add_argument(
        "--image",
        type=Path,
        default=DATA_DIR / "field_satellite.jpg",
        help="Path to the RGB satellite image.",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=MODELS_DIR / "ppo_ndvi_drone_final",
        help="Path to the trained PPO model (without .zip extension is fine).",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=GIFS_DIR / "final_hybrid_swarm.gif",
        help="Output GIF path.",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=CFG.max_steps_swarm,
        help="Rollout horizon in steps.",
    )
    parser.add_argument(
        "--seconds",
        type=int,
        default=CFG.gif_seconds_target,
        help="Target GIF duration (seconds).",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=10,
        help="GIF frame rate.",
    )
    parser.add_argument(
        "--num-agents",
        type=int,
        default=CFG.num_agents,
        help="Number of agents.",
    )
    parser.add_argument(
        "--train",
        action="store_true",
        help="Train PPO from scratch before running the hybrid rollout.",
    )
    parser.add_argument(
        "--deterministic-spawn",
        action="store_true",
        help="Seed np.random before spawning so runs are reproducible.",
    )
    args = parser.parse_args()

    # -----------------------------------------------------------------------
    # 1. Setup
    # -----------------------------------------------------------------------
    ensure_dirs()
    device = get_device()
    seed_everything(SEED)  # reproducible torch / SB3 side
    snap = save_config_snapshot(device_str=str(device))

    print("=" * 72)
    print("PPO-Driven Swarm Control | Hybrid Rollout Script")
    print("=" * 72)
    print(f"Repo root      : {REPO_ROOT}")
    print(f"Device         : {device}")
    print(f"Config snapshot: {snap}")
    print("-" * 72)

    # -----------------------------------------------------------------------
    # 2. Utility field (cached if available)
    # -----------------------------------------------------------------------
    cached_field = DATA_DIR / "ndvi_field.npy"
    if cached_field.exists():
        print(f"Loading cached utility field from {cached_field}")
        ndvi_field = load_utility_field(cached_field)
    else:
        print(f"Building VARI utility field from {args.image}")
        ndvi_field = build_utility_field(args.image)
    print(f"Utility field shape: {ndvi_field.shape}")

    # -----------------------------------------------------------------------
    # 3. PPO model
    # -----------------------------------------------------------------------
    if args.train:
        print("-" * 72)
        print("Training PPO from scratch (this takes several minutes)...")
        ppo_model = train_ppo(ndvi_field=ndvi_field, seed=SEED, model_path=args.model)
    else:
        if not (Path(str(args.model) + ".zip").exists() or Path(args.model).exists()):
            print(
                f"[ERROR] PPO model not found at {args.model}[.zip].\n"
                "Re-run with --train to train from scratch, or place the\n"
                "pretrained zip at models/ppo_ndvi_drone_final.zip."
            )
            return 1
        print(f"Loading PPO model from {args.model}")
        ppo_model = load_ppo_model(args.model)

    # -----------------------------------------------------------------------
    # 4. Spawn + rollout + GIF
    # -----------------------------------------------------------------------
    if args.deterministic_spawn:
        # Only affects the spawn RNG; hybrid controller still uses stdlib random
        # and the PPO stochastic policy, so role transitions remain varied.
        import numpy as np
        np.random.seed(SEED)

    initial_positions = random_spread_spawn(
        ndvi_shape=ndvi_field.shape,
        num_agents=args.num_agents,
    )
    print("-" * 72)
    print(f"Initial agent positions ({len(initial_positions)} agents):")
    for i, p in enumerate(initial_positions):
        print(f"  Agent {i}: {p}")

    print("-" * 72)
    print(
        f"Rendering hybrid rollout GIF: "
        f"{args.steps} steps, target={args.seconds}s @ {args.fps} fps"
    )
    gif_path, result = render_hybrid_rollout_gif(
        ppo_model=ppo_model,
        ndvi_field=ndvi_field,
        initial_positions=initial_positions,
        out_path=args.out,
        target_seconds=args.seconds,
        fps=args.fps,
        max_steps=args.steps,
    )

    # -----------------------------------------------------------------------
    # 5. Report
    # -----------------------------------------------------------------------
    from collections import Counter

    final_roles = Counter(result["final_roles"])
    print("-" * 72)
    print("Hybrid rollout summary")
    print("-" * 72)
    print(f"Coverage ratio           : {result['coverage_ratio']:.8f}")
    print(f"NDVI gain (first visits) : {result['ndvi_gain']:.4f}")
    print(f"Unique visited cells     : {result['unique_once']}")
    print(f"Redundancy index         : {result['redundancy']:.4f}")
    print(f"Min pairwise distance    : {result['min_distance']:.2f}")
    print(f"Mean pairwise distance   : {result['mean_pairwise_distance']:.2f}")
    print(f"Close encounters (<20)   : {result['close_encounters']}")
    print("Final role distribution:")
    for r in ("Explorer", "Surveyor", "Defender", "Idle"):
        print(f"  {r:<9s}: {final_roles[r]}")
    print("-" * 72)
    print(f"GIF saved to: {gif_path}")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
