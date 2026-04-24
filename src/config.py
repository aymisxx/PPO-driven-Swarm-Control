# CONFIG MODULE

from __future__ import annotations

import json
import random
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import torch

from stable_baselines3.common.utils import set_random_seed

# GLOBAL REPRODUCIBILITY

SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
set_random_seed(SEED)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# PROJECT PATHS

PROJECT_ROOT = Path.cwd()

DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"
MODELS_DIR = PROJECT_ROOT / "models"
LOGS_DIR = PROJECT_ROOT / "logs"

GIFS_DIR = RESULTS_DIR / "gifs"
PLOTS_DIR = RESULTS_DIR / "plots"
METRICS_DIR = RESULTS_DIR / "metrics"
FRAMES_DIR = RESULTS_DIR / "frames"

# Create directories
for path in [
    DATA_DIR,
    RESULTS_DIR,
    MODELS_DIR,
    LOGS_DIR,
    GIFS_DIR,
    PLOTS_DIR,
    METRICS_DIR,
    FRAMES_DIR,
]:
    path.mkdir(parents=True, exist_ok=True)

# DEVICE

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ROLE COLOR CONVENTION

ROLE_COLORS = {
    "Explorer": "tab:blue",
    "Surveyor": "tab:orange",
    "Defender": "tab:green",
    "Idle": "tab:red",
}

ROLE_NAMES = tuple(ROLE_COLORS.keys())

# GLOBAL CONFIGURATION

@dataclass
class Config:

    # image / preprocessing
    patch_size: int = 128
    vari_epsilon: float = 1e-6
    gaussian_blur_kernel: int = 5
    use_gaussian_smoothing: bool = True

    # single-agent environment
    max_steps_single: int = 300
    action_step_px: int = 1

    # PPO
    total_timesteps: int = 200_000
    learning_rate: float = 3e-4
    gamma: float = 0.99
    ppo_n_steps: int = 2048
    ppo_batch_size: int = 64
    ppo_n_epochs: int = 10

    # swarm
    num_agents: int = 8
    max_steps_swarm: int = 400
    dt: float = 1.0
    comm_radius: float = 120.0
    repulsion_radius: float = 60.0
    spawn_min_separation: float = 80.0
    spawn_margin: int = 64
    spawn_max_attempts: int = 10_000

    # hybrid control
    u_max: float = 4.0
    k_att: float = 1.00
    k_rep: float = 1.75
    k_visit: float = 1.25
    k_bnd: float = 1.00
    k_cons: float = 0.35

    # visualization
    show_patch_windows: bool = True
    show_apf_cues: bool = True
    show_graph_edges: bool = True
    role_color_tracks_current_role: bool = True
    gif_seconds_target: int = 60

    # outputs
    save_plots: bool = True
    save_metrics: bool = True
    save_gifs: bool = True


CFG = Config()

# SAVE CONFIG SNAPSHOT

config_payload = asdict(CFG)
config_payload["seed"] = SEED
config_payload["role_colors"] = ROLE_COLORS
config_payload["role_names"] = ROLE_NAMES
config_payload["device"] = str(DEVICE)

config_path = LOGS_DIR / "config.json"

with open(config_path, "w", encoding="utf-8") as f:
    json.dump(config_payload, f, indent=2)

# SANITY PRINT

def print_config_summary():
    print("=" * 72)
    print("CONFIG READY")
    print("=" * 72)
    print(f"Project root       : {PROJECT_ROOT}")
    print(f"Device             : {DEVICE}")
    print(f"Seed               : {SEED}")
    print(f"Agents             : {CFG.num_agents}")
    print(f"Patch size         : {CFG.patch_size}")
    print(f"Swarm steps        : {CFG.max_steps_swarm}")
    print("=" * 72)