"""Global configuration, paths, and role-color convention.

Mirrors Section 0 of the reproducibility notebook exactly.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
import json

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
SEED: int = 42

# ---------------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------------
# By default we resolve paths relative to the repository root (parent of src/).
# The run script can override PROJECT_ROOT via environment / argparse.
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent

DATA_DIR: Path = PROJECT_ROOT / "data"
RESULTS_DIR: Path = PROJECT_ROOT / "results"
MODELS_DIR: Path = PROJECT_ROOT / "models"
LOGS_DIR: Path = PROJECT_ROOT / "logs"
GIFS_DIR: Path = RESULTS_DIR / "gifs"
PLOTS_DIR: Path = RESULTS_DIR / "plots"
METRICS_DIR: Path = RESULTS_DIR / "metrics"
FRAMES_DIR: Path = RESULTS_DIR / "frames"

ALL_DIRS = [
    DATA_DIR,
    RESULTS_DIR,
    MODELS_DIR,
    LOGS_DIR,
    GIFS_DIR,
    PLOTS_DIR,
    METRICS_DIR,
    FRAMES_DIR,
]


def ensure_dirs() -> None:
    for p in ALL_DIRS:
        p.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Role-color convention (Section 0.5)
# ---------------------------------------------------------------------------
ROLE_COLORS = {
    "Explorer": "tab:blue",
    "Surveyor": "tab:orange",
    "Defender": "tab:green",
    "Idle": "tab:red",
}
ROLE_NAMES = tuple(ROLE_COLORS.keys())

# Plain names used inside the GIF rendering path (matches notebook Section 15).
ROLE_COLORS_SIMPLE = {
    "Explorer": "blue",
    "Surveyor": "orange",
    "Defender": "green",
    "Idle": "red",
}

# ---------------------------------------------------------------------------
# Role weights (Section 14)
# ---------------------------------------------------------------------------
ROLE_WEIGHTS = {
    "Explorer": {"ppo": 1.2, "rep": 0.8, "cons": 0.5},
    "Surveyor": {"ppo": 1.0, "rep": 1.0, "cons": 0.8},
    "Defender": {"ppo": 0.6, "rep": 1.5, "cons": 1.0},
    "Idle":     {"ppo": 0.2, "rep": 0.5, "cons": 0.2},
}

# Role transition graph (CRN-inspired, Section 14)
ROLE_TRANSITIONS = {
    "Explorer": ["Explorer", "Surveyor"],
    "Surveyor": ["Surveyor", "Defender"],
    "Defender": ["Defender", "Idle"],
    "Idle":     ["Idle", "Explorer"],
}

# Probability of considering a role switch at each step (Section 14)
ROLE_SWITCH_PROB: float = 0.05


# ---------------------------------------------------------------------------
# Experiment configuration
# ---------------------------------------------------------------------------
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

    # hybrid-control defaults
    u_max: float = 4.0
    k_att: float = 1.00
    k_rep: float = 1.75
    k_visit: float = 1.25
    k_bnd: float = 1.00
    k_cons: float = 0.35

    # visualization controls
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


def save_config_snapshot(device_str: str = "cpu") -> Path:
    """Save the config + seed + role info to logs/config_section0.json."""
    ensure_dirs()
    payload = asdict(CFG)
    payload["seed"] = SEED
    payload["role_colors"] = ROLE_COLORS
    payload["role_names"] = list(ROLE_NAMES)
    payload["device"] = str(device_str)
    out = LOGS_DIR / "config_section0.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    return out
