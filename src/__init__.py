"""PPO-driven Swarm Control — modular source package.

Mirrors the reproducibility notebook:
    Section 0  -> config, utils
    Section 1  -> utility_field
    Sections 2-4 -> env
    Section 5  -> ppo_train
    Sections 6-7 -> evaluation
    Sections 8-14 -> swarm.controllers, swarm.roles, swarm.spawning
    Section 15 -> visualization (GIF)
"""

__all__ = [
    "config",
    "utils",
    "utility_field",
    "env",
    "ppo_train",
    "evaluation",
    "visualization",
    "swarm",
]