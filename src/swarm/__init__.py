"""Swarm layer: spawning, roles, hybrid controllers."""

from .roles import ROLES, transition_role
from .spawning import fixed_cluster_spawn, random_spread_spawn
from .controllers import (
    rollout_naive_ppo,
    rollout_ppo_repulsion,
    rollout_ppo_repulsion_consensus,
    rollout_full_hybrid,
)

__all__ = [
    "ROLES",
    "transition_role",
    "fixed_cluster_spawn",
    "random_spread_spawn",
    "rollout_naive_ppo",
    "rollout_ppo_repulsion",
    "rollout_ppo_repulsion_consensus",
    "rollout_full_hybrid",
]