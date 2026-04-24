# ROLE SYSTEM

from __future__ import annotations

import random
from collections import Counter


# ROLE DEFINITIONS

ROLES = ["Explorer", "Surveyor", "Defender", "Idle"]

ROLE_COLORS = {
    "Explorer": "blue",
    "Surveyor": "orange",
    "Defender": "green",
    "Idle": "red",
}


# ROLE WEIGHTS (from pipeline behavior description)

ROLE_WEIGHTS = {
    "Explorer":  {"ppo": 1.2, "rep": 0.8, "cons": 0.5},
    "Surveyor":  {"ppo": 1.0, "rep": 1.0, "cons": 0.8},
    "Defender":  {"ppo": 0.6, "rep": 1.5, "cons": 1.0},
    "Idle":      {"ppo": 0.2, "rep": 0.5, "cons": 0.2},
}


# INITIAL ROLE ASSIGNMENT

def initialize_roles(num_agents: int):
    """
    Random initial role assignment.
    """
    return [random.choice(ROLES) for _ in range(num_agents)]


# STOCHASTIC TRANSITION (CRN-inspired)

def transition_role(current_role: str) -> str:
    """
    Simple stochastic role transition.

    NOTE:
    This follows the notebook structure:
    cyclic transitions, not fully arbitrary.
    """

    transition_map = {
        "Explorer": ["Explorer", "Surveyor"],
        "Surveyor": ["Surveyor", "Defender"],
        "Defender": ["Defender", "Idle"],
        "Idle": ["Idle", "Explorer"],
    }

    return random.choice(transition_map[current_role])


# APPLY ROLE SWITCHING

def update_roles(roles, switch_prob: float = 0.05):
    """
    Update roles with stochastic switching.

    switch_prob:
        probability of switching at each timestep
    """

    new_roles = []

    for r in roles:
        if random.random() < switch_prob:
            new_roles.append(transition_role(r))
        else:
            new_roles.append(r)

    return new_roles


# ROLE STATISTICS

def compute_role_counts(roles):
    """
    Count how many agents are in each role.
    """
    counts = Counter(roles)
    return {r: counts[r] for r in ROLES}