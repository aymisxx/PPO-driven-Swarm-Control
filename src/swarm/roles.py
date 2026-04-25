"""CRN-inspired stochastic role switching (Section 14).

Exposed:
    ROLES, ROLE_WEIGHTS, ROLE_COLORS_SIMPLE  (re-exported from config)
    transition_role(current_role) -> new_role
"""
from __future__ import annotations

import random

from ..config import (
    ROLE_COLORS_SIMPLE,
    ROLE_NAMES,
    ROLE_TRANSITIONS,
    ROLE_WEIGHTS,
)

ROLES = list(ROLE_NAMES)


def transition_role(current_role: str) -> str:
    """Draw the next role from the CRN-inspired transition graph.

    Each current role has two possible successors (stay or advance);
    selection is uniform over the two, matching the notebook's Section 14.
    """
    if current_role not in ROLE_TRANSITIONS:
        raise ValueError(f"Unknown role: {current_role}")
    return random.choice(ROLE_TRANSITIONS[current_role])


__all__ = [
    "ROLES",
    "ROLE_WEIGHTS",
    "ROLE_COLORS_SIMPLE",
    "transition_role",
]
