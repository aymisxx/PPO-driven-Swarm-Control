# HYBRID CONTROLLER 

from __future__ import annotations

import numpy as np

from src.swarm.apf import compute_repulsion_force
from src.swarm.consensus import compute_consensus_term
from src.swarm.roles import ROLE_WEIGHTS


# SINGLE AGENT CONTROL

def compute_control(
    i: int,
    positions,
    directions: np.ndarray,
    roles,
    R_rep: float,
    k_rep_eff: float,
    R_comm: float,
    k_cons: float,
):
    """
    Compute final control vector for agent i.

    Implements:
        u_i = w_ppo * u_ppo + w_pf * F_rep + w_cons * u_cons
    """

    role = roles[i]
    weights = ROLE_WEIGHTS[role]

    # PPO component (directional)

    u_ppo = directions[i] * weights["ppo"]

    # APF (repulsion only here)

    F_rep = compute_repulsion_force(
        i,
        positions,
        R_rep,
        k_rep_eff,
    )

    F_rep *= weights["rep"]

    # Consensus

    u_cons = compute_consensus_term(
        i,
        directions,
        positions,
        R_comm,
        k_cons,
    )

    u_cons *= weights["cons"]

    # Combine

    u = u_ppo + F_rep + u_cons

    # Normalize (important)

    norm = np.linalg.norm(u)

    if norm > 1e-6:
        u = u / norm

    return u


# BATCH CONTROL (ALL AGENTS)

def compute_all_controls(
    positions,
    directions: np.ndarray,
    roles,
    R_rep: float,
    k_rep_eff: float,
    R_comm: float,
    k_cons: float,
):
    """
    Compute control for all agents
    """

    controls = []

    for i in range(len(positions)):
        u = compute_control(
            i,
            positions,
            directions,
            roles,
            R_rep,
            k_rep_eff,
            R_comm,
            k_cons,
        )
        controls.append(u)

    return np.array(controls)