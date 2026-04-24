# ARTIFICIAL POTENTIAL FIELD

from __future__ import annotations

import numpy as np


# Pairwise distances

def compute_pairwise_distances(positions):
    dists = []
    for i in range(len(positions)):
        for j in range(i + 1, len(positions)):
            pi = np.array(positions[i])
            pj = np.array(positions[j])
            dists.append(np.linalg.norm(pi - pj))
    return dists


# Adaptive repulsion strength

def compute_k_rep_eff(k_rep, R_rep, positions):
    dists = compute_pairwise_distances(positions)

    if len(dists) == 0:
        return k_rep

    d_min = min(dists)

    if d_min < R_rep:
        return k_rep * (R_rep / (d_min + 1e-6))
    else:
        return k_rep


# Repulsion force for ONE agent

def compute_repulsion_force(
    i: int,
    positions,
    R_rep: float,
    k_rep_eff: float,
):
    """
    Compute repulsive force for agent i.
    """

    yi, xi = positions[i]
    F_rep = np.zeros(2, dtype=np.float32)

    for j, (yj, xj) in enumerate(positions):
        if i == j:
            continue

        diff = np.array([yi - yj, xi - xj], dtype=np.float32)
        dist = np.linalg.norm(diff)

        if dist < R_rep and dist > 1e-6:

            psi = max(0.0, (1.0 / dist) - (1.0 / R_rep))

            F_rep += k_rep_eff * psi * (diff / (dist + 1e-6))

    return F_rep