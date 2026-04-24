# GRAPH-BASED CONSENSUS

from __future__ import annotations

import numpy as np


# Neighbor detection (proximity graph)


def get_neighbors(
    positions,
    i: int,
    R_comm: float,
):
    """
    Return indices of neighbors of agent i
    """

    pi = np.array(positions[i])
    neighbors = []

    for j, pj in enumerate(positions):
        if i == j:
            continue

        dist = np.linalg.norm(pi - np.array(pj))

        if dist <= R_comm:
            neighbors.append(j)

    return neighbors


# Build full adjacency list

def build_neighbor_graph(
    positions,
    R_comm: float,
):
    """
    Returns adjacency list for all agents
    """

    graph = {}

    for i in range(len(positions)):
        graph[i] = get_neighbors(positions, i, R_comm)

    return graph


# Directional consensus term


def compute_consensus_term(
    i: int,
    directions: np.ndarray,
    positions,
    R_comm: float,
    k_cons: float,
):
    """
    Compute directional consensus correction.

    directions: (N,2) array of PPO directions
    """

    neighbors = get_neighbors(positions, i, R_comm)

    if len(neighbors) == 0:
        return np.zeros(2, dtype=np.float32)

    u_cons = np.zeros(2, dtype=np.float32)

    for j in neighbors:
        u_cons += (directions[j] - directions[i])

    u_cons *= k_cons

    return u_cons


# Count active edges (for metrics)


def count_edges(positions, R_comm: float):
    """
    Count number of active communication edges
    """

    edge_count = 0

    for i in range(len(positions)):
        for j in range(i + 1, len(positions)):
            pi = np.array(positions[i])
            pj = np.array(positions[j])

            if np.linalg.norm(pi - pj) <= R_comm:
                edge_count += 1

    return edge_count