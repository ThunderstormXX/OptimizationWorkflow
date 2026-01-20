"""Mixing weights for gossip-based distributed optimization.

This module provides functions to compute row-stochastic mixing weights
for gossip algorithms, ensuring convergence properties.

The main method is Metropolis-Hastings weights, which produce doubly-stochastic
matrices for undirected graphs, guaranteeing average preservation.
"""

from __future__ import annotations

from core.protocols import Topology
from core.types import NodeId

__all__ = ["metropolis_hastings_rows", "row_sums_close_to_one"]


def metropolis_hastings_rows(
    topology: Topology,
) -> dict[NodeId, dict[NodeId, float]]:
    """Compute Metropolis-Hastings row-stochastic weights for a topology.

    For an undirected graph, the Metropolis-Hastings weights are:
    - For i != j where j is a neighbor of i:
        w_ij = 1 / (1 + max(deg(i), deg(j)))
    - Self-weight:
        w_ii = 1 - sum_{j in neighbors(i)} w_ij

    These weights are:
    - Row-stochastic: sum_j w_ij = 1 for all i
    - Symmetric: w_ij = w_ji (for undirected graphs)
    - Doubly-stochastic: sum_i w_ij = 1 for all j (for undirected graphs)

    The doubly-stochastic property ensures that the global average is
    preserved under gossip mixing: mean(x') = mean(x).

    Args:
        topology: The network topology defining the graph structure.

    Returns:
        Nested dictionary where rows[i][j] = w_ij for all i and j in
        {i} ∪ neighbors(i). Self-weights (w_ii) are included.

    Example:
        >>> from distributed.topology import RingTopology
        >>> topo = RingTopology(n=4)
        >>> rows = metropolis_hastings_rows(topo)
        >>> rows[0]  # Weights for node 0
        {3: 0.333..., 1: 0.333..., 0: 0.333...}
    """
    n = topology.num_nodes()

    # Compute degrees for all nodes
    degrees: dict[NodeId, int] = {}
    for i in range(n):
        degrees[i] = len(topology.neighbors(i))

    # Compute row weights
    rows: dict[NodeId, dict[NodeId, float]] = {}

    for i in range(n):
        neighbors_i = topology.neighbors(i)
        row: dict[NodeId, float] = {}

        # Compute off-diagonal weights
        neighbor_weight_sum = 0.0
        for j in neighbors_i:
            # Metropolis-Hastings weight
            w_ij = 1.0 / (1.0 + max(degrees[i], degrees[j]))
            row[j] = w_ij
            neighbor_weight_sum += w_ij

        # Self-weight ensures row sums to 1
        row[i] = 1.0 - neighbor_weight_sum

        rows[i] = row

    return rows


def row_sums_close_to_one(
    rows: dict[NodeId, dict[NodeId, float]],
    tol: float = 1e-12,
) -> bool:
    """Check if all row sums are close to 1.

    Args:
        rows: Nested dictionary of weights from metropolis_hastings_rows.
        tol: Tolerance for checking sum == 1.

    Returns:
        True if all rows sum to 1 within tolerance, False otherwise.

    Example:
        >>> rows = {0: {0: 0.5, 1: 0.5}, 1: {0: 0.5, 1: 0.5}}
        >>> row_sums_close_to_one(rows)
        True
    """
    for _i, row in rows.items():
        row_sum = sum(row.values())
        if abs(row_sum - 1.0) > tol:
            return False
    return True
