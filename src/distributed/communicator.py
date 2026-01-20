"""Communicators for gossip-based distributed optimization.

This module provides communicator implementations that perform
synchronous gossip/mixing rounds for distributed optimization.

Supports both single-channel (ParamVector) and multi-channel (dict of ParamVectors)
payloads for advanced algorithms like Gradient Tracking.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

import numpy as np

from core.protocols import Topology
from core.types import NodeId, ParamVector
from distributed.weights import metropolis_hastings_rows

__all__ = ["SynchronousGossipCommunicator"]


@dataclass
class SynchronousGossipCommunicator:
    """Synchronous gossip communicator using weighted averaging.

    Performs one synchronous communication round where each node
    computes a weighted average of its own payload and its neighbors':

        x_i' = sum_j w_ij * x_j

    where w_ij are the mixing weights (row-stochastic).

    If no weights are provided, Metropolis-Hastings weights are computed
    automatically from the topology.

    Attributes:
        topology: The network topology defining connectivity.
        rows: Mixing weights as nested dict rows[i][j] = w_ij.
              If not provided, computed via metropolis_hastings_rows.

    Example:
        >>> from distributed.topology import RingTopology
        >>> topo = RingTopology(n=4)
        >>> comm = SynchronousGossipCommunicator(topology=topo)
        >>> payloads = {i: np.array([float(i)]) for i in range(4)}
        >>> mixed = comm.gossip(payloads)
    """

    topology: Topology
    rows: Mapping[NodeId, Mapping[NodeId, float]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Compute default weights if not provided."""
        if not self.rows:
            # Compute Metropolis-Hastings weights
            object.__setattr__(self, "rows", metropolis_hastings_rows(self.topology))

    def edge_weights(self) -> list[tuple[NodeId, NodeId, float]]:
        """Return undirected edges with their communication weights.

        Returns a list of (i, j, weight) tuples for all edges where i < j.
        The weight is computed as (w_ij + w_ji) / 2 for symmetry.
        Self-loops (i == j) are excluded.

        Returns:
            List of (node_i, node_j, weight) tuples sorted by (i, j).
        """
        n = self.topology.num_nodes()
        edges: list[tuple[NodeId, NodeId, float]] = []
        seen: set[tuple[NodeId, NodeId]] = set()

        for i in range(n):
            row_i = self.rows.get(i, {})
            for j in row_i:
                if i == j:
                    continue  # Skip self-loops
                # Ensure i < j for undirected edge
                edge = (min(i, j), max(i, j))
                if edge in seen:
                    continue
                seen.add(edge)

                # Compute symmetric weight
                w_ij = row_i.get(j, 0.0)
                row_j = self.rows.get(j, {})
                w_ji = row_j.get(i, 0.0)
                weight = (w_ij + w_ji) / 2.0

                edges.append((edge[0], edge[1], weight))

        # Sort by (i, j) for determinism
        edges.sort(key=lambda e: (e[0], e[1]))
        return edges

    def gossip(self, payloads: Mapping[NodeId, ParamVector]) -> dict[NodeId, ParamVector]:
        """Perform one synchronous gossip/mixing round.

        Each node i computes:
            x_i' = sum_{j in {i} ∪ neighbors(i)} w_ij * x_j

        The operation is deterministic: iterating over nodes and weights
        in sorted order ensures reproducible results.

        Args:
            payloads: Mapping from node ID to parameter vector.
                      Must contain exactly all nodes {0, 1, ..., n-1}.
                      All vectors must have the same shape.

        Returns:
            Dictionary mapping each node ID to its mixed payload.
            Returned vectors are independent copies (no aliasing).

        Raises:
            ValueError: If payloads is missing nodes, has extra nodes,
                        or contains vectors with mismatched shapes.
        """
        n = self.topology.num_nodes()
        expected_nodes = set(range(n))
        actual_nodes = set(payloads.keys())

        # Check for exact node match
        if actual_nodes != expected_nodes:
            missing = expected_nodes - actual_nodes
            extra = actual_nodes - expected_nodes
            msg_parts = []
            if missing:
                msg_parts.append(f"missing nodes: {sorted(missing)}")
            if extra:
                msg_parts.append(f"extra nodes: {sorted(extra)}")
            raise ValueError(f"Payload node mismatch: {', '.join(msg_parts)}")

        # Check shape consistency
        shapes = [payloads[i].shape for i in range(n)]
        reference_shape = shapes[0]
        for i, shape in enumerate(shapes):
            if shape != reference_shape:
                raise ValueError(
                    f"Shape mismatch: node 0 has shape {reference_shape}, "
                    f"node {i} has shape {shape}"
                )

        # Perform gossip mixing
        result: dict[NodeId, ParamVector] = {}

        for i in range(n):
            row = self.rows[i]
            # Initialize mixed vector
            mixed = np.zeros(reference_shape, dtype=np.float64)

            # Weighted sum over neighbors and self (sorted for determinism)
            for j in sorted(row.keys()):
                w_ij = row[j]
                mixed += w_ij * payloads[j]

            result[i] = mixed

        return result

    def gossip_multi(
        self, payloads: Mapping[NodeId, Mapping[str, ParamVector]]
    ) -> dict[NodeId, dict[str, ParamVector]]:
        """Perform one synchronous gossip/mixing round on multiple channels.

        Each channel key is mixed independently using the same mixing weights.
        This is useful for algorithms like Gradient Tracking that need to
        mix both parameters and gradient trackers.

        Args:
            payloads: Mapping from node ID to a dict of channel_name -> ParamVector.
                      All nodes must provide the same set of channel keys.
                      All vectors for a given channel must have the same shape.

        Returns:
            Dictionary mapping each node ID to a dict of mixed channel payloads.
            Returned vectors are independent copies (no aliasing).

        Raises:
            ValueError: If nodes have mismatched channel keys or shape mismatches.
        """
        n = self.topology.num_nodes()
        expected_nodes = set(range(n))
        actual_nodes = set(payloads.keys())

        # Check for exact node match
        if actual_nodes != expected_nodes:
            missing = expected_nodes - actual_nodes
            extra = actual_nodes - expected_nodes
            msg_parts = []
            if missing:
                msg_parts.append(f"missing nodes: {sorted(missing)}")
            if extra:
                msg_parts.append(f"extra nodes: {sorted(extra)}")
            raise ValueError(f"Payload node mismatch: {', '.join(msg_parts)}")

        # Check that all nodes have the same channel keys
        reference_keys: set[str] | None = None
        for node_id in range(n):
            node_keys = set(payloads[node_id].keys())
            if reference_keys is None:
                reference_keys = node_keys
            elif node_keys != reference_keys:
                raise ValueError(
                    f"Channel key mismatch: node 0 has keys {sorted(reference_keys)}, "
                    f"node {node_id} has keys {sorted(node_keys)}"
                )

        if reference_keys is None or len(reference_keys) == 0:
            raise ValueError("Payloads must have at least one channel key")

        # Check shape consistency per channel
        channel_shapes: dict[str, tuple[int, ...]] = {}
        for key in reference_keys:
            ref_shape = payloads[0][key].shape
            channel_shapes[key] = ref_shape
            for node_id in range(n):
                if payloads[node_id][key].shape != ref_shape:
                    raise ValueError(
                        f"Shape mismatch for channel '{key}': node 0 has shape {ref_shape}, "
                        f"node {node_id} has shape {payloads[node_id][key].shape}"
                    )

        # Perform gossip mixing per channel
        result: dict[NodeId, dict[str, ParamVector]] = {i: {} for i in range(n)}

        for key in sorted(reference_keys):  # Sorted for determinism
            ref_shape = channel_shapes[key]
            for i in range(n):
                row = self.rows[i]
                mixed = np.zeros(ref_shape, dtype=np.float64)

                for j in sorted(row.keys()):
                    w_ij = row[j]
                    mixed += w_ij * payloads[j][key]

                result[i][key] = mixed

        return result
