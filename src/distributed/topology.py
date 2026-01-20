"""Network topologies for distributed optimization.

This module provides concrete topology implementations for gossip-based
distributed optimization algorithms.

Supported topologies:
- RingTopology: Each node connected to its two neighbors in a ring
- CompleteTopology: All nodes connected to all other nodes
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

from core.types import NodeId

__all__ = ["RingTopology", "CompleteTopology"]


@dataclass(frozen=True)
class RingTopology:
    """Ring topology where each node is connected to its two neighbors.

    In a ring of n nodes, node i is connected to nodes (i-1) mod n and (i+1) mod n.

    For n=2, each node has exactly one neighbor (the other node).
    For n=1, the topology is invalid (raises ValueError).

    Attributes:
        n: Number of nodes in the ring. Must be >= 2.

    Example:
        >>> topo = RingTopology(n=4)
        >>> topo.neighbors(0)
        [3, 1]
        >>> topo.neighbors(1)
        [0, 2]
    """

    n: int
    _neighbor_cache: dict[NodeId, list[NodeId]] = field(
        default_factory=dict, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        """Validate that n >= 2."""
        if self.n < 2:
            raise ValueError(f"Ring topology requires n >= 2, got {self.n}")

    def num_nodes(self) -> int:
        """Return the number of nodes."""
        return self.n

    def neighbors(self, node: NodeId) -> Sequence[NodeId]:
        """Return the neighbors of a node in the ring.

        Args:
            node: Node ID in range [0, n).

        Returns:
            List of neighbor IDs: [(node-1) mod n, (node+1) mod n] for n >= 3,
            or [other_node] for n == 2.
        """
        if node in self._neighbor_cache:
            return self._neighbor_cache[node].copy()

        if self.n == 2:
            # Special case: each node has exactly one neighbor
            result = [1 - node]
        else:
            # General case: left and right neighbors
            left = (node - 1) % self.n
            right = (node + 1) % self.n
            result = [left, right]

        # Cache and return a copy
        # Note: We need to bypass frozen dataclass for caching
        object.__setattr__(self, "_neighbor_cache", {**self._neighbor_cache, node: result})
        return result.copy()


@dataclass(frozen=True)
class CompleteTopology:
    """Complete (fully-connected) topology where every node is connected to all others.

    In a complete graph of n nodes, each node is connected to all n-1 other nodes.

    Attributes:
        n: Number of nodes. Must be >= 2.

    Example:
        >>> topo = CompleteTopology(n=4)
        >>> topo.neighbors(0)
        [1, 2, 3]
        >>> topo.neighbors(2)
        [0, 1, 3]
    """

    n: int
    _neighbor_cache: dict[NodeId, list[NodeId]] = field(
        default_factory=dict, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        """Validate that n >= 2."""
        if self.n < 2:
            raise ValueError(f"Complete topology requires n >= 2, got {self.n}")

    def num_nodes(self) -> int:
        """Return the number of nodes."""
        return self.n

    def neighbors(self, node: NodeId) -> Sequence[NodeId]:
        """Return all other nodes as neighbors.

        Args:
            node: Node ID in range [0, n).

        Returns:
            List of all node IDs except the given node, in sorted order.
        """
        if node in self._neighbor_cache:
            return self._neighbor_cache[node].copy()

        result = [i for i in range(self.n) if i != node]

        # Cache and return a copy
        object.__setattr__(self, "_neighbor_cache", {**self._neighbor_cache, node: result})
        return result.copy()
