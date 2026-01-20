"""Distributed optimization components.

This package provides building blocks for gossip-based distributed
optimization algorithms:

- Topologies: Graph structures defining node connectivity
  - RingTopology: Each node connected to left/right neighbors
  - CompleteTopology: All nodes connected to all others

- Weights: Mixing weight computation
  - metropolis_hastings_rows: Doubly-stochastic weights for undirected graphs

- Communicators: Message passing abstractions
  - SynchronousGossipCommunicator: Weighted averaging in one round (single + multi-channel)

- Strategies: Step execution patterns
  - GossipNode: Data structure for a node in gossip optimization
  - LocalStepThenGossipParams: Local step followed by parameter gossip
  - GossipThenLocalStep: Parameter gossip followed by local step
  - GradientTrackingStrategy: Gradient tracking for variance reduction
"""

from __future__ import annotations

from distributed.communicator import SynchronousGossipCommunicator
from distributed.strategies import (
    GossipNode,
    GossipStrategy,
    GossipThenLocalStep,
    GradientTrackingStrategy,
    LocalStepThenGossipParams,
)
from distributed.topology import CompleteTopology, RingTopology
from distributed.weights import metropolis_hastings_rows, row_sums_close_to_one

__all__ = [
    # Topologies
    "RingTopology",
    "CompleteTopology",
    # Weights
    "metropolis_hastings_rows",
    "row_sums_close_to_one",
    # Communicators
    "SynchronousGossipCommunicator",
    # Strategies
    "GossipNode",
    "GossipStrategy",
    "LocalStepThenGossipParams",
    "GossipThenLocalStep",
    "GradientTrackingStrategy",
]
