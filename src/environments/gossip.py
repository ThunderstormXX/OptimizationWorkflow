"""Gossip-based distributed environment for multi-node optimization.

This module provides a GossipEnvironment that orchestrates distributed
optimization across multiple nodes using gossip-based communication.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Generic, TypeVar

import numpy as np

from core.protocols import Communicator
from core.types import NodeId, ParamVector, StepRecord
from distributed.strategies import (
    GossipNode,
    GossipStrategy,
    LocalStepThenGossipParams,
)
from environments.base import BaseEnvironment

__all__ = ["GossipEnvironment", "consensus_error"]

# Type variables for generic environment
_Batch = TypeVar("_Batch")
_Pred = TypeVar("_Pred")
_State = TypeVar("_State")


def consensus_error(params_by_node: Mapping[NodeId, ParamVector]) -> float:
    """Compute the consensus error across nodes.

    Consensus error measures how far the nodes are from agreement:
        error = mean_i ||x_i - x_bar||_2

    where x_bar = mean_i x_i is the global average.

    Args:
        params_by_node: Mapping from node ID to parameter vector.

    Returns:
        Mean L2 distance from each node to the global average.

    Example:
        >>> params = {0: np.array([0.0]), 1: np.array([2.0])}
        >>> consensus_error(params)  # |0-1| + |2-1| / 2 = 1.0
        1.0
    """
    if not params_by_node:
        return 0.0

    # Stack all parameter vectors
    vectors = [params_by_node[i] for i in sorted(params_by_node.keys())]
    stacked = np.stack(vectors, axis=0)  # shape: (n_nodes, dim)

    # Compute global mean
    x_bar = np.mean(stacked, axis=0)  # shape: (dim,)

    # Compute mean distance to average
    distances = [float(np.linalg.norm(v - x_bar)) for v in vectors]
    return float(np.mean(distances))


class GossipEnvironment(BaseEnvironment, Generic[_Batch, _Pred, _State]):
    """Multi-node gossip-based distributed optimization environment.

    This environment orchestrates distributed optimization where:
    1. Multiple nodes each have their own model, optimizer, and RNG
    2. Nodes perform local optimization steps
    3. Nodes communicate via gossip (parameter mixing)

    The step() method returns a Mapping[NodeId, StepResult] containing
    results from all nodes, which History handles transparently.

    Type Parameters:
        _Batch: The type of data batches.
        _Pred: The type of model predictions.
        _State: The optimizer's internal state type.

    Attributes:
        _nodes: List of gossip nodes (stable order by node_id).
        _communicator: Communicator for parameter mixing.
        _strategy: Strategy defining the order of operations.

    Example:
        >>> env = GossipEnvironment(
        ...     nodes=[node0, node1, node2],
        ...     communicator=SynchronousGossipCommunicator(topology),
        ... )
        >>> env.reset(seed=42)
        >>> history = env.run(steps=100)
        >>> print(f"Mean loss: {history.mean_loss()}")
    """

    def __init__(
        self,
        *,
        nodes: Sequence[GossipNode[_Batch, _Pred, _State]],
        communicator: Communicator,
        strategy: GossipStrategy[_Batch, _Pred, _State] | None = None,
    ) -> None:
        """Initialize the gossip environment.

        Args:
            nodes: Sequence of gossip nodes. Must have unique node_ids.
            communicator: Communicator for parameter mixing.
            strategy: Strategy for step execution. Defaults to
                      LocalStepThenGossipParams.

        Raises:
            ValueError: If node_ids are not unique.
        """
        super().__init__()

        # Validate unique node IDs
        node_ids = [node.node_id for node in nodes]
        if len(node_ids) != len(set(node_ids)):
            raise ValueError("Node IDs must be unique")

        # Store nodes sorted by node_id for stable ordering
        self._nodes: list[GossipNode[_Batch, _Pred, _State]] = sorted(
            nodes, key=lambda n: n.node_id
        )
        self._communicator = communicator
        self._strategy: GossipStrategy[_Batch, _Pred, _State] = (
            strategy if strategy is not None else LocalStepThenGossipParams()
        )
        self._seed: int | None = None

    @property
    def num_nodes(self) -> int:
        """Number of nodes in the environment."""
        return len(self._nodes)

    def reset(self, *, seed: int) -> None:
        """Reset the environment with deterministic per-node RNGs.

        This method:
        1. Resets the step counter to 0
        2. Stores the master seed
        3. Generates deterministic per-node seeds from the master seed
        4. Reinitializes each node's RNG with its per-node seed
        5. Reinitializes optimizer state for each node
        6. Calls strategy.reset() if the strategy has a reset method

        The per-node seeding ensures that:
        - Given the same master seed, each node gets the same RNG sequence
        - Different nodes get independent (but reproducible) RNG streams

        Args:
            seed: Master random seed for reproducibility.

        Note:
            This method does NOT recreate models or tasks. It operates on
            the provided node objects, only resetting RNG and optimizer state.
        """
        self._t = 0
        self._seed = seed

        # Create master RNG for generating per-node seeds
        master_rng = np.random.default_rng(seed)

        # Generate per-node seeds deterministically
        n = len(self._nodes)
        per_node_seeds = master_rng.integers(0, 2**32 - 1, size=n, dtype=np.uint64)

        # Reset each node
        for i, node in enumerate(self._nodes):
            # Set node RNG with deterministic seed
            node.rng = np.random.default_rng(int(per_node_seeds[i]))

            # Reinitialize optimizer state
            node.opt_state = node.optimizer.init_state(node.model)

        # Call strategy.reset() if available (e.g., for GradientTrackingStrategy)
        if hasattr(self._strategy, "reset") and callable(self._strategy.reset):
            self._strategy.reset(self._nodes, self._communicator)

    def step(self) -> StepRecord:
        """Execute one round of gossip-based optimization.

        Performs:
        1. Strategy step (e.g., local optimization + gossip)
        2. Increment step counter

        Returns:
            Mapping from node ID to StepResult for this round.

        Raises:
            RuntimeError: If reset() has not been called.
        """
        if self._seed is None:
            raise RuntimeError("Environment not initialized. Call reset(seed=...) before step().")

        # Execute strategy
        results = self._strategy.step(
            nodes=self._nodes,
            communicator=self._communicator,
        )

        # Increment step counter
        self._t += 1

        return results

    def state_dict(self) -> dict[str, Any]:
        """Return the current state of the environment.

        Returns a dictionary containing:
        - "t": Current step index
        - "seed": The master seed used for initialization
        - "params_by_node": Dict mapping node_id (as string) to parameter list

        The params_by_node uses string keys for JSON compatibility.

        Returns:
            Dictionary with environment state.
        """
        params_by_node: dict[str, list[float]] = {}
        for node in self._nodes:
            # Use string key for JSON compatibility
            key = str(node.node_id)
            params_by_node[key] = node.model.parameters_vector().tolist()

        return {
            "t": self._t,
            "seed": self._seed,
            "params_by_node": params_by_node,
        }

    def get_params_by_node(self) -> dict[NodeId, ParamVector]:
        """Get current parameters for all nodes.

        Returns:
            Dictionary mapping node ID to parameter vector (copies).
        """
        return {node.node_id: node.model.parameters_vector() for node in self._nodes}
