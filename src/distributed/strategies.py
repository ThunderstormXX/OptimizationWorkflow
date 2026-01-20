"""Gossip strategies for distributed optimization.

This module provides strategy implementations that define the order of
operations in gossip-based distributed optimization:
- Local optimization step
- Parameter gossip/mixing
- Gradient tracking (for variance reduction)

Different strategies can implement different orderings (e.g., step-then-gossip
vs gossip-then-step) for various algorithmic variants.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Generic, Protocol, TypeVar

import numpy as np

from core.protocols import Communicator, GradComputer, Model, Optimizer, Task
from core.types import NodeId, ParamVector, StepResult
from distributed.communicator import SynchronousGossipCommunicator

__all__ = [
    "GossipNode",
    "GossipStrategy",
    "LocalStepThenGossipParams",
    "GossipThenLocalStep",
    "GradientTrackingStrategy",
]

# Type variables for generic strategy
_Batch = TypeVar("_Batch")
_Pred = TypeVar("_Pred")
_State = TypeVar("_State")


@dataclass
class GossipNode(Generic[_Batch, _Pred, _State]):
    """A node in a gossip-based distributed optimization setup.

    Each node maintains its own model, optimizer state, and RNG for
    independent local computations.

    Attributes:
        node_id: Unique identifier for this node.
        task: The optimization task (data sampling, loss, metrics).
        model: The model being optimized (parameters are modified in-place).
        optimizer: The optimization algorithm.
        grad_computer: Gradient computation strategy.
        opt_state: Current optimizer state (mutable).
        rng: Random number generator for this node (mutable).
    """

    node_id: NodeId
    task: Task[_Batch, _Pred]
    model: Model[_Batch, _Pred]
    optimizer: Optimizer[_Batch, _Pred, _State]
    grad_computer: GradComputer[_Batch, _Pred]
    opt_state: _State
    rng: np.random.Generator


class GossipStrategy(Protocol[_Batch, _Pred, _State]):
    """Protocol for gossip step strategies.

    A strategy defines the order of operations in one round of
    gossip-based distributed optimization.
    """

    def step(
        self,
        *,
        nodes: Sequence[GossipNode[_Batch, _Pred, _State]],
        communicator: Communicator,
    ) -> dict[NodeId, StepResult]:
        """Execute one round of the gossip strategy.

        Args:
            nodes: Sequence of gossip nodes (will be mutated).
            communicator: Communicator for parameter mixing.

        Returns:
            Mapping from node ID to step result.
        """
        ...


class LocalStepThenGossipParams:
    """Strategy: local optimization step, then gossip parameters.

    This strategy implements the standard "local SGD with gossip averaging":
    1. Each node performs a local optimization step (updates its model)
    2. All nodes gossip (mix) their parameters via the communicator
    3. Each node's model is updated with the mixed parameters

    This ordering is suitable for algorithms like:
    - Decentralized SGD
    - Gossip-based Frank-Wolfe
    - Local SGD with periodic averaging

    Note:
        The optimizer step is performed BEFORE gossip, so each node
        first moves toward its local objective, then parameters are
        averaged with neighbors.
    """

    def step(
        self,
        *,
        nodes: Sequence[GossipNode[_Batch, _Pred, _State]],
        communicator: Communicator,
    ) -> dict[NodeId, StepResult]:
        """Execute local step then gossip.

        Algorithm:
        1. For each node i:
           - Sample batch from task
           - Perform optimizer.step() to update model
           - Collect parameters and results
        2. Gossip: mix parameters across nodes
        3. Update each node's model with mixed parameters

        Args:
            nodes: Sequence of gossip nodes.
            communicator: Communicator for parameter mixing.

        Returns:
            Mapping from node ID to step result from local optimization.
        """
        results: dict[NodeId, StepResult] = {}
        payloads: dict[NodeId, ParamVector] = {}

        # Phase 1: Local optimization step for each node
        for node in nodes:
            # Sample batch
            batch = node.task.sample_batch(rng=node.rng)

            # Perform local optimization step
            new_state, result = node.optimizer.step(
                task=node.task,
                model=node.model,
                batch=batch,
                grad_computer=node.grad_computer,
                state=node.opt_state,
                rng=node.rng,
            )

            # Update node state
            node.opt_state = new_state

            # Collect payload (parameters after local step)
            payloads[node.node_id] = node.model.parameters_vector()
            results[node.node_id] = result

        # Phase 2: Gossip - mix parameters across nodes
        mixed = communicator.gossip(payloads)

        # Phase 3: Update each node's model with mixed parameters
        for node in nodes:
            node.model.set_parameters_vector(mixed[node.node_id])

        return results


class GossipThenLocalStep:
    """Strategy: gossip parameters first, then local optimization step.

    This strategy implements gossip-first distributed optimization:
    1. All nodes gossip (mix) their current parameters via the communicator
    2. Each node's model is updated with the mixed parameters
    3. Each node performs a local optimization step (updates its model)

    This ordering differs from LocalStepThenGossipParams in that consensus
    is reached before the optimization step, which can lead to different
    convergence behavior.

    Note:
        The gossip is performed BEFORE the optimizer step, so each node
        first averages with neighbors, then moves toward its local objective.
    """

    def step(
        self,
        *,
        nodes: Sequence[GossipNode[_Batch, _Pred, _State]],
        communicator: Communicator,
    ) -> dict[NodeId, StepResult]:
        """Execute gossip then local step.

        Algorithm:
        1. Collect payloads = current parameters from all nodes
        2. Gossip: mix parameters across nodes
        3. Update each node's model with mixed parameters
        4. For each node i:
           - Sample batch from task
           - Perform optimizer.step() to update model
           - Collect results

        Args:
            nodes: Sequence of gossip nodes.
            communicator: Communicator for parameter mixing.

        Returns:
            Mapping from node ID to step result from local optimization.
        """
        # Phase 1: Collect current parameters (sorted by node_id for determinism)
        payloads: dict[NodeId, ParamVector] = {}
        for node in sorted(nodes, key=lambda n: n.node_id):
            payloads[node.node_id] = node.model.parameters_vector()

        # Phase 2: Gossip - mix parameters across nodes
        mixed = communicator.gossip(payloads)

        # Phase 3: Update each node's model with mixed parameters
        for node in sorted(nodes, key=lambda n: n.node_id):
            node.model.set_parameters_vector(mixed[node.node_id])

        # Phase 4: Local optimization step for each node
        results: dict[NodeId, StepResult] = {}
        for node in sorted(nodes, key=lambda n: n.node_id):
            # Sample batch
            batch = node.task.sample_batch(rng=node.rng)

            # Perform local optimization step
            new_state, result = node.optimizer.step(
                task=node.task,
                model=node.model,
                batch=batch,
                grad_computer=node.grad_computer,
                state=node.opt_state,
                rng=node.rng,
            )

            # Update node state
            node.opt_state = new_state
            results[node.node_id] = result

        return results


@dataclass
class _GTNodeState:
    """Internal state for a node in Gradient Tracking.

    Attributes:
        y: Gradient tracker vector (approximates global average gradient).
        g_prev: Previous local gradient (for computing gradient difference).
    """

    y: ParamVector
    g_prev: ParamVector


@dataclass
class GradientTrackingStrategy:
    """Strategy: Gradient Tracking for distributed optimization.

    Implements the Gradient Tracking algorithm (also known as NEXT or DIGing):

    State per node i:
    - x_i: parameters (stored in model)
    - y_i: gradient tracker (stored in this strategy)
    - g_i_prev: previous local gradient (stored in this strategy)

    Per step:
    1. Compute local gradient g_i at current x_i
    2. Update tracker: y_i <- sum_j w_ij * y_j + (g_i - g_i_prev)
    3. Update parameters: x_i <- sum_j w_ij * x_j - lr * y_i
    4. Store g_i as g_i_prev for next iteration

    This strategy uses a fixed learning rate for simplicity.
    The gradient tracker y_i approximates the global average gradient,
    enabling faster convergence than local SGD on heterogeneous data.

    Attributes:
        lr: Learning rate for parameter updates.
        _node_states: Internal state per node (initialized on reset).
        _initialized: Whether reset() has been called.
    """

    lr: float = 0.1
    _node_states: dict[NodeId, _GTNodeState] = field(default_factory=dict, repr=False)
    _initialized: bool = field(default=False, repr=False)

    def reset(
        self,
        nodes: Sequence[GossipNode[Any, Any, Any]],
        communicator: Communicator,
    ) -> None:
        """Initialize gradient tracking state for all nodes.

        Computes initial gradients and sets y_i = g_i, g_prev_i = g_i.

        Args:
            nodes: Sequence of gossip nodes.
            communicator: Communicator (used to verify topology).
        """
        self._node_states = {}

        for node in nodes:
            # Compute initial gradient
            batch = node.task.sample_batch(rng=node.rng)
            g_i = node.grad_computer.grad(task=node.task, model=node.model, batch=batch)

            # Initialize y_i = g_i and g_prev_i = g_i
            self._node_states[node.node_id] = _GTNodeState(
                y=g_i.copy(),
                g_prev=g_i.copy(),
            )

        self._initialized = True

    def step(
        self,
        *,
        nodes: Sequence[GossipNode[Any, Any, Any]],
        communicator: Communicator,
    ) -> dict[NodeId, StepResult]:
        """Execute one round of Gradient Tracking.

        Algorithm:
        1. For each node i: compute local gradient g_i
        2. Gossip parameters x and trackers y (multi-channel)
        3. For each node i:
           - y_i <- mixed_y_i + (g_i - g_prev_i)
           - x_i <- mixed_x_i - lr * y_i
           - g_prev_i <- g_i
        4. Compute and return step results

        Args:
            nodes: Sequence of gossip nodes.
            communicator: Communicator for parameter mixing.

        Returns:
            Mapping from node ID to step result.

        Raises:
            RuntimeError: If reset() has not been called.
        """
        if not self._initialized:
            raise RuntimeError("GradientTrackingStrategy.reset() must be called before step()")

        # Ensure communicator supports gossip_multi
        if not isinstance(communicator, SynchronousGossipCommunicator):
            raise TypeError("GradientTrackingStrategy requires SynchronousGossipCommunicator")

        # Phase 1: Compute local gradients for all nodes
        gradients: dict[NodeId, ParamVector] = {}
        for node in sorted(nodes, key=lambda n: n.node_id):
            batch = node.task.sample_batch(rng=node.rng)
            g_i = node.grad_computer.grad(task=node.task, model=node.model, batch=batch)
            gradients[node.node_id] = g_i

        # Phase 2: Prepare multi-channel payloads (x and y)
        payloads: dict[NodeId, dict[str, ParamVector]] = {}
        for node in sorted(nodes, key=lambda n: n.node_id):
            state = self._node_states[node.node_id]
            payloads[node.node_id] = {
                "x": node.model.parameters_vector(),
                "y": state.y.copy(),
            }

        # Phase 3: Gossip both channels
        mixed = communicator.gossip_multi(payloads)

        # Phase 4: Update trackers and parameters
        results: dict[NodeId, StepResult] = {}
        for node in sorted(nodes, key=lambda n: n.node_id):
            nid = node.node_id
            state = self._node_states[nid]
            g_i = gradients[nid]

            # Update tracker: y_i <- mixed_y_i + (g_i - g_prev_i)
            new_y = mixed[nid]["y"] + (g_i - state.g_prev)

            # Update parameters: x_i <- mixed_x_i - lr * y_i
            new_x = mixed[nid]["x"] - self.lr * new_y

            # Apply updates
            node.model.set_parameters_vector(new_x)
            state.y = new_y
            state.g_prev = g_i.copy()

            # Compute step result (loss and metrics after update)
            batch = node.task.sample_batch(rng=node.rng)
            loss = node.task.loss(node.model, batch)
            metrics = dict(node.task.metrics(node.model, batch))
            results[nid] = StepResult(loss=loss, metrics=metrics)

        return results
