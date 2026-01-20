"""Protocol definitions for the benchmark framework.

This module contains Protocol classes defining interfaces for:
- Models: neural networks or other parameterized functions
- Tasks: data sampling and loss/metric computation
- GradComputers: gradient computation strategies
- ConstraintSets: constraint handling for Frank-Wolfe methods
- Optimizers: optimization algorithms
- Environments: execution contexts (single-process or distributed)
- Topologies: graph structures for distributed communication
- Communicators: message passing abstractions
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Protocol, TypeVar, runtime_checkable

import numpy as np

from core.types import (
    History,
    MultiPayload,
    NodeId,
    ParamVector,
    StepRecord,
    StepResult,
)

__all__ = [
    "Batch_co",
    "Pred_co",
    "State_co",
    "Model",
    "Task",
    "GradComputer",
    "ConstraintSet",
    "Optimizer",
    "Environment",
    "Topology",
    "Communicator",
    "MultiCommunicator",
]

# Type variables for generic protocols
# Using covariant for return types, contravariant for parameter types
# For simplicity in this framework, we use covariant TypeVars
Batch_co = TypeVar("Batch_co", covariant=True)
Pred_co = TypeVar("Pred_co", covariant=True)
State_co = TypeVar("State_co", covariant=True)


@runtime_checkable
class Model(Protocol[Batch_co, Pred_co]):
    """Protocol for parameterized models.

    A Model represents any parameterized function that can:
    - Perform forward passes on batches of data
    - Expose its parameters as a flat vector
    - Accept parameter updates as a flat vector

    Type Parameters:
        Batch_co: The type of input data batches (covariant).
        Pred_co: The type of model predictions/outputs (covariant).
    """

    def forward(self, batch: Any) -> Any:
        """Compute forward pass on a batch of data.

        Args:
            batch: Input data batch.

        Returns:
            Model predictions for the batch.
        """
        ...

    def parameters_vector(self) -> ParamVector:
        """Return model parameters as a flat vector.

        Returns:
            A 1D numpy array containing all model parameters.
        """
        ...

    def set_parameters_vector(self, v: ParamVector) -> None:
        """Set model parameters from a flat vector.

        Args:
            v: A 1D numpy array containing all model parameters.
        """
        ...


@runtime_checkable
class Task(Protocol[Batch_co, Pred_co]):
    """Protocol for optimization tasks.

    A Task defines:
    - How to sample data batches
    - How to compute loss given a model and batch
    - How to compute additional metrics

    Type Parameters:
        Batch_co: The type of data batches (covariant).
        Pred_co: The type of model predictions (covariant).
    """

    def sample_batch(self, *, rng: np.random.Generator) -> Any:
        """Sample a batch of data.

        Args:
            rng: Random number generator for reproducible sampling.

        Returns:
            A batch of data.
        """
        ...

    def loss(self, model: Model[Any, Any], batch: Any) -> float:
        """Compute loss for a model on a batch.

        Args:
            model: The model to evaluate.
            batch: The data batch.

        Returns:
            The scalar loss value.
        """
        ...

    def metrics(self, model: Model[Any, Any], batch: Any) -> Mapping[str, float]:
        """Compute additional metrics for a model on a batch.

        Args:
            model: The model to evaluate.
            batch: The data batch.

        Returns:
            A mapping of metric names to values.
        """
        ...


@runtime_checkable
class GradComputer(Protocol[Batch_co, Pred_co]):
    """Protocol for gradient computation strategies.

    Abstracts how gradients are computed, allowing for:
    - Exact gradients
    - Stochastic gradients
    - Variance-reduced gradients
    - etc.

    Type Parameters:
        Batch_co: The type of data batches (covariant).
        Pred_co: The type of model predictions (covariant).
    """

    def grad(
        self,
        task: Task[Any, Any],
        model: Model[Any, Any],
        batch: Any,
    ) -> ParamVector:
        """Compute gradient of the loss with respect to model parameters.

        Args:
            task: The task defining the loss function.
            model: The model to compute gradients for.
            batch: The data batch.

        Returns:
            Gradient as a flat parameter vector.
        """
        ...


@runtime_checkable
class ConstraintSet(Protocol):
    """Protocol for constraint sets in Frank-Wolfe optimization.

    Defines the feasible region for constrained optimization.
    The key operation is the Linear Minimization Oracle (LMO).

    Note:
        The `project` method is optional. Concrete implementations
        may choose not to implement it if projection is not needed
        or not efficiently computable. Implementations that don't
        support projection should raise NotImplementedError.
    """

    def lmo(self, grad: ParamVector) -> ParamVector:
        """Linear Minimization Oracle.

        Solves: argmin_{s in C} <grad, s>
        where C is the constraint set.

        Args:
            grad: The gradient direction.

        Returns:
            The minimizer over the constraint set.
        """
        ...

    def project(self, x: ParamVector) -> ParamVector:
        """Project a point onto the constraint set.

        This method is optional. Implementations that don't support
        projection should raise NotImplementedError.

        Args:
            x: The point to project.

        Returns:
            The projection of x onto the constraint set.

        Raises:
            NotImplementedError: If projection is not supported.
        """
        ...


@runtime_checkable
class Optimizer(Protocol[Batch_co, Pred_co, State_co]):
    """Protocol for optimization algorithms.

    An Optimizer maintains internal state and performs optimization steps.
    It is generic over:
    - Batch_co: the type of data batches
    - Pred_co: the type of model predictions
    - State_co: the optimizer's internal state type

    Type Parameters:
        Batch_co: The type of data batches (covariant).
        Pred_co: The type of model predictions (covariant).
        State_co: The optimizer's internal state type (covariant).
    """

    def init_state(self, model: Model[Any, Any]) -> Any:
        """Initialize optimizer state for a model.

        Args:
            model: The model to optimize.

        Returns:
            Initial optimizer state.
        """
        ...

    def step(
        self,
        *,
        task: Task[Any, Any],
        model: Model[Any, Any],
        batch: Any,
        grad_computer: GradComputer[Any, Any],
        state: Any,
        rng: np.random.Generator,
    ) -> tuple[Any, StepResult]:
        """Perform one optimization step.

        Args:
            task: The task defining loss and metrics.
            model: The model being optimized (will be mutated).
            batch: The data batch for this step.
            grad_computer: The gradient computation strategy.
            state: Current optimizer state.
            rng: Random number generator.

        Returns:
            A tuple of (new_state, step_result).
        """
        ...


@runtime_checkable
class Environment(Protocol):
    """Protocol for execution environments.

    An Environment orchestrates the optimization loop, handling:
    - Model, task, and optimizer setup
    - Step execution
    - History tracking

    This is a general interface that supports both:
    - Single-process environments: step() returns StepResult
    - Multi-node distributed environments: step() returns Mapping[NodeId, StepResult]

    The History class handles both cases transparently.

    Note:
        Implementations should document whether they return StepResult
        or Mapping[NodeId, StepResult] from step().
    """

    def reset(self, *, seed: int) -> None:
        """Reset the environment with a new seed.

        This should reinitialize all random state and reset the model
        to its initial parameters.

        Args:
            seed: Random seed for reproducibility.
        """
        ...

    def step(self) -> StepRecord:
        """Execute one optimization step.

        Returns:
            - StepResult for single-process environments
            - Mapping[NodeId, StepResult] for multi-node environments
        """
        ...

    def run(self, *, steps: int) -> History:
        """Run multiple optimization steps.

        Args:
            steps: Number of steps to execute.

        Returns:
            History containing all step records.
        """
        ...

    def state_dict(self) -> dict[str, Any]:
        """Return the current state of the environment.

        This can be used for checkpointing and resuming experiments.

        Returns:
            A dictionary containing the environment's state.
        """
        ...


@runtime_checkable
class Topology(Protocol):
    """Protocol for network topologies in distributed optimization.

    A Topology defines the communication graph structure, specifying
    which nodes can directly communicate with each other.

    Contract:
    - neighbors(i) must not include node i itself
    - For undirected topologies: if j in neighbors(i), then i in neighbors(j)
    - Node IDs are integers in range [0, num_nodes())
    """

    def num_nodes(self) -> int:
        """Return the number of nodes in the topology.

        Returns:
            The total number of nodes (n >= 1).
        """
        ...

    def neighbors(self, node: NodeId) -> Sequence[NodeId]:
        """Return the neighbors of a node.

        Args:
            node: The node ID to query.

        Returns:
            Sequence of neighbor node IDs (not including the node itself).
        """
        ...


@runtime_checkable
class Communicator(Protocol):
    """Protocol for communication in distributed optimization.

    A Communicator performs one synchronous communication/mixing round,
    where each node's payload is combined with its neighbors' payloads
    according to some mixing weights.

    The gossip operation is synchronous: all nodes send and receive
    simultaneously in one round.
    """

    def gossip(self, payloads: Mapping[NodeId, ParamVector]) -> dict[NodeId, ParamVector]:
        """Perform one synchronous gossip/mixing round.

        Each node i computes a weighted combination of payloads from
        itself and its neighbors: x_i' = sum_j w_ij * x_j

        Args:
            payloads: Mapping from node ID to parameter vector.
                      Must contain exactly all nodes {0, 1, ..., n-1}.

        Returns:
            Dictionary mapping each node ID to its mixed payload.
            The returned vectors are independent copies (no aliasing).

        Raises:
            ValueError: If payloads is missing nodes or has shape mismatches.
        """
        ...


@runtime_checkable
class MultiCommunicator(Protocol):
    """Protocol for multi-channel communication in distributed optimization.

    Extends the basic Communicator concept to support multiple payload channels
    per node (e.g., parameters and gradient trackers for Gradient Tracking).

    Each channel is mixed independently using the same topology and weights.
    """

    def gossip_multi(
        self, payloads: Mapping[NodeId, Mapping[str, ParamVector]]
    ) -> dict[NodeId, dict[str, ParamVector]]:
        """Perform one synchronous gossip/mixing round on multiple channels.

        Each channel key is mixed independently using the same mixing weights.

        Args:
            payloads: Mapping from node ID to a dict of channel_name -> ParamVector.
                      All nodes must provide the same set of channel keys.
                      All vectors for a given channel must have the same shape.

        Returns:
            Dictionary mapping each node ID to a dict of mixed channel payloads.
            The returned vectors are independent copies (no aliasing).

        Raises:
            ValueError: If nodes have mismatched channel keys or shape mismatches.
        """
        ...


# Re-export types that protocols depend on for convenience
__all__ += ["ParamVector", "StepResult", "StepRecord", "History", "NodeId", "MultiPayload"]
