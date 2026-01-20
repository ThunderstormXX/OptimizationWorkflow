"""Single-process (non-distributed) environment implementation.

This module provides an environment for training a single model on a single
task using a pluggable optimizer, with strict determinism via seeded RNG.
"""

from __future__ import annotations

from typing import Any, Generic, TypeVar

import numpy as np

from core.protocols import GradComputer, Model, Optimizer, Task
from core.types import StepResult
from environments.base import BaseEnvironment

__all__ = ["SingleProcessEnvironment"]

# Type variables for generic environment
_Batch = TypeVar("_Batch")
_Pred = TypeVar("_Pred")
_State = TypeVar("_State")


class SingleProcessEnvironment(BaseEnvironment, Generic[_Batch, _Pred, _State]):
    """Non-distributed environment for single-model optimization.

    This environment orchestrates the training loop for a single model
    on a single task, using a pluggable optimizer and gradient computer.
    It ensures deterministic runs via seeded random number generation.

    Type Parameters:
        _Batch: The type of data batches produced by the task.
        _Pred: The type of model predictions.
        _State: The optimizer's internal state type.

    Example:
        >>> env = SingleProcessEnvironment(
        ...     task=my_task,
        ...     model=my_model,
        ...     optimizer=my_optimizer,
        ...     grad_computer=my_grad_computer,
        ... )
        >>> env.reset(seed=42)
        >>> history = env.run(steps=100)
        >>> print(f"Final loss: {history.last().loss}")

    Attributes:
        task: The task defining data sampling and loss computation.
        model: The model being optimized.
        optimizer: The optimization algorithm.
        grad_computer: The gradient computation strategy.
    """

    def __init__(
        self,
        *,
        task: Task[_Batch, _Pred],
        model: Model[_Batch, _Pred],
        optimizer: Optimizer[_Batch, _Pred, _State],
        grad_computer: GradComputer[_Batch, _Pred],
    ) -> None:
        """Initialize the single-process environment.

        Args:
            task: Task defining data sampling, loss, and metrics.
            model: Model to be optimized.
            optimizer: Optimizer algorithm.
            grad_computer: Gradient computation strategy.
        """
        super().__init__()
        self.task = task
        self.model = model
        self.optimizer = optimizer
        self.grad_computer = grad_computer

        # Internal state (initialized on reset)
        self._rng: np.random.Generator | None = None
        self._opt_state: _State | None = None
        self._seed: int | None = None

    def reset(self, *, seed: int) -> None:
        """Reset the environment with a new random seed.

        Initializes:
        - Step counter to 0
        - Random number generator with the given seed
        - Optimizer state for the current model

        Args:
            seed: Random seed for reproducibility.
        """
        self._t = 0
        self._seed = seed
        self._rng = np.random.default_rng(seed)
        self._opt_state = self.optimizer.init_state(self.model)

    def step(self) -> StepResult:
        """Execute one optimization step.

        Performs:
        1. Sample a batch from the task
        2. Call optimizer.step() to update model parameters
        3. Increment step counter
        4. Return the step result

        Returns:
            StepResult containing loss and metrics for this step.

        Raises:
            RuntimeError: If reset() has not been called.
        """
        if self._rng is None or self._opt_state is None:
            raise RuntimeError("Environment not initialized. Call reset(seed=...) before step().")

        # Sample batch
        batch = self.task.sample_batch(rng=self._rng)

        # Perform optimization step
        self._opt_state, result = self.optimizer.step(
            task=self.task,
            model=self.model,
            batch=batch,
            grad_computer=self.grad_computer,
            state=self._opt_state,
            rng=self._rng,
        )

        # Increment step counter
        self._t += 1

        return result

    def state_dict(self) -> dict[str, Any]:
        """Return the current state of the environment.

        Returns a dictionary containing:
        - "t": Current step index
        - "seed": The seed used for initialization (None if not reset)
        - "params": Model parameters as a list (JSON-serializable)

        Note:
            The "params" field is converted to a Python list for JSON
            compatibility. Use np.array(state["params"]) to reconstruct.

        Returns:
            Dictionary with environment state.
        """
        return {
            "t": self._t,
            "seed": self._seed,
            "params": self.model.parameters_vector().tolist(),
        }
