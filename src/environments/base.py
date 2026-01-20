"""Base environment class for the benchmark framework.

This module provides an abstract base class that implements the common
run loop logic while leaving environment-specific behavior to subclasses.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from core.types import History, StepRecord

__all__ = ["BaseEnvironment"]


class BaseEnvironment(ABC):
    """Abstract base class for optimization environments.

    Provides a reusable run loop that integrates with History for tracking
    optimization progress. Subclasses must implement `reset()` and `step()`.

    This class is mega-general and supports:
    - Single-process environments: step() returns StepResult
    - Multi-node distributed environments: step() returns Mapping[NodeId, StepResult]

    The History class handles both formats transparently and provides
    aggregation methods like mean_loss() and mean_metric().

    Attributes:
        _t: Internal step counter, starts at 0 after reset() and increments
            on each step() call.

    Example:
        >>> class MyEnv(BaseEnvironment):
        ...     def reset(self, *, seed: int) -> None:
        ...         self._t = 0
        ...         self._rng = np.random.default_rng(seed)
        ...
        ...     def step(self) -> StepRecord:
        ...         result = StepResult(loss=compute_loss(), metrics={})
        ...         self._t += 1
        ...         return result
        ...
        >>> env = MyEnv()
        >>> env.reset(seed=42)
        >>> history = env.run(steps=100)
        >>> print(history.mean_loss())
    """

    _t: int

    def __init__(self) -> None:
        """Initialize the environment with step counter at 0."""
        self._t = 0

    @property
    def t(self) -> int:
        """Current step index (read-only).

        Returns 0 after reset() and before any step() calls.
        Increments by 1 after each step() call.
        """
        return self._t

    @abstractmethod
    def reset(self, *, seed: int) -> None:
        """Reset the environment with a new random seed.

        Subclasses must:
        - Reset internal step counter to 0 (set self._t = 0)
        - Reinitialize any random state using the provided seed
        - Reset model parameters to initial values (if applicable)

        Args:
            seed: Random seed for reproducibility.
        """
        ...

    @abstractmethod
    def step(self) -> StepRecord:
        """Execute one optimization step.

        Subclasses must:
        - Perform one step of optimization
        - Increment self._t by 1
        - Return the step result

        Returns:
            - StepResult for single-process environments
            - Mapping[NodeId, StepResult] for multi-node environments

        Note:
            Subclasses should increment self._t at the end of this method.
        """
        ...

    def run(self, *, steps: int) -> History:
        """Run multiple optimization steps and collect history.

        Args:
            steps: Number of steps to execute. Must be >= 1.

        Returns:
            History containing all step records. Use history.mean_loss()
            or history.mean_metric(key) for aggregated statistics.

        Raises:
            ValueError: If steps < 1.
        """
        if steps < 1:
            raise ValueError(f"steps must be >= 1, got {steps}")

        history = History()
        for _ in range(steps):
            record = self.step()
            history.append(record)

        return history

    def state_dict(self) -> dict[str, Any]:
        """Return the current state of the environment.

        This base implementation returns minimal state. Subclasses should
        override to include model parameters, optimizer state, etc.

        Returns:
            A dictionary containing:
            - "t": Current step index
        """
        return {"t": self._t}
