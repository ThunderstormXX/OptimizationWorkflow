"""Gradient Descent optimizers.

This module provides:
- GradientDescentOptimizer: Vanilla gradient descent for unconstrained problems
- ProjectedGradientDescentOptimizer: Projected GD for constrained problems

These serve as baseline optimizers for comparing against Frank-Wolfe.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from core.protocols import ConstraintSet, GradComputer, Model, Task
from core.types import StepResult

__all__ = [
    "GDState",
    "GradientDescentOptimizer",
    "ProjectedGradientDescentOptimizer",
]


@dataclass
class GDState:
    """State for gradient descent optimizers.

    Attributes:
        t: Current iteration count.
    """

    t: int = 0


class GradientDescentOptimizer:
    """Vanilla Gradient Descent optimizer.

    Performs the update:
        x_{t+1} = x_t - lr * grad(f, x_t)

    This is for unconstrained optimization. For constrained problems,
    use ProjectedGradientDescentOptimizer.

    Attributes:
        lr: Learning rate (step size).
    """

    def __init__(self, lr: float) -> None:
        """Initialize the optimizer.

        Args:
            lr: Learning rate. Must be positive.

        Raises:
            ValueError: If lr <= 0.
        """
        if lr <= 0:
            raise ValueError(f"Learning rate must be positive, got {lr}")
        self.lr = lr

    def init_state(self, model: Model[Any, Any]) -> GDState:
        """Initialize optimizer state.

        Args:
            model: The model being optimized.

        Returns:
            Initial GDState with t=0.
        """
        return GDState(t=0)

    def step(
        self,
        *,
        task: Task[Any, Any],
        model: Model[Any, Any],
        batch: Any,
        grad_computer: GradComputer[Any, Any],
        state: GDState,
        rng: np.random.Generator,
    ) -> tuple[GDState, StepResult]:
        """Perform one gradient descent step.

        Args:
            task: The optimization task.
            model: The model being optimized.
            batch: Current batch (may be None for batchless tasks).
            grad_computer: Gradient computation strategy.
            state: Current optimizer state.
            rng: Random number generator (unused in vanilla GD).

        Returns:
            Tuple of (new_state, step_result).
        """
        # Get current parameters
        x = model.parameters_vector()

        # Compute gradient
        grad = grad_computer.grad(task, model, batch)

        # Gradient descent update
        x_new = x - self.lr * grad

        # Update model
        model.set_parameters_vector(x_new)

        # Compute metrics after update
        loss = task.loss(model, batch)
        metrics = task.metrics(model, batch)

        new_state = GDState(t=state.t + 1)
        return new_state, StepResult(loss=loss, metrics=dict(metrics))


class ProjectedGradientDescentOptimizer:
    """Projected Gradient Descent optimizer.

    Performs the update:
        x_temp = x_t - lr * grad(f, x_t)
        x_{t+1} = project(x_temp)

    where project() is the projection onto the constraint set.

    Attributes:
        lr: Learning rate (step size).
        constraint: Constraint set with a project() method.
    """

    def __init__(self, lr: float, constraint: ConstraintSet) -> None:
        """Initialize the optimizer.

        Args:
            lr: Learning rate. Must be positive.
            constraint: Constraint set. Must have a project() method.

        Raises:
            ValueError: If lr <= 0 or constraint lacks project().
        """
        if lr <= 0:
            raise ValueError(f"Learning rate must be positive, got {lr}")

        # Check that constraint has project method
        project = getattr(constraint, "project", None)
        if not callable(project):
            raise ValueError(
                f"Constraint {type(constraint).__name__} does not have a project() method. "
                "PGD requires a projectable constraint (e.g., L2BallConstraint)."
            )

        self.lr = lr
        self.constraint = constraint
        self._project = project

    def init_state(self, model: Model[Any, Any]) -> GDState:
        """Initialize optimizer state, projecting initial parameters if needed.

        Args:
            model: The model being optimized.

        Returns:
            Initial GDState with t=0.
        """
        # Project initial parameters onto the constraint set
        x = model.parameters_vector()
        x_proj = self._project(x)
        model.set_parameters_vector(x_proj)
        return GDState(t=0)

    def step(
        self,
        *,
        task: Task[Any, Any],
        model: Model[Any, Any],
        batch: Any,
        grad_computer: GradComputer[Any, Any],
        state: GDState,
        rng: np.random.Generator,
    ) -> tuple[GDState, StepResult]:
        """Perform one projected gradient descent step.

        Args:
            task: The optimization task.
            model: The model being optimized.
            batch: Current batch (may be None for batchless tasks).
            grad_computer: Gradient computation strategy.
            state: Current optimizer state.
            rng: Random number generator (unused in PGD).

        Returns:
            Tuple of (new_state, step_result).
        """
        # Get current parameters
        x = model.parameters_vector()

        # Compute gradient
        grad = grad_computer.grad(task, model, batch)

        # Gradient descent update
        x_temp = x - self.lr * grad

        # Project onto constraint set
        x_new = self._project(x_temp)

        # Update model
        model.set_parameters_vector(x_new)

        # Compute metrics after update
        loss = task.loss(model, batch)
        metrics = task.metrics(model, batch)

        new_state = GDState(t=state.t + 1)
        return new_state, StepResult(loss=loss, metrics=dict(metrics))
