"""Adam optimizer with optional projection.

This module provides:
- AdamState: State for Adam optimizer (m, v, t)
- AdamOptimizer: Adam for unconstrained optimization
- AdamPGDOptimizer: Adam with projection for constrained optimization
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from core.protocols import ConstraintSet, GradComputer, Model, Task
from core.types import ParamVector, StepResult

__all__ = [
    "AdamState",
    "AdamOptimizer",
    "AdamPGDOptimizer",
]


@dataclass
class AdamState:
    """State for Adam optimizer.

    Attributes:
        t: Current iteration count.
        m: First moment estimate (momentum).
        v: Second moment estimate (squared gradients).
    """

    t: int = 0
    m: ParamVector = field(default_factory=lambda: np.array([]))
    v: ParamVector = field(default_factory=lambda: np.array([]))


class AdamOptimizer:
    """Adam optimizer for unconstrained optimization.

    Implements the Adam algorithm:
        m_t = beta1 * m_{t-1} + (1 - beta1) * g_t
        v_t = beta2 * v_{t-1} + (1 - beta2) * g_t^2
        m_hat = m_t / (1 - beta1^t)
        v_hat = v_t / (1 - beta2^t)
        x_{t+1} = x_t - lr * m_hat / (sqrt(v_hat) + eps)

    Attributes:
        lr: Learning rate.
        beta1: Exponential decay rate for first moment.
        beta2: Exponential decay rate for second moment.
        eps: Small constant for numerical stability.
    """

    def __init__(
        self,
        lr: float = 0.001,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8,
    ) -> None:
        """Initialize the optimizer.

        Args:
            lr: Learning rate. Must be positive.
            beta1: First moment decay rate. Default 0.9.
            beta2: Second moment decay rate. Default 0.999.
            eps: Numerical stability constant. Default 1e-8.

        Raises:
            ValueError: If lr <= 0 or beta values not in [0, 1).
        """
        if lr <= 0:
            raise ValueError(f"Learning rate must be positive, got {lr}")
        if not (0 <= beta1 < 1):
            raise ValueError(f"beta1 must be in [0, 1), got {beta1}")
        if not (0 <= beta2 < 1):
            raise ValueError(f"beta2 must be in [0, 1), got {beta2}")

        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps

    def init_state(self, model: Model[Any, Any]) -> AdamState:
        """Initialize optimizer state.

        Args:
            model: The model being optimized.

        Returns:
            Initial AdamState with zero moments.
        """
        x = model.parameters_vector()
        return AdamState(
            t=0,
            m=np.zeros_like(x),
            v=np.zeros_like(x),
        )

    def step(
        self,
        *,
        task: Task[Any, Any],
        model: Model[Any, Any],
        batch: Any,
        grad_computer: GradComputer[Any, Any],
        state: AdamState,
        rng: np.random.Generator,
    ) -> tuple[AdamState, StepResult]:
        """Perform one Adam step.

        Args:
            task: The optimization task.
            model: The model being optimized.
            batch: Current batch (may be None for batchless tasks).
            grad_computer: Gradient computation strategy.
            state: Current optimizer state.
            rng: Random number generator (unused).

        Returns:
            Tuple of (new_state, step_result).
        """
        # Get current parameters
        x = model.parameters_vector()

        # Compute gradient
        grad = grad_computer.grad(task, model, batch)

        # Update biased first moment estimate
        m = self.beta1 * state.m + (1 - self.beta1) * grad

        # Update biased second raw moment estimate
        v = self.beta2 * state.v + (1 - self.beta2) * (grad**2)

        # Increment step counter
        t = state.t + 1

        # Bias correction
        m_hat = m / (1 - self.beta1**t)
        v_hat = v / (1 - self.beta2**t)

        # Adam update
        x_new = x - self.lr * m_hat / (np.sqrt(v_hat) + self.eps)

        # Update model
        model.set_parameters_vector(x_new)

        # Compute metrics after update
        loss = task.loss(model, batch)
        metrics = task.metrics(model, batch)

        new_state = AdamState(t=t, m=m, v=v)
        return new_state, StepResult(loss=loss, metrics=dict(metrics))


class AdamPGDOptimizer:
    """Adam optimizer with projection for constrained optimization.

    Same as Adam, but projects onto constraint set after each update:
        x_temp = Adam update
        x_{t+1} = project(x_temp)

    Attributes:
        lr: Learning rate.
        beta1: Exponential decay rate for first moment.
        beta2: Exponential decay rate for second moment.
        eps: Small constant for numerical stability.
        constraint: Constraint set with project() method.
    """

    def __init__(
        self,
        constraint: ConstraintSet,
        lr: float = 0.001,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8,
    ) -> None:
        """Initialize the optimizer.

        Args:
            lr: Learning rate. Must be positive.
            constraint: Constraint set with project() method.
            beta1: First moment decay rate. Default 0.9.
            beta2: Second moment decay rate. Default 0.999.
            eps: Numerical stability constant. Default 1e-8.

        Raises:
            ValueError: If lr <= 0, beta values not in [0, 1), or constraint
                lacks project() method.
        """
        if lr <= 0:
            raise ValueError(f"Learning rate must be positive, got {lr}")
        if not (0 <= beta1 < 1):
            raise ValueError(f"beta1 must be in [0, 1), got {beta1}")
        if not (0 <= beta2 < 1):
            raise ValueError(f"beta2 must be in [0, 1), got {beta2}")

        # Check that constraint has project method
        project = getattr(constraint, "project", None)
        if not callable(project):
            raise ValueError(
                f"Constraint {type(constraint).__name__} does not have a project() method."
            )

        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.constraint = constraint
        self._project = project

    def init_state(self, model: Model[Any, Any]) -> AdamState:
        """Initialize optimizer state, projecting initial parameters.

        Args:
            model: The model being optimized.

        Returns:
            Initial AdamState with zero moments.
        """
        # Project initial parameters onto constraint set
        x = model.parameters_vector()
        x_proj = self._project(x)
        model.set_parameters_vector(x_proj)

        return AdamState(
            t=0,
            m=np.zeros_like(x_proj),
            v=np.zeros_like(x_proj),
        )

    def step(
        self,
        *,
        task: Task[Any, Any],
        model: Model[Any, Any],
        batch: Any,
        grad_computer: GradComputer[Any, Any],
        state: AdamState,
        rng: np.random.Generator,
    ) -> tuple[AdamState, StepResult]:
        """Perform one Adam step with projection.

        Args:
            task: The optimization task.
            model: The model being optimized.
            batch: Current batch (may be None for batchless tasks).
            grad_computer: Gradient computation strategy.
            state: Current optimizer state.
            rng: Random number generator (unused).

        Returns:
            Tuple of (new_state, step_result).
        """
        # Get current parameters
        x = model.parameters_vector()

        # Compute gradient
        grad = grad_computer.grad(task, model, batch)

        # Update biased first moment estimate
        m = self.beta1 * state.m + (1 - self.beta1) * grad

        # Update biased second raw moment estimate
        v = self.beta2 * state.v + (1 - self.beta2) * (grad**2)

        # Increment step counter
        t = state.t + 1

        # Bias correction
        m_hat = m / (1 - self.beta1**t)
        v_hat = v / (1 - self.beta2**t)

        # Adam update
        x_temp = x - self.lr * m_hat / (np.sqrt(v_hat) + self.eps)

        # Project onto constraint set
        x_new = self._project(x_temp)

        # Update model
        model.set_parameters_vector(x_new)

        # Compute metrics after update
        loss = task.loss(model, batch)
        metrics = task.metrics(model, batch)

        new_state = AdamState(t=t, m=m, v=v)
        return new_state, StepResult(loss=loss, metrics=dict(metrics))
