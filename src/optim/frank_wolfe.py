"""Frank-Wolfe (Conditional Gradient) optimizer.

This module provides a typed Frank-Wolfe optimizer that works with any
constraint set implementing the ConstraintSet protocol.

The Frank-Wolfe algorithm maintains feasibility by taking convex combinations
of the current iterate with vertices of the constraint set, rather than
performing projections.

Update rule:
    s_t = LMO(∇f(x_t))           # Linear minimization oracle
    x_{t+1} = (1-γ_t) x_t + γ_t s_t   # Convex combination

where γ_t is the step size (typically γ_t = 2/(t+2) for O(1/t) convergence).
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Generic, TypeVar

import numpy as np

from core.protocols import ConstraintSet, GradComputer, Model, Task
from core.types import ParamVector, StepResult

__all__ = [
    "FWState",
    "FrankWolfeOptimizer",
    "StepSize",
    "constant_step_size",
    "harmonic_step_size",
]

# Type alias for step size schedules
StepSize = Callable[[int], float]

# Type variables for generic optimizer
_Batch = TypeVar("_Batch")
_Pred = TypeVar("_Pred")


def constant_step_size(gamma: float) -> StepSize:
    """Create a constant step size schedule.

    Args:
        gamma: The constant step size. Must be in (0, 1].

    Returns:
        A callable that returns gamma for any iteration t.

    Raises:
        ValueError: If gamma is not in (0, 1].

    Example:
        >>> schedule = constant_step_size(0.1)
        >>> schedule(0)
        0.1
        >>> schedule(100)
        0.1
    """
    if not (0 < gamma <= 1):
        raise ValueError(f"gamma must be in (0, 1], got {gamma}")

    def schedule(t: int) -> float:
        return gamma

    return schedule


def harmonic_step_size() -> StepSize:
    """Create a harmonic (diminishing) step size schedule.

    Returns gamma(t) = 2 / (t + 2), which gives:
    - gamma(0) = 1.0
    - gamma(1) = 2/3 ≈ 0.667
    - gamma(2) = 0.5
    - gamma(t) → 0 as t → ∞

    This schedule provides O(1/t) convergence rate for convex functions.

    Returns:
        A callable that returns 2/(t+2) for iteration t.

    Example:
        >>> schedule = harmonic_step_size()
        >>> schedule(0)
        1.0
        >>> schedule(1)
        0.6666666666666666
    """

    def schedule(t: int) -> float:
        return 2.0 / (t + 2)

    return schedule


@dataclass
class FWState:
    """State for the Frank-Wolfe optimizer.

    Attributes:
        t: Iteration counter, starting at 0.
    """

    t: int


class FrankWolfeOptimizer(Generic[_Batch, _Pred]):
    """Frank-Wolfe (Conditional Gradient) optimizer.

    This optimizer implements the Frank-Wolfe algorithm for constrained
    optimization. It maintains feasibility by taking convex combinations
    with vertices returned by the Linear Minimization Oracle (LMO).

    The algorithm:
    1. Compute gradient g = ∇f(x_t)
    2. Find s_t = argmin_{s ∈ C} <g, s> via LMO
    3. Update x_{t+1} = (1-γ_t) x_t + γ_t s_t

    Feasibility is preserved because:
    - s_t ∈ C (LMO returns a feasible point)
    - x_{t+1} is a convex combination of x_t and s_t
    - Convex sets are closed under convex combinations

    Type Parameters:
        _Batch: The type of data batches (can be None for batchless tasks).
        _Pred: The type of model predictions.

    Attributes:
        constraint: The constraint set defining the feasible region.
        step_size: The step size schedule (function from iteration to gamma).

    Example:
        >>> from optim.constraints import L2BallConstraint
        >>> constraint = L2BallConstraint(radius=1.0)
        >>> optimizer = FrankWolfeOptimizer(
        ...     constraint=constraint,
        ...     step_size=harmonic_step_size(),
        ... )
    """

    def __init__(
        self,
        *,
        constraint: ConstraintSet,
        step_size: StepSize,
    ) -> None:
        """Initialize the Frank-Wolfe optimizer.

        Args:
            constraint: Constraint set with LMO method.
            step_size: Step size schedule (callable: iteration -> gamma).
        """
        self.constraint = constraint
        self.step_size = step_size

    def _try_project(self, x: ParamVector) -> ParamVector:
        """Try to project x if the constraint supports projection.

        Args:
            x: Point to project.

        Returns:
            Projected point if projection is available, otherwise x unchanged.
        """
        project_fn: Callable[[ParamVector], ParamVector] | None = getattr(
            self.constraint, "project", None
        )
        if project_fn is not None and callable(project_fn):
            try:
                return project_fn(x)
            except NotImplementedError:
                # Constraint doesn't support projection
                return x
        return x

    def init_state(self, model: Model[Any, Any]) -> FWState:
        """Initialize optimizer state and project model to feasible set.

        If the constraint supports projection, the model parameters are
        projected onto the feasible set. This ensures feasibility from
        the start, even if the initial parameters are infeasible.

        Args:
            model: The model to optimize.

        Returns:
            Initial optimizer state with t=0.
        """
        # Get current parameters
        x = model.parameters_vector()

        # Try to project onto feasible set
        x_proj = self._try_project(x)

        # Update model if projection changed the parameters
        if not np.array_equal(x, x_proj):
            model.set_parameters_vector(x_proj)

        return FWState(t=0)

    def step(
        self,
        *,
        task: Task[Any, Any],
        model: Model[Any, Any],
        batch: Any,
        grad_computer: GradComputer[Any, Any],
        state: FWState,
        rng: np.random.Generator,
    ) -> tuple[FWState, StepResult]:
        """Perform one Frank-Wolfe update step.

        Algorithm:
        1. x = current parameters
        2. g = gradient at x
        3. s = LMO(g) = argmin_{s ∈ C} <g, s>
        4. γ = step_size(t)
        5. x_new = (1-γ) x + γ s
        6. Update model with x_new

        Args:
            task: Task defining loss and metrics.
            model: Model being optimized (will be mutated).
            batch: Data batch for this step (can be None).
            grad_computer: Gradient computation strategy.
            state: Current optimizer state.
            rng: Random number generator (unused by FW, but part of protocol).

        Returns:
            Tuple of (new_state, step_result).

        Raises:
            ValueError: If step size is not in (0, 1] or not finite.
        """
        # Get current parameters
        x = model.parameters_vector()

        # Compute gradient
        g = grad_computer.grad(task, model, batch)

        # Linear minimization oracle
        s = self.constraint.lmo(g)

        # Get step size
        gamma = self.step_size(state.t)

        # Validate step size
        if not math.isfinite(gamma):
            raise ValueError(f"Step size must be finite, got {gamma} at t={state.t}")
        if not (0 < gamma <= 1):
            raise ValueError(f"Step size must be in (0, 1], got {gamma} at t={state.t}")

        # Frank-Wolfe update: convex combination
        x_new = (1 - gamma) * x + gamma * s
        x_new = np.asarray(x_new, dtype=np.float64)

        # Update model
        model.set_parameters_vector(x_new)

        # Compute loss and metrics after update
        loss = task.loss(model, batch)
        metrics = dict(task.metrics(model, batch))

        # Create new state
        new_state = FWState(t=state.t + 1)

        return new_state, StepResult(loss=loss, metrics=metrics)
