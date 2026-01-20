"""Optimization algorithms module.

This package contains optimizer implementations including:
- Frank-Wolfe (Conditional Gradient) optimizer
- Gradient Descent (GD) optimizer
- Projected Gradient Descent (PGD) optimizer
- Constraint sets (L2 ball, simplex)
- Step size schedules
"""

from __future__ import annotations

from optim.constraints import L2BallConstraint, SimplexConstraint
from optim.frank_wolfe import (
    FrankWolfeOptimizer,
    FWState,
    StepSize,
    constant_step_size,
    harmonic_step_size,
)
from optim.gradient_descent import (
    GDState,
    GradientDescentOptimizer,
    ProjectedGradientDescentOptimizer,
)

__all__ = [
    # Constraints
    "L2BallConstraint",
    "SimplexConstraint",
    # Frank-Wolfe
    "FrankWolfeOptimizer",
    "FWState",
    "StepSize",
    "constant_step_size",
    "harmonic_step_size",
    # Gradient Descent
    "GradientDescentOptimizer",
    "ProjectedGradientDescentOptimizer",
    "GDState",
]
