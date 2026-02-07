"""Optimization algorithms module.

This package contains optimizer implementations including:
- Frank-Wolfe (Conditional Gradient) optimizer
- Gradient Descent (GD) optimizer
- Projected Gradient Descent (PGD) optimizer
- Adam optimizer (with optional projection)
- Constraint sets (L1 ball, L2 ball, simplex)
- Step size schedules
"""

from __future__ import annotations

from optim.adam import AdamOptimizer, AdamPGDOptimizer, AdamState
from optim.constraints import L1BallConstraint, L2BallConstraint, SimplexConstraint
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
from optim.muon import Muon
from optim.stochastic_frank_wolfe import StochasticFrankWolfe
from optim.torch_gd import TorchGD

__all__ = [
    # Constraints
    "L1BallConstraint",
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
    # Adam
    "AdamOptimizer",
    "AdamPGDOptimizer",
    "AdamState",
    # Torch optimizers
    "StochasticFrankWolfe",
    "Muon",
    "TorchGD",
]
