"""Optimizers and constraints."""

from __future__ import annotations

from optim.adam import AdamOptimizer, AdamState
from optim.legacy_frankwolfe import (
    FWState,
    GDState,
    GradientDescentOptimizer,
    FrankWolfeOptimizer,
    L1BallConstraint,
    L2BallConstraint,
    L2BallTensorConstraint,
    OrthogonalMatrixConstraint,
    ProjectedGradientDescentOptimizer,
    SimplexConstraint,
    StepSize,
    StochasticFrankWolfe,
    StochasticFrankWolfeMomentumPost,
    StochasticFrankWolfeMomentumPre,
    OrthogonalSGDM,
    constant_step_size,
    harmonic_step_size,
)
from optim.muon import (
    Muon,
    MuonWithAuxAdam,
    SingleDeviceMuon,
    SingleDeviceMuonWithAuxAdam,
)
from optim.scion import Scion, ScionLight

__all__ = [
    # Legacy constraints + step sizes
    "L1BallConstraint",
    "L2BallConstraint",
    "L2BallTensorConstraint",
    "OrthogonalMatrixConstraint",
    "SimplexConstraint",
    "StepSize",
    "constant_step_size",
    "harmonic_step_size",
    # Legacy optimizers
    "FrankWolfeOptimizer",
    "FWState",
    "GradientDescentOptimizer",
    "ProjectedGradientDescentOptimizer",
    "GDState",
    "AdamOptimizer",
    "AdamState",
    # Torch stochastic Frank-Wolfe
    "StochasticFrankWolfe",
    "StochasticFrankWolfeMomentumPost",
    "StochasticFrankWolfeMomentumPre",
    "OrthogonalSGDM",
    # Muon
    "Muon",
    "SingleDeviceMuon",
    "MuonWithAuxAdam",
    "SingleDeviceMuonWithAuxAdam",
    # Scion
    "Scion",
    "ScionLight",
]
