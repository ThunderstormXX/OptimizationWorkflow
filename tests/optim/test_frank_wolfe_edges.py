from __future__ import annotations

import math
import numpy as np
import pytest

from models.numpy_vector import NumpyVectorModel
from optim.legacy_frankwolfe import FrankWolfeOptimizer, FWState
from tasks.synthetic_quadratic import QuadraticGradComputer, QuadraticProblem, SyntheticQuadraticTask


class _NoProjectConstraint:
    def __init__(self, radius: float = 1.0) -> None:
        self.radius = radius

    def lmo(self, grad: np.ndarray) -> np.ndarray:
        if np.all(grad == 0):
            return np.zeros_like(grad)
        return -self.radius * grad / np.linalg.norm(grad)


def test_fw_init_state_without_project() -> None:
    constraint = _NoProjectConstraint(radius=1.0)
    optimizer = FrankWolfeOptimizer(constraint=constraint, step_size=lambda t: 0.5)
    model = NumpyVectorModel(np.array([2.0, 0.0]))

    state = optimizer.init_state(model)
    assert isinstance(state, FWState)
    # No projection should occur
    assert np.allclose(model.parameters_vector(), np.array([2.0, 0.0]))


def test_fw_step_size_validation() -> None:
    task = SyntheticQuadraticTask(QuadraticProblem(np.eye(2), np.zeros(2)))
    model = NumpyVectorModel(np.array([1.0, 1.0]))
    grad_computer = QuadraticGradComputer()
    constraint = _NoProjectConstraint(radius=1.0)

    optimizer_inf = FrankWolfeOptimizer(constraint=constraint, step_size=lambda t: math.inf)
    with pytest.raises(ValueError):
        optimizer_inf.step(
            task=task,
            model=model,
            batch=None,
            grad_computer=grad_computer,
            state=FWState(t=0),
            rng=np.random.default_rng(0),
        )

    optimizer_bad = FrankWolfeOptimizer(constraint=constraint, step_size=lambda t: 2.0)
    with pytest.raises(ValueError):
        optimizer_bad.step(
            task=task,
            model=model,
            batch=None,
            grad_computer=grad_computer,
            state=FWState(t=0),
            rng=np.random.default_rng(0),
        )
