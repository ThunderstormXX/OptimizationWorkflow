from __future__ import annotations

import numpy as np
import pytest

from models.numpy_vector import NumpyVectorModel
from optim.adam import AdamOptimizer
from optim.legacy_frankwolfe import (
    L1BallConstraint,
    L2BallConstraint,
    ProjectedGradientDescentOptimizer,
    SimplexConstraint,
)
from tasks.synthetic_quadratic import QuadraticGradComputer, QuadraticProblem, SyntheticQuadraticTask


def _build_quadratic_task() -> SyntheticQuadraticTask:
    A = np.eye(3)
    b = np.array([1.0, -2.0, 0.5])
    return SyntheticQuadraticTask(QuadraticProblem(A, b))


def test_adam_optimizer_step_updates_state() -> None:
    task = _build_quadratic_task()
    model = NumpyVectorModel(np.array([0.5, -0.5, 1.0]))
    grad_computer = QuadraticGradComputer()

    optimizer = AdamOptimizer(lr=0.1)
    state = optimizer.init_state(model)

    new_state, result = optimizer.step(
        task=task,
        model=model,
        batch=None,
        grad_computer=grad_computer,
        state=state,
        rng=np.random.default_rng(0),
    )

    assert new_state.t == 1
    assert result.loss == pytest.approx(task.loss(model, None))


def test_adam_optimizer_invalid_params() -> None:
    with pytest.raises(ValueError):
        AdamOptimizer(lr=0.0)
    with pytest.raises(ValueError):
        AdamOptimizer(beta1=1.5)
    with pytest.raises(ValueError):
        AdamOptimizer(beta2=-0.1)


def test_constraints_behavior() -> None:
    with pytest.raises(ValueError):
        L1BallConstraint(radius=0.0)
    with pytest.raises(ValueError):
        L2BallConstraint(radius=-1.0)

    l1 = L1BallConstraint(radius=2.0)
    grad = np.array([0.0, -3.0, 1.0])
    assert np.allclose(l1.lmo(grad), np.array([0.0, 2.0, 0.0]))
    assert np.allclose(l1.lmo(np.zeros(3)), np.zeros(3))

    inside = np.array([0.5, -0.5, 0.0])
    projected_inside = l1.project(inside)
    assert np.allclose(projected_inside, inside)

    outside = np.array([3.0, 0.0, 0.0])
    projected_outside = l1.project(outside)
    assert pytest.approx(np.linalg.norm(projected_outside, ord=1)) == 2.0

    l2 = L2BallConstraint(radius=1.0)
    assert np.allclose(l2.lmo(np.zeros(2)), np.zeros(2))
    proj = l2.project(np.array([3.0, 4.0]))
    assert pytest.approx(np.linalg.norm(proj)) == 1.0

    simplex = SimplexConstraint(dim=3)
    assert np.allclose(simplex.lmo(np.array([3.0, 1.0, 2.0])), np.array([0.0, 1.0, 0.0]))
    with pytest.raises(NotImplementedError):
        simplex.project(np.array([0.2, 0.3, 0.5]))


def test_pgd_requires_projectable_constraint() -> None:
    class NoProject:
        def lmo(self, grad: np.ndarray) -> np.ndarray:
            return grad

    with pytest.raises(ValueError):
        ProjectedGradientDescentOptimizer(lr=0.1, constraint=NoProject())
