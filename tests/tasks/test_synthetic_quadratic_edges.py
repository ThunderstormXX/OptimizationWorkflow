from __future__ import annotations

import numpy as np
import pytest

from tasks.synthetic_quadratic import QuadraticProblem, make_spd_quadratic


def test_quadratic_problem_validation_errors() -> None:
    with pytest.raises(ValueError, match="A must be 2D"):
        QuadraticProblem(A=np.array([1.0, 2.0]), b=np.array([1.0, 2.0]))

    with pytest.raises(ValueError, match="A must be square"):
        QuadraticProblem(A=np.ones((2, 3)), b=np.ones(2))

    with pytest.raises(ValueError, match="b must be 1D"):
        QuadraticProblem(A=np.eye(2), b=np.ones((2, 1)))

    with pytest.raises(ValueError, match="A must be symmetric"):
        QuadraticProblem(A=np.array([[1.0, 2.0], [0.0, 1.0]]), b=np.ones(2))


def test_quadratic_problem_large_dim_spd_check() -> None:
    # Non-PD symmetric matrix (zeros) should fail Cholesky for dim > 100
    A = np.zeros((101, 101))
    b = np.zeros(101)
    with pytest.raises(ValueError, match="positive definite"):
        QuadraticProblem(A=A, b=b)


def test_make_spd_quadratic_dim_one() -> None:
    rng = np.random.default_rng(0)
    problem = make_spd_quadratic(dim=1, rng=rng, cond=5.0)
    assert problem.A.shape == (1, 1)
    assert problem.b.shape == (1,)
