"""Tests for synthetic quadratic optimization task.

This module tests:
- QuadraticProblem: loss, gradient, optimal solution
- SyntheticQuadraticTask: metrics computation
- QuadraticGradComputer: gradient computation
- NumpyVectorModel: parameter handling and copy semantics
- make_spd_quadratic: problem generation
"""

from __future__ import annotations

import numpy as np
import pytest

from models.numpy_vector import NumpyVectorModel
from tasks.synthetic_quadratic import (
    QuadraticGradComputer,
    QuadraticProblem,
    SyntheticQuadraticTask,
    make_spd_quadratic,
)

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def simple_problem() -> QuadraticProblem:
    """A simple 2D quadratic problem with known solution."""
    # f(x) = 0.5 * (2*x1^2 + x2^2) + x1 - x2
    # = x1^2 + 0.5*x2^2 + x1 - x2
    # grad = [2*x1 + 1, x2 - 1]
    # x* = [-0.5, 1.0]
    A = np.array([[2.0, 0.0], [0.0, 1.0]])
    b = np.array([1.0, -1.0])
    return QuadraticProblem(A, b)


@pytest.fixture
def random_problem() -> QuadraticProblem:
    """A random 3D quadratic problem."""
    rng = np.random.default_rng(42)
    return make_spd_quadratic(dim=3, rng=rng, cond=10.0)


# =============================================================================
# Tests for QuadraticProblem
# =============================================================================


class TestQuadraticProblem:
    """Tests for QuadraticProblem dataclass."""

    def test_loss_at_origin(self, simple_problem: QuadraticProblem) -> None:
        """Loss at origin should be 0 for f(x) = 0.5*x^T A x + b^T x."""
        x = np.zeros(2)
        assert simple_problem.loss(x) == 0.0

    def test_loss_computation(self, simple_problem: QuadraticProblem) -> None:
        """Verify loss computation for known input."""
        x = np.array([1.0, 2.0])
        # f(x) = 0.5 * (2*1 + 1*4) + 1*1 + (-1)*2
        # = 0.5 * 6 + 1 - 2 = 3 - 1 = 2.0
        expected = 0.5 * (2.0 * 1.0 + 1.0 * 4.0) + 1.0 - 2.0
        assert simple_problem.loss(x) == pytest.approx(expected)

    def test_grad_at_origin(self, simple_problem: QuadraticProblem) -> None:
        """Gradient at origin should equal b."""
        x = np.zeros(2)
        grad = simple_problem.grad(x)
        np.testing.assert_allclose(grad, simple_problem.b)

    def test_grad_computation(self, simple_problem: QuadraticProblem) -> None:
        """Verify gradient computation: grad = Ax + b."""
        x = np.array([1.0, 2.0])
        expected = simple_problem.A @ x + simple_problem.b
        grad = simple_problem.grad(x)
        np.testing.assert_allclose(grad, expected)

    def test_x_star_computation(self, simple_problem: QuadraticProblem) -> None:
        """Verify optimal solution x* = -A^{-1} b."""
        x_star = simple_problem.x_star()
        # For our simple problem: x* = [-0.5, 1.0]
        expected = np.array([-0.5, 1.0])
        np.testing.assert_allclose(x_star, expected)

    def test_grad_at_x_star_is_zero(self, simple_problem: QuadraticProblem) -> None:
        """Gradient at optimal solution should be zero."""
        x_star = simple_problem.x_star()
        grad = simple_problem.grad(x_star)
        np.testing.assert_allclose(grad, np.zeros(2), atol=1e-10)

    def test_validation_non_symmetric(self) -> None:
        """Non-symmetric A should raise ValueError."""
        A = np.array([[1.0, 0.5], [0.0, 1.0]])  # Not symmetric
        b = np.array([0.0, 0.0])
        with pytest.raises(ValueError, match="symmetric"):
            QuadraticProblem(A, b)

    def test_validation_not_positive_definite(self) -> None:
        """Non-positive definite A should raise ValueError."""
        A = np.array([[1.0, 0.0], [0.0, -1.0]])  # Has negative eigenvalue
        b = np.array([0.0, 0.0])
        with pytest.raises(ValueError, match="positive definite"):
            QuadraticProblem(A, b)

    def test_validation_dimension_mismatch(self) -> None:
        """Dimension mismatch between A and b should raise ValueError."""
        A = np.array([[1.0, 0.0], [0.0, 1.0]])
        b = np.array([0.0, 0.0, 0.0])  # Wrong size
        with pytest.raises(ValueError, match="mismatch"):
            QuadraticProblem(A, b)


class TestFiniteDifferenceGradient:
    """Finite-difference gradient verification."""

    def test_gradient_matches_finite_difference(self) -> None:
        """Analytical gradient should match finite-difference approximation."""
        rng = np.random.default_rng(123)
        problem = make_spd_quadratic(dim=3, rng=rng, cond=5.0)

        # Random test point
        x = rng.standard_normal(3)

        # Analytical gradient
        analytical_grad = problem.grad(x)

        # Finite-difference gradient
        eps = 1e-6
        fd_grad = np.zeros(3)
        for i in range(3):
            x_plus = x.copy()
            x_plus[i] += eps
            x_minus = x.copy()
            x_minus[i] -= eps
            fd_grad[i] = (problem.loss(x_plus) - problem.loss(x_minus)) / (2 * eps)

        np.testing.assert_allclose(analytical_grad, fd_grad, rtol=1e-4, atol=1e-4)


class TestOptimalityCertificate:
    """Tests verifying optimality conditions."""

    def test_grad_norm_at_optimum_is_zero(self, random_problem: QuadraticProblem) -> None:
        """Gradient norm at x* should be essentially zero."""
        x_star = random_problem.x_star()
        grad = random_problem.grad(x_star)
        grad_norm = np.linalg.norm(grad)
        assert grad_norm < 1e-8

    def test_metrics_at_optimum(self, random_problem: QuadraticProblem) -> None:
        """Task metrics at x* should show dist_to_opt ≈ 0."""
        task = SyntheticQuadraticTask(random_problem)
        model = NumpyVectorModel(random_problem.x_star())

        metrics = task.metrics(model, None)

        assert metrics["grad_norm"] < 1e-8
        assert metrics["dist_to_opt"] < 1e-10


class TestGradientDescentStep:
    """Test that a GD step decreases loss."""

    def test_loss_decreases_under_gd_step(self) -> None:
        """A small gradient descent step should decrease the loss."""
        rng = np.random.default_rng(456)
        problem = make_spd_quadratic(dim=5, rng=rng, cond=10.0)

        # Random starting point (away from optimum)
        x0 = rng.standard_normal(5) * 10.0

        # Compute gradient
        g = problem.grad(x0)

        # Learning rate: use a conservative step size
        # For quadratic, optimal lr is 1/L where L is max eigenvalue
        # We use a smaller lr to ensure decrease
        eigvals = np.linalg.eigvalsh(problem.A)
        max_eig = eigvals.max()
        lr = 0.5 / max_eig  # Conservative step size

        # Take gradient step
        x1 = x0 - lr * g

        # Loss should decrease (strictly)
        loss0 = problem.loss(x0)
        loss1 = problem.loss(x1)

        assert loss1 < loss0, f"Loss did not decrease: {loss0} -> {loss1}"


# =============================================================================
# Tests for make_spd_quadratic
# =============================================================================


class TestMakeSPDQuadratic:
    """Tests for the problem generator."""

    def test_generates_valid_problem(self) -> None:
        """Generated problem should pass validation."""
        rng = np.random.default_rng(42)
        problem = make_spd_quadratic(dim=5, rng=rng, cond=10.0)

        assert problem.dim == 5
        assert problem.A.shape == (5, 5)
        assert problem.b.shape == (5,)

    def test_condition_number(self) -> None:
        """Generated A should have approximately the requested condition number."""
        rng = np.random.default_rng(42)
        cond = 100.0
        problem = make_spd_quadratic(dim=10, rng=rng, cond=cond)

        eigvals = np.linalg.eigvalsh(problem.A)
        actual_cond = eigvals.max() / eigvals.min()

        assert actual_cond == pytest.approx(cond, rel=0.01)

    def test_determinism_with_same_seed(self) -> None:
        """Same RNG seed should produce identical problems."""
        rng1 = np.random.default_rng(42)
        problem1 = make_spd_quadratic(dim=5, rng=rng1, cond=10.0)

        rng2 = np.random.default_rng(42)
        problem2 = make_spd_quadratic(dim=5, rng=rng2, cond=10.0)

        np.testing.assert_array_equal(problem1.A, problem2.A)
        np.testing.assert_array_equal(problem1.b, problem2.b)

    def test_invalid_dim(self) -> None:
        """dim < 1 should raise ValueError."""
        rng = np.random.default_rng(42)
        with pytest.raises(ValueError, match="dim must be >= 1"):
            make_spd_quadratic(dim=0, rng=rng)

    def test_invalid_cond(self) -> None:
        """cond < 1 should raise ValueError."""
        rng = np.random.default_rng(42)
        with pytest.raises(ValueError, match="cond must be >= 1"):
            make_spd_quadratic(dim=5, rng=rng, cond=0.5)


# =============================================================================
# Tests for SyntheticQuadraticTask
# =============================================================================


class TestSyntheticQuadraticTask:
    """Tests for the task wrapper."""

    def test_sample_batch_returns_none(self, random_problem: QuadraticProblem) -> None:
        """sample_batch should return None (batchless task)."""
        task = SyntheticQuadraticTask(random_problem)
        rng = np.random.default_rng(42)
        assert task.sample_batch(rng=rng) is None

    def test_loss_matches_problem(self, random_problem: QuadraticProblem) -> None:
        """Task loss should match problem loss."""
        task = SyntheticQuadraticTask(random_problem)
        x = np.random.default_rng(42).standard_normal(random_problem.dim)
        model = NumpyVectorModel(x)

        assert task.loss(model, None) == pytest.approx(random_problem.loss(x))

    def test_metrics_keys(self, random_problem: QuadraticProblem) -> None:
        """Metrics should contain expected keys."""
        task = SyntheticQuadraticTask(random_problem)
        model = NumpyVectorModel(np.zeros(random_problem.dim))

        metrics = task.metrics(model, None)

        assert "loss" in metrics
        assert "grad_norm" in metrics
        assert "dist_to_opt" in metrics

    def test_x_star_caching(self, random_problem: QuadraticProblem) -> None:
        """x_star property should be cached."""
        task = SyntheticQuadraticTask(random_problem)

        x_star1 = task.x_star
        x_star2 = task.x_star

        # Should be the same object (cached)
        assert x_star1 is x_star2


# =============================================================================
# Tests for QuadraticGradComputer
# =============================================================================


class TestQuadraticGradComputer:
    """Tests for the gradient computer."""

    def test_grad_matches_problem(self, random_problem: QuadraticProblem) -> None:
        """GradComputer should return same gradient as problem.grad()."""
        task = SyntheticQuadraticTask(random_problem)
        grad_computer = QuadraticGradComputer()

        x = np.random.default_rng(42).standard_normal(random_problem.dim)
        model = NumpyVectorModel(x)

        computed_grad = grad_computer.grad(task, model, None)
        expected_grad = random_problem.grad(x)

        np.testing.assert_allclose(computed_grad, expected_grad)


# =============================================================================
# Tests for NumpyVectorModel
# =============================================================================


class TestNumpyVectorModel:
    """Tests for the model class."""

    def test_initialization(self) -> None:
        """Model should initialize with given parameters."""
        x = np.array([1.0, 2.0, 3.0])
        model = NumpyVectorModel(x)

        assert model.dim == 3
        np.testing.assert_array_equal(model.parameters_vector(), x)

    def test_parameters_vector_returns_copy(self) -> None:
        """parameters_vector() should return a copy, not the internal array."""
        x = np.array([1.0, 2.0, 3.0])
        model = NumpyVectorModel(x)

        params = model.parameters_vector()
        params[0] = 999.0  # Mutate the returned array

        # Internal state should be unchanged
        assert model.parameters_vector()[0] == 1.0

    def test_set_parameters_vector_makes_copy(self) -> None:
        """set_parameters_vector() should store a copy."""
        x = np.array([1.0, 2.0, 3.0])
        model = NumpyVectorModel(np.zeros(3))

        model.set_parameters_vector(x)
        x[0] = 999.0  # Mutate the original array

        # Internal state should be unchanged
        assert model.parameters_vector()[0] == 1.0

    def test_initialization_makes_copy(self) -> None:
        """__init__ should store a copy of the input."""
        x = np.array([1.0, 2.0, 3.0])
        model = NumpyVectorModel(x)

        x[0] = 999.0  # Mutate the original array

        # Internal state should be unchanged
        assert model.parameters_vector()[0] == 1.0

    def test_validation_non_1d(self) -> None:
        """Non-1D input should raise ValueError."""
        x = np.array([[1.0, 2.0], [3.0, 4.0]])
        with pytest.raises(ValueError, match="1-dimensional"):
            NumpyVectorModel(x)

    def test_set_parameters_shape_mismatch(self) -> None:
        """set_parameters_vector with wrong shape should raise ValueError."""
        model = NumpyVectorModel(np.zeros(3))
        with pytest.raises(ValueError, match="Shape mismatch"):
            model.set_parameters_vector(np.zeros(5))

    def test_float64_dtype(self) -> None:
        """Parameters should be stored as float64."""
        x = np.array([1, 2, 3], dtype=np.int32)
        model = NumpyVectorModel(x)

        assert model.parameters_vector().dtype == np.float64
