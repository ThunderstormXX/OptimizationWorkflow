"""Tests for Gradient Descent optimizers.

This module tests:
- GradientDescentOptimizer correctness and convergence
- ProjectedGradientDescentOptimizer feasibility and convergence
"""

from __future__ import annotations

import numpy as np
import pytest

from environments.single_process import SingleProcessEnvironment
from models.numpy_vector import NumpyVectorModel
from optim.constraints import L2BallConstraint, SimplexConstraint
from optim.gradient_descent import (
    GDState,
    GradientDescentOptimizer,
    ProjectedGradientDescentOptimizer,
)
from tasks.synthetic_quadratic import (
    QuadraticGradComputer,
    SyntheticQuadraticTask,
    make_spd_quadratic,
)

# =============================================================================
# Tests for GradientDescentOptimizer
# =============================================================================


class TestGradientDescentOptimizer:
    """Tests for vanilla gradient descent."""

    def test_init_requires_positive_lr(self) -> None:
        """Learning rate must be positive."""
        with pytest.raises(ValueError, match="positive"):
            GradientDescentOptimizer(lr=0.0)

        with pytest.raises(ValueError, match="positive"):
            GradientDescentOptimizer(lr=-0.1)

    def test_init_state(self) -> None:
        """init_state should return GDState with t=0."""
        optimizer = GradientDescentOptimizer(lr=0.1)
        model = NumpyVectorModel(np.array([1.0, 2.0]))

        state = optimizer.init_state(model)

        assert isinstance(state, GDState)
        assert state.t == 0

    def test_loss_decreases_on_quadratic(self) -> None:
        """GD should decrease loss on a quadratic problem with stable lr."""
        dim = 5
        rng = np.random.default_rng(42)
        problem = make_spd_quadratic(dim=dim, rng=rng, cond=10.0)
        task = SyntheticQuadraticTask(problem)
        grad_computer = QuadraticGradComputer()

        # Compute stable learning rate: lr = 1 / max_eigenvalue
        max_eig = float(np.linalg.eigvalsh(problem.A).max())
        lr = 0.5 / max_eig  # Conservative step size

        optimizer = GradientDescentOptimizer(lr=lr)

        # Start from random point
        x0 = rng.standard_normal(dim)
        model = NumpyVectorModel(x0)

        # Build environment
        env: SingleProcessEnvironment[None, None, GDState] = SingleProcessEnvironment(
            task=task,  # type: ignore[arg-type]
            model=model,
            optimizer=optimizer,
            grad_computer=grad_computer,  # type: ignore[arg-type]
        )

        env.reset(seed=0)

        # Get initial loss
        initial_loss = task.loss(model, None)
        initial_grad_norm = float(np.linalg.norm(problem.grad(model.parameters_vector())))

        # Run 20 steps
        env.run(steps=20)

        # Get final loss
        final_loss = task.loss(model, None)
        final_grad_norm = float(np.linalg.norm(problem.grad(model.parameters_vector())))

        # Loss should decrease
        assert final_loss < initial_loss, f"Loss did not decrease: {initial_loss} -> {final_loss}"

        # Gradient norm should decrease
        assert final_grad_norm < initial_grad_norm, (
            f"Grad norm did not decrease: {initial_grad_norm} -> {final_grad_norm}"
        )

    def test_determinism(self) -> None:
        """Same seed should produce identical results."""
        dim = 4
        rng = np.random.default_rng(42)
        problem = make_spd_quadratic(dim=dim, rng=rng, cond=5.0)
        task = SyntheticQuadraticTask(problem)
        grad_computer = QuadraticGradComputer()

        lr = 0.01
        x0 = np.ones(dim)

        def run_gd() -> np.ndarray:
            model = NumpyVectorModel(x0.copy())
            optimizer = GradientDescentOptimizer(lr=lr)
            env: SingleProcessEnvironment[None, None, GDState] = SingleProcessEnvironment(
                task=task,  # type: ignore[arg-type]
                model=model,
                optimizer=optimizer,
                grad_computer=grad_computer,  # type: ignore[arg-type]
            )
            env.reset(seed=123)
            env.run(steps=10)
            return model.parameters_vector()

        params1 = run_gd()
        params2 = run_gd()

        np.testing.assert_allclose(params1, params2)


# =============================================================================
# Tests for ProjectedGradientDescentOptimizer
# =============================================================================


class TestProjectedGradientDescentOptimizer:
    """Tests for projected gradient descent."""

    def test_init_requires_positive_lr(self) -> None:
        """Learning rate must be positive."""
        constraint = L2BallConstraint(radius=1.0)

        with pytest.raises(ValueError, match="positive"):
            ProjectedGradientDescentOptimizer(lr=0.0, constraint=constraint)

    def test_init_with_simplex_raises_on_first_projection(self) -> None:
        """SimplexConstraint's project() raises NotImplementedError."""
        # SimplexConstraint has project() but it raises NotImplementedError
        constraint = SimplexConstraint(dim=5)
        optimizer = ProjectedGradientDescentOptimizer(lr=0.1, constraint=constraint)

        model = NumpyVectorModel(np.ones(5))

        # Should raise when init_state tries to project
        with pytest.raises(NotImplementedError, match="Simplex projection"):
            optimizer.init_state(model)

    def test_init_state_projects_initial_params(self) -> None:
        """init_state should project initial parameters onto constraint."""
        radius = 1.0
        constraint = L2BallConstraint(radius=radius)
        optimizer = ProjectedGradientDescentOptimizer(lr=0.1, constraint=constraint)

        # Start outside the ball
        x0 = np.array([10.0, 0.0, 0.0])  # norm = 10 >> radius
        model = NumpyVectorModel(x0)

        state = optimizer.init_state(model)

        # After init_state, params should be on the ball
        x_after = model.parameters_vector()
        norm_after = float(np.linalg.norm(x_after))

        assert norm_after <= radius + 1e-10, (
            f"Params not projected: norm={norm_after}, radius={radius}"
        )
        assert isinstance(state, GDState)
        assert state.t == 0

    def test_feasibility_preserved(self) -> None:
        """PGD should keep parameters feasible throughout optimization."""
        dim = 5
        radius = 2.0
        rng = np.random.default_rng(42)
        problem = make_spd_quadratic(dim=dim, rng=rng, cond=10.0)
        task = SyntheticQuadraticTask(problem)
        grad_computer = QuadraticGradComputer()

        constraint = L2BallConstraint(radius=radius)
        lr = 0.1

        optimizer = ProjectedGradientDescentOptimizer(lr=lr, constraint=constraint)

        # Start outside the ball
        x0 = rng.standard_normal(dim) * 10  # Large initial point
        model = NumpyVectorModel(x0)

        env: SingleProcessEnvironment[None, None, GDState] = SingleProcessEnvironment(
            task=task,  # type: ignore[arg-type]
            model=model,
            optimizer=optimizer,
            grad_computer=grad_computer,  # type: ignore[arg-type]
        )

        env.reset(seed=0)

        # Check feasibility after each step
        for _ in range(20):
            env.step()
            x = model.parameters_vector()
            norm = float(np.linalg.norm(x))
            assert norm <= radius + 1e-10, f"Feasibility violated: norm={norm}, R={radius}"

    def test_loss_decreases_on_quadratic(self) -> None:
        """PGD should decrease loss on a quadratic problem."""
        dim = 5
        radius = 5.0  # Large enough to not be too restrictive
        rng = np.random.default_rng(42)
        problem = make_spd_quadratic(dim=dim, rng=rng, cond=10.0)
        task = SyntheticQuadraticTask(problem)
        grad_computer = QuadraticGradComputer()

        constraint = L2BallConstraint(radius=radius)

        # Conservative learning rate
        max_eig = float(np.linalg.eigvalsh(problem.A).max())
        lr = 0.5 / max_eig

        optimizer = ProjectedGradientDescentOptimizer(lr=lr, constraint=constraint)

        # Start from point inside the ball
        x0 = rng.standard_normal(dim)
        x0 = x0 / np.linalg.norm(x0) * (radius * 0.5)  # Normalize to half radius
        model = NumpyVectorModel(x0)

        env: SingleProcessEnvironment[None, None, GDState] = SingleProcessEnvironment(
            task=task,  # type: ignore[arg-type]
            model=model,
            optimizer=optimizer,
            grad_computer=grad_computer,  # type: ignore[arg-type]
        )

        env.reset(seed=0)

        initial_loss = task.loss(model, None)

        env.run(steps=20)

        final_loss = task.loss(model, None)

        assert final_loss < initial_loss, f"Loss did not decrease: {initial_loss} -> {final_loss}"

    def test_determinism(self) -> None:
        """Same seed should produce identical results."""
        dim = 4
        radius = 2.0
        rng = np.random.default_rng(42)
        problem = make_spd_quadratic(dim=dim, rng=rng, cond=5.0)
        task = SyntheticQuadraticTask(problem)
        grad_computer = QuadraticGradComputer()

        constraint = L2BallConstraint(radius=radius)
        lr = 0.05
        x0 = np.ones(dim) * 5  # Start outside ball

        def run_pgd() -> np.ndarray:
            model = NumpyVectorModel(x0.copy())
            optimizer = ProjectedGradientDescentOptimizer(lr=lr, constraint=constraint)
            env: SingleProcessEnvironment[None, None, GDState] = SingleProcessEnvironment(
                task=task,  # type: ignore[arg-type]
                model=model,
                optimizer=optimizer,
                grad_computer=grad_computer,  # type: ignore[arg-type]
            )
            env.reset(seed=456)
            env.run(steps=10)
            return model.parameters_vector()

        params1 = run_pgd()
        params2 = run_pgd()

        np.testing.assert_allclose(params1, params2)


# =============================================================================
# Tests for GDState
# =============================================================================


class TestGDState:
    """Tests for GDState dataclass."""

    def test_default_t(self) -> None:
        """Default t should be 0."""
        state = GDState()
        assert state.t == 0

    def test_custom_t(self) -> None:
        """Custom t should be preserved."""
        state = GDState(t=5)
        assert state.t == 5
