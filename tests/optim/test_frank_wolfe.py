"""Tests for Frank-Wolfe optimizer and constraints.

This module tests:
- L2BallConstraint: LMO and projection correctness
- SimplexConstraint: LMO correctness
- FrankWolfeOptimizer: feasibility preservation, convergence, determinism
"""

from __future__ import annotations

import numpy as np
import pytest

from environments.single_process import SingleProcessEnvironment
from models.numpy_vector import NumpyVectorModel
from optim.constraints import L2BallConstraint, SimplexConstraint
from optim.frank_wolfe import (
    FrankWolfeOptimizer,
    FWState,
    constant_step_size,
    harmonic_step_size,
)
from tasks.synthetic_quadratic import (
    QuadraticGradComputer,
    SyntheticQuadraticTask,
    make_spd_quadratic,
)

# =============================================================================
# Tests for L2BallConstraint
# =============================================================================


class TestL2BallConstraint:
    """Tests for L2 ball constraint."""

    def test_lmo_returns_point_on_boundary(self) -> None:
        """LMO should return a point with norm equal to radius."""
        constraint = L2BallConstraint(radius=2.5)
        rng = np.random.default_rng(42)

        for _ in range(10):
            grad = rng.standard_normal(5)
            s = constraint.lmo(grad)
            norm = np.linalg.norm(s)
            assert norm == pytest.approx(2.5, rel=1e-10)

    def test_lmo_direction(self) -> None:
        """LMO should return point in opposite direction of gradient."""
        constraint = L2BallConstraint(radius=1.0)
        grad = np.array([3.0, 4.0])  # norm = 5

        s = constraint.lmo(grad)

        # s should be -grad/||grad|| * radius = [-0.6, -0.8]
        expected = np.array([-0.6, -0.8])
        np.testing.assert_allclose(s, expected)

    def test_lmo_zero_gradient(self) -> None:
        """LMO with zero gradient should return zeros."""
        constraint = L2BallConstraint(radius=1.0)
        grad = np.zeros(5)

        s = constraint.lmo(grad)

        np.testing.assert_array_equal(s, np.zeros(5))

    def test_project_inside_ball(self) -> None:
        """Points inside ball should be unchanged by projection."""
        constraint = L2BallConstraint(radius=2.0)
        x = np.array([0.5, 0.5, 0.5])  # norm ≈ 0.866 < 2.0

        projected = constraint.project(x)

        np.testing.assert_allclose(projected, x)

    def test_project_outside_ball(self) -> None:
        """Points outside ball should be projected to boundary."""
        constraint = L2BallConstraint(radius=1.0)
        x = np.array([3.0, 4.0])  # norm = 5

        projected = constraint.project(x)

        # Should have norm = 1.0
        assert np.linalg.norm(projected) == pytest.approx(1.0)
        # Should be in same direction as x
        expected = np.array([0.6, 0.8])
        np.testing.assert_allclose(projected, expected)

    def test_project_on_boundary(self) -> None:
        """Points on boundary should be unchanged."""
        constraint = L2BallConstraint(radius=5.0)
        x = np.array([3.0, 4.0])  # norm = 5

        projected = constraint.project(x)

        np.testing.assert_allclose(projected, x)

    def test_project_returns_copy(self) -> None:
        """Projection should return a copy, not modify input."""
        constraint = L2BallConstraint(radius=2.0)
        x = np.array([0.5, 0.5])
        x_original = x.copy()

        projected = constraint.project(x)
        projected[0] = 999.0  # Mutate returned array

        np.testing.assert_array_equal(x, x_original)

    def test_invalid_radius(self) -> None:
        """Radius <= 0 should raise ValueError."""
        with pytest.raises(ValueError, match="positive"):
            L2BallConstraint(radius=0.0)
        with pytest.raises(ValueError, match="positive"):
            L2BallConstraint(radius=-1.0)


# =============================================================================
# Tests for SimplexConstraint
# =============================================================================


class TestSimplexConstraint:
    """Tests for simplex constraint."""

    def test_lmo_returns_one_hot(self) -> None:
        """LMO should return a one-hot vector."""
        constraint = SimplexConstraint(dim=4)
        grad = np.array([3.0, 1.0, 2.0, 5.0])

        s = constraint.lmo(grad)

        # Should be one-hot at index 1 (minimum gradient entry)
        expected = np.array([0.0, 1.0, 0.0, 0.0])
        np.testing.assert_array_equal(s, expected)

    def test_lmo_sum_is_one(self) -> None:
        """LMO result should sum to 1 (on simplex)."""
        constraint = SimplexConstraint(dim=5)
        rng = np.random.default_rng(42)

        for _ in range(10):
            grad = rng.standard_normal(5)
            s = constraint.lmo(grad)
            assert np.sum(s) == pytest.approx(1.0)

    def test_lmo_all_nonnegative(self) -> None:
        """LMO result should have all non-negative entries."""
        constraint = SimplexConstraint(dim=5)
        rng = np.random.default_rng(42)

        for _ in range(10):
            grad = rng.standard_normal(5)
            s = constraint.lmo(grad)
            assert np.all(s >= 0)

    def test_lmo_example(self) -> None:
        """Test specific example: grad = [3, 1, 2] -> e_1."""
        constraint = SimplexConstraint(dim=3)
        grad = np.array([3.0, 1.0, 2.0])

        s = constraint.lmo(grad)

        expected = np.array([0.0, 1.0, 0.0])
        np.testing.assert_array_equal(s, expected)

    def test_project_not_implemented(self) -> None:
        """Simplex projection should raise NotImplementedError."""
        constraint = SimplexConstraint(dim=3)
        x = np.array([0.5, 0.3, 0.2])

        with pytest.raises(NotImplementedError):
            constraint.project(x)

    def test_invalid_dim(self) -> None:
        """dim < 1 should raise ValueError."""
        with pytest.raises(ValueError, match="dim must be >= 1"):
            SimplexConstraint(dim=0)


# =============================================================================
# Tests for step size schedules
# =============================================================================


class TestStepSizeSchedules:
    """Tests for step size schedule factories."""

    def test_constant_step_size(self) -> None:
        """Constant schedule should return same value."""
        schedule = constant_step_size(0.5)

        assert schedule(0) == 0.5
        assert schedule(10) == 0.5
        assert schedule(1000) == 0.5

    def test_constant_step_size_invalid(self) -> None:
        """Invalid gamma should raise ValueError."""
        with pytest.raises(ValueError, match="\\(0, 1\\]"):
            constant_step_size(0.0)
        with pytest.raises(ValueError, match="\\(0, 1\\]"):
            constant_step_size(1.5)
        with pytest.raises(ValueError, match="\\(0, 1\\]"):
            constant_step_size(-0.1)

    def test_harmonic_step_size(self) -> None:
        """Harmonic schedule should return 2/(t+2)."""
        schedule = harmonic_step_size()

        assert schedule(0) == pytest.approx(1.0)  # 2/2
        assert schedule(1) == pytest.approx(2 / 3)  # 2/3
        assert schedule(2) == pytest.approx(0.5)  # 2/4
        assert schedule(8) == pytest.approx(0.2)  # 2/10

    def test_harmonic_decreasing(self) -> None:
        """Harmonic schedule should be strictly decreasing."""
        schedule = harmonic_step_size()

        for t in range(100):
            assert schedule(t) > schedule(t + 1)


# =============================================================================
# Tests for FrankWolfeOptimizer
# =============================================================================


class TestFrankWolfeOptimizer:
    """Tests for Frank-Wolfe optimizer."""

    def test_init_state_returns_fw_state(self) -> None:
        """init_state should return FWState with t=0."""
        constraint = L2BallConstraint(radius=1.0)
        optimizer = FrankWolfeOptimizer(
            constraint=constraint,
            step_size=harmonic_step_size(),
        )
        model = NumpyVectorModel(np.zeros(5))

        state = optimizer.init_state(model)

        assert isinstance(state, FWState)
        assert state.t == 0

    def test_init_state_projects_infeasible(self) -> None:
        """init_state should project infeasible starting point."""
        constraint = L2BallConstraint(radius=1.0)
        optimizer = FrankWolfeOptimizer(
            constraint=constraint,
            step_size=harmonic_step_size(),
        )
        # Start outside the ball
        x0 = np.array([3.0, 4.0])  # norm = 5
        model = NumpyVectorModel(x0)

        optimizer.init_state(model)

        # Model should now be on the boundary
        x = model.parameters_vector()
        assert np.linalg.norm(x) == pytest.approx(1.0)


# =============================================================================
# Integration tests: FW + SingleProcessEnvironment
# =============================================================================


class TestFWIntegration:
    """Integration tests for Frank-Wolfe with SingleProcessEnvironment."""

    def test_feasibility_preservation_l2_ball(self) -> None:
        """FW should preserve feasibility throughout optimization."""
        # Create problem
        rng = np.random.default_rng(42)
        problem = make_spd_quadratic(dim=5, rng=rng, cond=10.0)
        task = SyntheticQuadraticTask(problem)
        grad_computer = QuadraticGradComputer()

        # Compute optimal solution and set radius
        x_star = task.x_star
        radius = 1.25 * float(np.linalg.norm(x_star))

        # Create constraint and optimizer
        constraint = L2BallConstraint(radius=radius)
        optimizer: FrankWolfeOptimizer[None, None] = FrankWolfeOptimizer(
            constraint=constraint,
            step_size=harmonic_step_size(),
        )

        # Start intentionally outside the ball
        x0 = 10.0 * x_star
        model = NumpyVectorModel(x0)

        # Create environment
        env: SingleProcessEnvironment[None, None, FWState] = SingleProcessEnvironment(
            task=task,
            model=model,
            optimizer=optimizer,
            grad_computer=grad_computer,
        )

        # Reset and check initial projection
        env.reset(seed=123)
        x_after_init = model.parameters_vector()
        assert np.linalg.norm(x_after_init) <= radius + 1e-8, (
            f"After init: ||x|| = {np.linalg.norm(x_after_init)} > radius = {radius}"
        )

        # Run optimization
        history = env.run(steps=50)

        # Check feasibility after run
        x_final = model.parameters_vector()
        assert np.linalg.norm(x_final) <= radius + 1e-8, (
            f"After run: ||x|| = {np.linalg.norm(x_final)} > radius = {radius}"
        )

        # Check that we have correct history length
        assert len(history) == 50

    def test_loss_decreases(self) -> None:
        """FW should decrease loss over iterations."""
        # Create problem
        rng = np.random.default_rng(42)
        problem = make_spd_quadratic(dim=5, rng=rng, cond=10.0)
        task = SyntheticQuadraticTask(problem)
        grad_computer = QuadraticGradComputer()

        # Set radius large enough to include optimum
        x_star = task.x_star
        radius = 2.0 * float(np.linalg.norm(x_star))

        # Create constraint and optimizer
        constraint = L2BallConstraint(radius=radius)
        optimizer: FrankWolfeOptimizer[None, None] = FrankWolfeOptimizer(
            constraint=constraint,
            step_size=harmonic_step_size(),
        )

        # Start from a point on the boundary (feasible)
        x0 = radius * np.ones(5) / np.sqrt(5)
        model = NumpyVectorModel(x0)

        # Create environment
        env: SingleProcessEnvironment[None, None, FWState] = SingleProcessEnvironment(
            task=task,
            model=model,
            optimizer=optimizer,
            grad_computer=grad_computer,
        )

        # Reset and get initial loss
        env.reset(seed=456)
        initial_metrics = task.metrics(model, None)
        initial_loss = initial_metrics["loss"]
        initial_dist = initial_metrics["dist_to_opt"]

        # Run optimization
        history = env.run(steps=50)

        # Get final metrics
        final_metrics = task.metrics(model, None)
        final_loss = final_metrics["loss"]
        final_dist = final_metrics["dist_to_opt"]

        # Loss should decrease
        assert final_loss < initial_loss, f"Loss did not decrease: {initial_loss} -> {final_loss}"

        # Distance to optimum should decrease
        assert final_dist < initial_dist, (
            f"Distance to optimum did not decrease: {initial_dist} -> {final_dist}"
        )

        # Mean loss from history should be reasonable
        assert history.mean_loss() < initial_loss

    def test_determinism(self) -> None:
        """Same seed should produce identical trajectories."""
        # Create problem
        rng = np.random.default_rng(42)
        problem = make_spd_quadratic(dim=5, rng=rng, cond=10.0)

        def make_env() -> tuple[
            SingleProcessEnvironment[None, None, FWState],
            NumpyVectorModel,
        ]:
            """Create fresh environment and model."""
            task = SyntheticQuadraticTask(problem)
            grad_computer = QuadraticGradComputer()
            x_star = task.x_star
            radius = 1.5 * float(np.linalg.norm(x_star))
            constraint = L2BallConstraint(radius=radius)
            optimizer: FrankWolfeOptimizer[None, None] = FrankWolfeOptimizer(
                constraint=constraint,
                step_size=harmonic_step_size(),
            )
            x0 = np.ones(5)
            model = NumpyVectorModel(x0)
            env: SingleProcessEnvironment[None, None, FWState] = SingleProcessEnvironment(
                task=task,
                model=model,
                optimizer=optimizer,
                grad_computer=grad_computer,
            )
            return env, model

        # First run
        env1, model1 = make_env()
        env1.reset(seed=789)
        history1 = env1.run(steps=20)
        params1 = model1.parameters_vector()

        # Second run with same seed
        env2, model2 = make_env()
        env2.reset(seed=789)
        history2 = env2.run(steps=20)
        params2 = model2.parameters_vector()

        # Should be identical
        np.testing.assert_allclose(params1, params2)
        assert history1.mean_loss() == pytest.approx(history2.mean_loss())

    def test_different_seeds_different_results(self) -> None:
        """Different seeds should produce different trajectories.

        Note: For the batchless quadratic task, the seed doesn't affect
        the optimization trajectory since there's no randomness in the
        gradient computation. This test verifies that the environment
        handles different seeds correctly (even if results are the same
        for deterministic tasks).
        """
        # For batchless tasks, different seeds don't change the trajectory
        # This test just verifies the mechanism works without errors
        rng = np.random.default_rng(42)
        problem = make_spd_quadratic(dim=5, rng=rng, cond=10.0)
        task = SyntheticQuadraticTask(problem)
        grad_computer = QuadraticGradComputer()

        x_star = task.x_star
        radius = 1.5 * float(np.linalg.norm(x_star))
        constraint = L2BallConstraint(radius=radius)

        # Run with seed 1
        optimizer1: FrankWolfeOptimizer[None, None] = FrankWolfeOptimizer(
            constraint=constraint,
            step_size=harmonic_step_size(),
        )
        model1 = NumpyVectorModel(np.ones(5))
        env1: SingleProcessEnvironment[None, None, FWState] = SingleProcessEnvironment(
            task=task,
            model=model1,
            optimizer=optimizer1,
            grad_computer=grad_computer,
        )
        env1.reset(seed=1)
        env1.run(steps=10)

        # Run with seed 2
        optimizer2: FrankWolfeOptimizer[None, None] = FrankWolfeOptimizer(
            constraint=constraint,
            step_size=harmonic_step_size(),
        )
        model2 = NumpyVectorModel(np.ones(5))
        env2: SingleProcessEnvironment[None, None, FWState] = SingleProcessEnvironment(
            task=task,
            model=model2,
            optimizer=optimizer2,
            grad_computer=grad_computer,
        )
        env2.reset(seed=2)
        env2.run(steps=10)

        # For batchless quadratic, results should actually be the same
        # since there's no randomness in the optimization
        np.testing.assert_allclose(
            model1.parameters_vector(),
            model2.parameters_vector(),
        )


class TestFWWithSimplex:
    """Tests for Frank-Wolfe with simplex constraint."""

    def test_simplex_feasibility(self) -> None:
        """FW should preserve simplex feasibility."""
        # Create a simple quadratic problem
        dim = 5
        A = np.eye(dim) * 2.0  # Simple diagonal SPD
        b = np.zeros(dim)
        from tasks.synthetic_quadratic import QuadraticProblem

        problem = QuadraticProblem(A=A, b=b)
        task = SyntheticQuadraticTask(problem)
        grad_computer = QuadraticGradComputer()

        # Create simplex constraint
        constraint = SimplexConstraint(dim=dim)
        optimizer: FrankWolfeOptimizer[None, None] = FrankWolfeOptimizer(
            constraint=constraint,
            step_size=harmonic_step_size(),
        )

        # Start from a feasible point (uniform distribution)
        x0 = np.ones(dim) / dim
        model = NumpyVectorModel(x0)

        # Create environment
        env: SingleProcessEnvironment[None, None, FWState] = SingleProcessEnvironment(
            task=task,
            model=model,
            optimizer=optimizer,
            grad_computer=grad_computer,
        )

        env.reset(seed=42)

        # Run optimization
        for _ in range(20):
            env.step()
            x = model.parameters_vector()

            # Check simplex constraints
            assert np.all(x >= -1e-10), f"Negative entry: {x}"
            assert np.sum(x) == pytest.approx(1.0, abs=1e-10), f"Sum != 1: {np.sum(x)}"
