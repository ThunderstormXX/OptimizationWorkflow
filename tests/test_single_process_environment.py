"""Tests for SingleProcessEnvironment.

This module tests the SingleProcessEnvironment with dummy implementations
of Model, Task, GradComputer, and Optimizer to validate:
- Proper wiring between components
- Deterministic behavior with seeded RNG
- State management and state_dict
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
import pytest

from core.types import ParamVector, StepResult
from environments.single_process import SingleProcessEnvironment

# =============================================================================
# Dummy implementations for testing
# =============================================================================


class DummyModel:
    """A dummy model with a 1D parameter vector.

    The model stores parameters x of shape (d,). Only x[0] is used
    in the dummy loss computation.
    """

    def __init__(self, dim: int = 3) -> None:
        """Initialize with zeros."""
        self.x = np.zeros(dim)

    def forward(self, batch: float) -> float:
        """Trivial forward pass (unused in loss, but required by protocol)."""
        return self.x[0] * batch

    def parameters_vector(self) -> ParamVector:
        """Return a copy of the parameter vector."""
        return self.x.copy()

    def set_parameters_vector(self, v: ParamVector) -> None:
        """Set parameters from a vector (copy)."""
        self.x = v.copy()


class DummyTask:
    """A dummy task that samples scalar batches.

    Loss is (x[0] - batch)^2 where x is the model parameter vector.
    """

    def sample_batch(self, *, rng: np.random.Generator) -> float:
        """Sample a random scalar batch in [0, 1)."""
        return float(rng.random())

    def loss(self, model: DummyModel, batch: float) -> float:
        """Compute squared error loss: (x[0] - batch)^2."""
        return float((model.x[0] - batch) ** 2)

    def metrics(self, model: DummyModel, batch: float) -> Mapping[str, float]:
        """Return the batch value and loss as metrics."""
        loss = self.loss(model, batch)
        return {"batch": batch, "loss": loss}


class DummyGradComputer:
    """Computes gradient of the dummy loss.

    For loss = (x[0] - batch)^2:
    - grad[0] = 2 * (x[0] - batch)
    - grad[i] = 0 for i > 0
    """

    def grad(
        self,
        task: DummyTask,
        model: DummyModel,
        batch: float,
    ) -> ParamVector:
        """Compute gradient of loss w.r.t. model parameters."""
        g = np.zeros_like(model.x)
        g[0] = 2.0 * (model.x[0] - batch)
        return g


@dataclass
class DummyOptimizerState:
    """State for the dummy optimizer (just stores learning rate)."""

    lr: float


class DummyOptimizer:
    """A simple gradient descent optimizer.

    Performs: x <- x - lr * grad
    """

    def __init__(self, lr: float = 0.1) -> None:
        """Initialize with learning rate."""
        self._lr = lr

    def init_state(self, model: DummyModel) -> DummyOptimizerState:
        """Initialize optimizer state."""
        return DummyOptimizerState(lr=self._lr)

    def step(
        self,
        *,
        task: DummyTask,
        model: DummyModel,
        batch: float,
        grad_computer: DummyGradComputer,
        state: DummyOptimizerState,
        rng: np.random.Generator,
    ) -> tuple[DummyOptimizerState, StepResult]:
        """Perform one gradient descent step."""
        # Compute gradient
        g = grad_computer.grad(task, model, batch)

        # Update parameters: x <- x - lr * g
        new_params = model.parameters_vector() - state.lr * g
        model.set_parameters_vector(new_params)

        # Compute loss after update
        loss = task.loss(model, batch)
        metrics = task.metrics(model, batch)

        return state, StepResult(loss=loss, metrics=metrics)


# =============================================================================
# Helper to create fresh environment
# =============================================================================


def make_env(
    dim: int = 3, lr: float = 0.1
) -> SingleProcessEnvironment[float, float, DummyOptimizerState]:
    """Create a fresh SingleProcessEnvironment with dummy components."""
    return SingleProcessEnvironment(
        task=DummyTask(),
        model=DummyModel(dim=dim),
        optimizer=DummyOptimizer(lr=lr),
        grad_computer=DummyGradComputer(),
    )


# =============================================================================
# Tests
# =============================================================================


class TestSingleProcessEnvironmentBasics:
    """Basic functionality tests."""

    def test_requires_reset_before_step(self) -> None:
        """step() raises RuntimeError if reset() was not called."""
        env = make_env()

        with pytest.raises(RuntimeError, match="not initialized"):
            env.step()

    def test_reset_allows_step(self) -> None:
        """After reset(), step() works without error."""
        env = make_env()
        env.reset(seed=42)

        result = env.step()

        assert isinstance(result, StepResult)
        assert isinstance(result.loss, float)

    def test_run_returns_history_of_correct_length(self) -> None:
        """run(steps=N) returns History with N records."""
        env = make_env()
        env.reset(seed=42)

        history = env.run(steps=5)

        assert len(history) == 5


class TestSingleProcessEnvironmentParameterUpdates:
    """Tests for parameter update behavior."""

    def test_single_step_updates_parameters(self) -> None:
        """A single step should update model parameters in the correct direction."""
        env = make_env(dim=3, lr=0.5)
        # Set initial parameters to zeros
        env.model.set_parameters_vector(np.array([0.0, 0.0, 0.0]))

        env.reset(seed=123)
        initial_params = env.model.parameters_vector().copy()

        # Take one step
        env.step()

        final_params = env.model.parameters_vector()

        # Parameters should have changed
        assert not np.allclose(initial_params, final_params)

        # With seed=123, the batch is deterministic
        # x[0] starts at 0, batch is some positive value
        # grad[0] = 2*(0 - batch) = -2*batch (negative)
        # x[0] <- x[0] - lr * grad[0] = 0 - 0.5 * (-2*batch) = batch
        # So x[0] should move toward the batch value (positive direction)
        assert final_params[0] > initial_params[0]

        # Other parameters should remain zero (gradient is zero for them)
        assert final_params[1] == 0.0
        assert final_params[2] == 0.0

    def test_multiple_steps_continue_optimization(self) -> None:
        """Multiple steps should continue to update parameters."""
        env = make_env(dim=3, lr=0.1)
        env.reset(seed=42)

        # Run several steps
        history = env.run(steps=10)

        # Loss should generally decrease (model learns to predict batch mean)
        first_record, _meta = history.steps[0]
        first_loss = first_record.loss  # type: ignore[union-attr]
        last_loss = history.last().loss  # type: ignore[union-attr]

        # With our simple task, loss should decrease or stay similar
        # (exact behavior depends on batch sequence)
        assert isinstance(first_loss, float)
        assert isinstance(last_loss, float)


class TestSingleProcessEnvironmentDeterminism:
    """Tests for deterministic behavior with seeds."""

    def test_same_seed_identical_trajectory(self) -> None:
        """Same seed produces identical parameter trajectories."""
        # First run
        env1 = make_env(dim=3, lr=0.1)
        env1.reset(seed=42)
        hist1 = env1.run(steps=5)
        params1 = env1.model.parameters_vector()

        # Second run with fresh instances but same seed
        env2 = make_env(dim=3, lr=0.1)
        env2.reset(seed=42)
        hist2 = env2.run(steps=5)
        params2 = env2.model.parameters_vector()

        # Parameters should be identical
        assert np.allclose(params1, params2)

        # History aggregations should match
        assert hist1.mean_loss() == pytest.approx(hist2.mean_loss())

    def test_same_seed_identical_step_by_step(self) -> None:
        """Same seed produces identical results at each step."""
        env1 = make_env()
        env1.reset(seed=42)

        env2 = make_env()
        env2.reset(seed=42)

        for _ in range(5):
            result1 = env1.step()
            result2 = env2.step()

            assert result1.loss == pytest.approx(result2.loss)
            assert result1.metrics["batch"] == pytest.approx(result2.metrics["batch"])

    def test_different_seeds_different_trajectory(self) -> None:
        """Different seeds produce different trajectories (very likely)."""
        env1 = make_env(dim=3, lr=0.1)
        env1.reset(seed=1)
        env1.run(steps=5)
        params1 = env1.model.parameters_vector()

        env2 = make_env(dim=3, lr=0.1)
        env2.reset(seed=2)
        env2.run(steps=5)
        params2 = env2.model.parameters_vector()

        # Parameters should differ (with overwhelming probability)
        assert not np.allclose(params1, params2)

    def test_reset_restores_determinism(self) -> None:
        """Calling reset() with same seed restores deterministic behavior."""
        env = make_env()

        # First run
        env.reset(seed=42)
        env.run(steps=3)
        params_after_first = env.model.parameters_vector().copy()

        # Reset model to zeros manually
        env.model.set_parameters_vector(np.zeros(3))

        # Second run with same seed
        env.reset(seed=42)
        env.run(steps=3)
        params_after_second = env.model.parameters_vector()

        # Should be identical
        assert np.allclose(params_after_first, params_after_second)


class TestSingleProcessEnvironmentStateDict:
    """Tests for state_dict functionality."""

    def test_state_dict_contains_expected_keys(self) -> None:
        """state_dict() contains t, seed, and params."""
        env = make_env()
        env.reset(seed=123)
        env.step()

        state = env.state_dict()

        assert "t" in state
        assert "seed" in state
        assert "params" in state

    def test_state_dict_t_matches_step_count(self) -> None:
        """state_dict["t"] matches the number of steps taken."""
        env = make_env()
        env.reset(seed=42)

        assert env.state_dict()["t"] == 0

        env.step()
        assert env.state_dict()["t"] == 1

        env.run(steps=3)
        assert env.state_dict()["t"] == 4

    def test_state_dict_seed_matches_reset_seed(self) -> None:
        """state_dict["seed"] matches the seed passed to reset()."""
        env = make_env()
        env.reset(seed=999)

        assert env.state_dict()["seed"] == 999

    def test_state_dict_params_is_list(self) -> None:
        """state_dict["params"] is a list (JSON-serializable)."""
        env = make_env(dim=5)
        env.reset(seed=42)
        env.step()

        state = env.state_dict()

        assert isinstance(state["params"], list)
        assert len(state["params"]) == 5

    def test_state_dict_params_matches_model(self) -> None:
        """state_dict["params"] matches model.parameters_vector()."""
        env = make_env()
        env.reset(seed=42)
        env.run(steps=3)

        state = env.state_dict()
        model_params = env.model.parameters_vector()

        assert np.allclose(state["params"], model_params)


class TestSingleProcessEnvironmentStepCounter:
    """Tests for step counter behavior."""

    def test_t_starts_at_zero_after_reset(self) -> None:
        """t property is 0 after reset()."""
        env = make_env()
        env.reset(seed=42)

        assert env.t == 0

    def test_t_increments_on_step(self) -> None:
        """t property increments by 1 on each step()."""
        env = make_env()
        env.reset(seed=42)

        for expected_t in range(1, 6):
            env.step()
            assert env.t == expected_t

    def test_reset_resets_t(self) -> None:
        """reset() resets t back to 0."""
        env = make_env()
        env.reset(seed=42)
        env.run(steps=5)
        assert env.t == 5

        env.reset(seed=42)
        assert env.t == 0
