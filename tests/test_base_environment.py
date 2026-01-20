"""Tests for BaseEnvironment.

This module tests the BaseEnvironment abstract base class using two
dummy implementations:
- DummySingleEnv: Returns StepResult (single-process)
- DummyMultiNodeEnv: Returns Mapping[NodeId, StepResult] (multi-node)
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pytest

from core.types import NodeId, StepRecord, StepResult
from environments.base import BaseEnvironment

# =============================================================================
# Dummy implementations for testing
# =============================================================================


class DummySingleEnv(BaseEnvironment):
    """A dummy single-process environment for testing.

    Returns StepResult with loss and metric equal to the step index.
    """

    def reset(self, *, seed: int) -> None:
        """Reset the environment."""
        self._t = 0
        self._seed = seed

    def step(self) -> StepRecord:
        """Return StepResult with loss=t, metrics={"m": t}."""
        t = self._t
        result = StepResult(loss=float(t), metrics={"m": float(t)})
        self._t += 1
        return result


class DummyMultiNodeEnv(BaseEnvironment):
    """A dummy multi-node environment for testing.

    Returns Mapping[NodeId, StepResult] for 2 nodes:
    - Node 0: loss = t
    - Node 1: loss = t + 2
    """

    def reset(self, *, seed: int) -> None:
        """Reset the environment."""
        self._t = 0
        self._seed = seed

    def step(self) -> StepRecord:
        """Return mapping with 2 nodes having different losses."""
        t = self._t
        results: dict[NodeId, StepResult] = {
            0: StepResult(loss=float(t), metrics={"m": float(t)}),
            1: StepResult(loss=float(t + 2), metrics={"m": float(t + 2)}),
        }
        self._t += 1
        return results


# =============================================================================
# Tests for DummySingleEnv (single-process)
# =============================================================================


class TestDummySingleEnv:
    """Tests for single-process environment behavior."""

    def test_run_returns_correct_length(self) -> None:
        """run(steps=3) returns History of length 3."""
        env = DummySingleEnv()
        env.reset(seed=42)

        history = env.run(steps=3)

        assert len(history) == 3

    def test_run_records_are_step_results(self) -> None:
        """All records in history are StepResult instances."""
        env = DummySingleEnv()
        env.reset(seed=42)

        history = env.run(steps=3)

        for record, _meta in history.steps:
            assert isinstance(record, StepResult)

    def test_mean_loss(self) -> None:
        """mean_loss() equals (0+1+2)/3 = 1.0 for steps=3."""
        env = DummySingleEnv()
        env.reset(seed=42)

        history = env.run(steps=3)

        # Losses are [0.0, 1.0, 2.0], mean = 1.0
        assert history.mean_loss() == 1.0

    def test_mean_metric(self) -> None:
        """mean_metric("m") equals 1.0 for steps=3."""
        env = DummySingleEnv()
        env.reset(seed=42)

        history = env.run(steps=3)

        # Metrics "m" are [0.0, 1.0, 2.0], mean = 1.0
        assert history.mean_metric("m") == 1.0

    def test_run_with_zero_steps_raises(self) -> None:
        """run(steps=0) raises ValueError."""
        env = DummySingleEnv()
        env.reset(seed=42)

        with pytest.raises(ValueError, match="steps must be >= 1"):
            env.run(steps=0)

    def test_run_with_negative_steps_raises(self) -> None:
        """run(steps=-1) raises ValueError."""
        env = DummySingleEnv()
        env.reset(seed=42)

        with pytest.raises(ValueError, match="steps must be >= 1"):
            env.run(steps=-1)

    def test_step_counter_increments(self) -> None:
        """Step counter t increments correctly."""
        env = DummySingleEnv()
        env.reset(seed=42)

        assert env.t == 0
        env.step()
        assert env.t == 1
        env.step()
        assert env.t == 2

    def test_state_dict_contains_t(self) -> None:
        """state_dict() contains the step counter."""
        env = DummySingleEnv()
        env.reset(seed=42)
        env.run(steps=5)

        state = env.state_dict()
        assert state["t"] == 5


# =============================================================================
# Tests for DummyMultiNodeEnv (multi-node)
# =============================================================================


class TestDummyMultiNodeEnv:
    """Tests for multi-node environment behavior."""

    def test_run_returns_correct_length(self) -> None:
        """run(steps=2) returns History of length 2."""
        env = DummyMultiNodeEnv()
        env.reset(seed=42)

        history = env.run(steps=2)

        assert len(history) == 2

    def test_run_records_are_mappings(self) -> None:
        """All records in history are Mapping instances."""
        env = DummyMultiNodeEnv()
        env.reset(seed=42)

        history = env.run(steps=2)

        for record, _meta in history.steps:
            assert isinstance(record, Mapping)
            assert len(record) == 2  # 2 nodes

    def test_mean_loss_aggregation(self) -> None:
        """mean_loss() correctly aggregates across nodes then steps.

        For steps=2:
        - Step 0: node0=0, node1=2 -> step mean = 1.0
        - Step 1: node0=1, node1=3 -> step mean = 2.0
        - Overall mean = (1.0 + 2.0) / 2 = 1.5
        """
        env = DummyMultiNodeEnv()
        env.reset(seed=42)

        history = env.run(steps=2)

        assert history.mean_loss() == 1.5

    def test_mean_metric_aggregation(self) -> None:
        """mean_metric("m") matches the same aggregation as mean_loss.

        Since metrics["m"] == loss for each node, the aggregation is identical.
        """
        env = DummyMultiNodeEnv()
        env.reset(seed=42)

        history = env.run(steps=2)

        assert history.mean_metric("m") == 1.5

    def test_determinism_same_seed(self) -> None:
        """Running twice with same seed produces identical results."""
        env = DummyMultiNodeEnv()

        # First run
        env.reset(seed=123)
        history1 = env.run(steps=3)

        # Second run with same seed
        env.reset(seed=123)
        history2 = env.run(steps=3)

        assert history1.mean_loss() == history2.mean_loss()
        assert history1.mean_metric("m") == history2.mean_metric("m")

    def test_determinism_different_seeds(self) -> None:
        """Different seeds reset state correctly.

        Even though our dummy env doesn't use randomness, this verifies
        that reset() properly resets the step counter.
        """
        env = DummyMultiNodeEnv()

        # First run
        env.reset(seed=1)
        history1 = env.run(steps=2)

        # Second run with different seed
        env.reset(seed=2)
        history2 = env.run(steps=2)

        # Results should be identical since losses depend only on t
        assert history1.mean_loss() == history2.mean_loss()

    def test_individual_step_losses(self) -> None:
        """Verify individual step losses are correct."""
        env = DummyMultiNodeEnv()
        env.reset(seed=42)

        history = env.run(steps=2)

        # Step 0: node0=0, node1=2
        step0, _meta0 = history.steps[0]
        assert isinstance(step0, Mapping)
        assert step0[0].loss == 0.0
        assert step0[1].loss == 2.0

        # Step 1: node0=1, node1=3
        step1, _meta1 = history.steps[1]
        assert isinstance(step1, Mapping)
        assert step1[0].loss == 1.0
        assert step1[1].loss == 3.0


# =============================================================================
# Tests for BaseEnvironment general behavior
# =============================================================================


class TestBaseEnvironmentGeneral:
    """General tests for BaseEnvironment behavior."""

    def test_t_property_is_readonly(self) -> None:
        """The t property should be read-only (no setter)."""
        env = DummySingleEnv()
        env.reset(seed=42)

        # This should work (reading)
        _ = env.t

        # Attempting to set should raise AttributeError
        with pytest.raises(AttributeError):
            env.t = 10  # type: ignore[misc]

    def test_reset_resets_counter(self) -> None:
        """reset() should reset the step counter to 0."""
        env = DummySingleEnv()
        env.reset(seed=42)
        env.run(steps=5)
        assert env.t == 5

        env.reset(seed=42)
        assert env.t == 0

    def test_history_last_returns_final_step(self) -> None:
        """history.last() returns the final step record."""
        env = DummySingleEnv()
        env.reset(seed=42)

        history = env.run(steps=3)

        last = history.last()
        assert isinstance(last, StepResult)
        assert last.loss == 2.0  # t=2 for the third step

    def test_longer_run(self) -> None:
        """Test a longer run to verify consistency."""
        env = DummySingleEnv()
        env.reset(seed=42)

        history = env.run(steps=100)

        assert len(history) == 100
        # Mean of 0..99 = 49.5
        assert np.isclose(history.mean_loss(), 49.5)
