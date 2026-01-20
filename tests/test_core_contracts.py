"""Tests for core protocols and types.

This module contains:
- Dummy implementations of all protocols to verify they work
- Tests for History aggregation (single-process and multi-node)
- Smoke tests for Environment.run()
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from core.protocols import (
    ConstraintSet,
    Environment,
    GradComputer,
    Model,
    Optimizer,
    Task,
)
from core.types import (
    History,
    NodeId,
    ParamVector,
    StepRecord,
    StepResult,
)

# =============================================================================
# Dummy implementations for protocol verification
# =============================================================================


class DummyModel:
    """A dummy model that stores a single parameter vector."""

    def __init__(self, dim: int = 10) -> None:
        self._params = np.zeros(dim)

    def forward(self, batch: np.ndarray) -> np.ndarray:
        return batch @ self._params

    def parameters_vector(self) -> ParamVector:
        return self._params.copy()

    def set_parameters_vector(self, v: ParamVector) -> None:
        self._params = v.copy()


class DummyTask:
    """A dummy task that returns constant loss and metrics."""

    def __init__(self, dim: int = 10, batch_size: int = 4) -> None:
        self._dim = dim
        self._batch_size = batch_size

    def sample_batch(self, *, rng: np.random.Generator) -> np.ndarray:
        return rng.standard_normal((self._batch_size, self._dim))

    def loss(self, model: Model[np.ndarray, np.ndarray], batch: np.ndarray) -> float:
        _ = model.forward(batch)
        return 1.0  # Constant loss for testing

    def metrics(
        self, model: Model[np.ndarray, np.ndarray], batch: np.ndarray
    ) -> Mapping[str, float]:
        _ = model.forward(batch)
        return {"accuracy": 0.5}


class DummyGradComputer:
    """A dummy gradient computer that returns zero gradients."""

    def grad(
        self,
        task: Task[np.ndarray, np.ndarray],
        model: Model[np.ndarray, np.ndarray],
        batch: np.ndarray,
    ) -> ParamVector:
        return np.zeros_like(model.parameters_vector())


class DummyConstraintSet:
    """A dummy constraint set (unconstrained)."""

    def lmo(self, grad: ParamVector) -> ParamVector:
        # For unconstrained, LMO returns -infinity in gradient direction
        # But for testing, just return zeros
        return np.zeros_like(grad)

    def project(self, x: ParamVector) -> ParamVector:
        # Unconstrained: projection is identity
        return x.copy()


class DummyOptimizerState:
    """State for the dummy optimizer."""

    def __init__(self, step_count: int = 0) -> None:
        self.step_count = step_count


class DummyOptimizer:
    """A dummy optimizer that doesn't actually update parameters."""

    def init_state(self, model: Model[np.ndarray, np.ndarray]) -> DummyOptimizerState:
        return DummyOptimizerState()

    def step(
        self,
        *,
        task: Task[np.ndarray, np.ndarray],
        model: Model[np.ndarray, np.ndarray],
        batch: np.ndarray,
        grad_computer: GradComputer[np.ndarray, np.ndarray],
        state: DummyOptimizerState,
        rng: np.random.Generator,
    ) -> tuple[DummyOptimizerState, StepResult]:
        # Compute gradient (not used, but verifies interface)
        _ = grad_computer.grad(task, model, batch)

        # Compute loss and metrics
        loss = task.loss(model, batch)
        metrics = task.metrics(model, batch)

        new_state = DummyOptimizerState(state.step_count + 1)
        return new_state, StepResult(loss=loss, metrics=metrics)


class DummyEnvironment:
    """A dummy single-process environment for testing."""

    def __init__(self, dim: int = 10) -> None:
        self._dim = dim
        self._model = DummyModel(dim)
        self._task = DummyTask(dim)
        self._optimizer = DummyOptimizer()
        self._grad_computer = DummyGradComputer()
        self._state: DummyOptimizerState | None = None
        self._rng: np.random.Generator | None = None
        self._step_count = 0

    def reset(self, *, seed: int) -> None:
        self._rng = np.random.default_rng(seed)
        self._model = DummyModel(self._dim)
        self._state = self._optimizer.init_state(self._model)
        self._step_count = 0

    def step(self) -> StepRecord:
        if self._rng is None or self._state is None:
            raise RuntimeError("Environment not reset. Call reset() first.")

        batch = self._task.sample_batch(rng=self._rng)
        self._state, result = self._optimizer.step(
            task=self._task,
            model=self._model,
            batch=batch,
            grad_computer=self._grad_computer,
            state=self._state,
            rng=self._rng,
        )
        self._step_count += 1
        return result

    def run(self, *, steps: int) -> History:
        history = History()
        for _ in range(steps):
            record = self.step()
            history.append(record)
        return history

    def state_dict(self) -> dict[str, Any]:
        return {
            "step_count": self._step_count,
            "params": self._model.parameters_vector().tolist(),
        }


class DummyMultiNodeEnvironment:
    """A dummy multi-node environment for testing.

    Returns Mapping[NodeId, StepResult] from step().
    """

    def __init__(self, num_nodes: int = 3, dim: int = 10) -> None:
        self._num_nodes = num_nodes
        self._dim = dim
        self._rng: np.random.Generator | None = None
        self._step_count = 0

    def reset(self, *, seed: int) -> None:
        self._rng = np.random.default_rng(seed)
        self._step_count = 0

    def step(self) -> StepRecord:
        if self._rng is None:
            raise RuntimeError("Environment not reset. Call reset() first.")

        # Return different losses/metrics per node
        results: dict[NodeId, StepResult] = {}
        for node_id in range(self._num_nodes):
            loss = float(node_id + 1)  # Node 0: 1.0, Node 1: 2.0, etc.
            results[node_id] = StepResult(
                loss=loss,
                metrics={"accuracy": 0.1 * (node_id + 1)},
            )
        self._step_count += 1
        return results

    def run(self, *, steps: int) -> History:
        history = History()
        for _ in range(steps):
            record = self.step()
            history.append(record)
        return history

    def state_dict(self) -> dict[str, Any]:
        return {"step_count": self._step_count, "num_nodes": self._num_nodes}


# =============================================================================
# Tests
# =============================================================================


class TestProtocolCompliance:
    """Verify dummy implementations satisfy protocols."""

    def test_model_protocol(self) -> None:
        model = DummyModel(dim=5)
        assert isinstance(model, Model)

        # Test interface
        params = model.parameters_vector()
        assert params.shape == (5,)

        model.set_parameters_vector(np.ones(5))
        assert np.allclose(model.parameters_vector(), np.ones(5))

    def test_task_protocol(self) -> None:
        task = DummyTask(dim=5)
        assert isinstance(task, Task)

        rng = np.random.default_rng(42)
        batch = task.sample_batch(rng=rng)
        assert batch.shape == (4, 5)

    def test_grad_computer_protocol(self) -> None:
        grad_computer = DummyGradComputer()
        assert isinstance(grad_computer, GradComputer)

    def test_constraint_set_protocol(self) -> None:
        constraint = DummyConstraintSet()
        assert isinstance(constraint, ConstraintSet)

        grad = np.array([1.0, 2.0, 3.0])
        lmo_result = constraint.lmo(grad)
        assert lmo_result.shape == grad.shape

    def test_optimizer_protocol(self) -> None:
        optimizer = DummyOptimizer()
        assert isinstance(optimizer, Optimizer)

    def test_environment_protocol(self) -> None:
        env = DummyEnvironment()
        assert isinstance(env, Environment)


class TestHistory:
    """Tests for History aggregation."""

    def test_append_and_len(self) -> None:
        history = History()
        assert len(history) == 0

        history.append(StepResult(loss=1.0, metrics={}))
        assert len(history) == 1

        history.append(StepResult(loss=2.0, metrics={}))
        assert len(history) == 2

    def test_last(self) -> None:
        history = History()
        history.append(StepResult(loss=1.0, metrics={"a": 1.0}))
        history.append(StepResult(loss=2.0, metrics={"a": 2.0}))

        last = history.last()
        assert isinstance(last, StepResult)
        assert last.loss == 2.0

    def test_mean_loss_single_process(self) -> None:
        """Test mean_loss for single-process records (StepResult)."""
        history = History()
        history.append(StepResult(loss=1.0, metrics={}))
        history.append(StepResult(loss=2.0, metrics={}))
        history.append(StepResult(loss=3.0, metrics={}))

        # Mean of [1.0, 2.0, 3.0] = 2.0
        assert history.mean_loss() == 2.0

    def test_mean_metric_single_process(self) -> None:
        """Test mean_metric for single-process records."""
        history = History()
        history.append(StepResult(loss=1.0, metrics={"acc": 0.8}))
        history.append(StepResult(loss=2.0, metrics={"acc": 0.9}))

        # Mean of [0.8, 0.9] = 0.85
        assert np.isclose(history.mean_metric("acc"), 0.85)

    def test_mean_loss_multi_node(self) -> None:
        """Test mean_loss for multi-node records (Mapping[NodeId, StepResult]).

        Aggregation: average across nodes first, then average across steps.
        """
        history = History()

        # Step 1: Node 0 -> loss=1.0, Node 1 -> loss=3.0
        # Node average = (1.0 + 3.0) / 2 = 2.0
        step1: Mapping[NodeId, StepResult] = {
            0: StepResult(loss=1.0, metrics={}),
            1: StepResult(loss=3.0, metrics={}),
        }
        history.append(step1)

        # Step 2: Node 0 -> loss=2.0, Node 1 -> loss=4.0
        # Node average = (2.0 + 4.0) / 2 = 3.0
        step2: Mapping[NodeId, StepResult] = {
            0: StepResult(loss=2.0, metrics={}),
            1: StepResult(loss=4.0, metrics={}),
        }
        history.append(step2)

        # Overall mean = (2.0 + 3.0) / 2 = 2.5
        assert history.mean_loss() == 2.5

    def test_mean_metric_multi_node(self) -> None:
        """Test mean_metric for multi-node records."""
        history = History()

        # Step 1: Node 0 -> acc=0.6, Node 1 -> acc=0.8
        # Node average = (0.6 + 0.8) / 2 = 0.7
        step1: Mapping[NodeId, StepResult] = {
            0: StepResult(loss=1.0, metrics={"acc": 0.6}),
            1: StepResult(loss=1.0, metrics={"acc": 0.8}),
        }
        history.append(step1)

        # Step 2: Node 0 -> acc=0.7, Node 1 -> acc=0.9
        # Node average = (0.7 + 0.9) / 2 = 0.8
        step2: Mapping[NodeId, StepResult] = {
            0: StepResult(loss=1.0, metrics={"acc": 0.7}),
            1: StepResult(loss=1.0, metrics={"acc": 0.9}),
        }
        history.append(step2)

        # Overall mean = (0.7 + 0.8) / 2 = 0.75
        assert history.mean_metric("acc") == 0.75


class TestEnvironmentRun:
    """Smoke tests for Environment.run()."""

    def test_single_process_environment_run(self) -> None:
        """Test that Environment.run() returns History of correct length."""
        env = DummyEnvironment()
        env.reset(seed=42)

        history = env.run(steps=3)

        assert len(history) == 3
        # All steps should be StepResult (single-process)
        for record, _meta in history.steps:
            assert isinstance(record, StepResult)

    def test_multi_node_environment_run(self) -> None:
        """Test multi-node Environment.run()."""
        env = DummyMultiNodeEnvironment(num_nodes=3)
        env.reset(seed=42)

        history = env.run(steps=3)

        assert len(history) == 3
        # All steps should be Mapping[NodeId, StepResult]
        for record, _meta in history.steps:
            assert isinstance(record, Mapping)
            assert len(record) == 3  # 3 nodes

        # Check aggregation
        # Each step: nodes have losses [1.0, 2.0, 3.0], mean = 2.0
        # All steps same, so overall mean = 2.0
        assert np.isclose(history.mean_loss(), 2.0)

        # Each step: nodes have acc [0.1, 0.2, 0.3], mean = 0.2
        assert np.isclose(history.mean_metric("accuracy"), 0.2)

    def test_environment_state_dict(self) -> None:
        """Test that state_dict returns expected structure."""
        env = DummyEnvironment()
        env.reset(seed=42)
        env.run(steps=2)

        state = env.state_dict()
        assert "step_count" in state
        assert state["step_count"] == 2
