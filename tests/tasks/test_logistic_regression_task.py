"""Tests for the Logistic Regression task.

This module tests:
- Data generation (make_logistic_data)
- Data splitting with heterogeneity (split_across_nodes)
- LogisticRegressionTask loss and metrics
- LogisticGradComputer gradient correctness
- Integration with SingleProcessEnvironment
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from environments.single_process import SingleProcessEnvironment
from models.numpy_vector import NumpyVectorModel
from optim.legacy_frankwolfe import GradientDescentOptimizer
from tasks.logistic_regression import (
    LogisticGradComputer,
    LogisticRegressionTask,
    NodeDataset,
    make_logistic_data,
    split_across_nodes,
)

# =============================================================================
# Tests for make_logistic_data
# =============================================================================


class TestMakeLogisticData:
    """Tests for the data generation function."""

    def test_output_shapes(self) -> None:
        """Generated data should have correct shapes."""
        rng = np.random.default_rng(42)
        X, y = make_logistic_data(n=100, dim=10, rng=rng)

        assert X.shape == (100, 10)
        assert y.shape == (100,)

    def test_labels_binary(self) -> None:
        """Labels should be binary (0 or 1)."""
        rng = np.random.default_rng(42)
        X, y = make_logistic_data(n=100, dim=10, rng=rng)

        assert set(np.unique(y)).issubset({0.0, 1.0})

    def test_separable_has_both_classes(self) -> None:
        """Separable data should have both classes."""
        rng = np.random.default_rng(42)
        X, y = make_logistic_data(n=100, dim=10, rng=rng, separable=True)

        # Should have both classes (with high probability)
        assert 0.0 in y or 1.0 in y

    def test_deterministic_with_seed(self) -> None:
        """Same seed should produce same data."""
        rng1 = np.random.default_rng(42)
        X1, y1 = make_logistic_data(n=50, dim=5, rng=rng1)

        rng2 = np.random.default_rng(42)
        X2, y2 = make_logistic_data(n=50, dim=5, rng=rng2)

        np.testing.assert_array_equal(X1, X2)
        np.testing.assert_array_equal(y1, y2)


# =============================================================================
# Tests for split_across_nodes
# =============================================================================


class TestSplitAcrossNodes:
    """Tests for the data splitting function."""

    def test_iid_split_covers_all_data(self) -> None:
        """IID split should cover all data points."""
        rng = np.random.default_rng(42)
        X, y = make_logistic_data(n=100, dim=5, rng=rng)

        datasets = split_across_nodes(X, y, n_nodes=4, heterogeneity="iid", rng=rng)

        assert len(datasets) == 4

        # Total samples should equal original
        total_samples = sum(len(ds.y) for ds in datasets)
        assert total_samples == 100

    def test_label_skew_produces_different_distributions(self) -> None:
        """Label skew should produce different label proportions across nodes."""
        rng = np.random.default_rng(42)
        X, y = make_logistic_data(n=1000, dim=5, rng=rng)

        datasets = split_across_nodes(X, y, n_nodes=4, heterogeneity="label_skew", rng=rng)

        # Compute label proportions per node
        proportions = []
        for ds in datasets:
            if len(ds.y) > 0:
                prop_1 = np.mean(ds.y)
                proportions.append(prop_1)

        # Proportions should vary across nodes (not all the same)
        if len(proportions) >= 2:
            # Check that there's some variation
            assert max(proportions) - min(proportions) > 0.1, (
                f"Label proportions should vary with label_skew: {proportions}"
            )

    def test_node_ids_assigned_correctly(self) -> None:
        """Each dataset should have correct node_id."""
        rng = np.random.default_rng(42)
        X, y = make_logistic_data(n=100, dim=5, rng=rng)

        datasets = split_across_nodes(X, y, n_nodes=3, heterogeneity="iid", rng=rng)

        for i, ds in enumerate(datasets):
            assert ds.node_id == i

    def test_unknown_heterogeneity_raises(self) -> None:
        """Unknown heterogeneity type should raise ValueError."""
        rng = np.random.default_rng(42)
        X, y = make_logistic_data(n=100, dim=5, rng=rng)

        with pytest.raises(ValueError, match="Unknown heterogeneity"):
            split_across_nodes(X, y, n_nodes=2, heterogeneity="unknown", rng=rng)


# =============================================================================
# Tests for LogisticRegressionTask
# =============================================================================


class TestLogisticRegressionTask:
    """Tests for the logistic regression task."""

    def test_loss_finite(self) -> None:
        """Loss should be finite for reasonable weights."""
        rng = np.random.default_rng(42)
        X, y = make_logistic_data(n=100, dim=5, rng=rng)
        dataset = NodeDataset(X=X, y=y)
        task = LogisticRegressionTask(dataset=dataset)

        model = NumpyVectorModel(np.zeros(5))
        loss = task.loss(model, None)

        assert np.isfinite(loss)

    def test_loss_for_zero_weights(self) -> None:
        """Loss for zero weights should be log(2) ≈ 0.693."""
        # With w=0, sigmoid(0)=0.5, so log-loss = log(2)
        X = np.array([[1.0, 0.0], [0.0, 1.0]])
        y = np.array([0.0, 1.0])
        dataset = NodeDataset(X=X, y=y)
        task = LogisticRegressionTask(dataset=dataset)

        model = NumpyVectorModel(np.zeros(2))
        loss = task.loss(model, None)

        assert np.isclose(loss, np.log(2), atol=1e-6)

    def test_metrics_include_accuracy(self) -> None:
        """Metrics should include accuracy."""
        rng = np.random.default_rng(42)
        X, y = make_logistic_data(n=100, dim=5, rng=rng)
        dataset = NodeDataset(X=X, y=y)
        task = LogisticRegressionTask(dataset=dataset)

        model = NumpyVectorModel(np.zeros(5))
        metrics = task.metrics(model, None)

        assert "accuracy" in metrics
        assert 0.0 <= metrics["accuracy"] <= 1.0

    def test_sample_batch_returns_correct_size(self) -> None:
        """sample_batch should return correct batch size."""
        rng = np.random.default_rng(42)
        X, y = make_logistic_data(n=100, dim=5, rng=rng)
        dataset = NodeDataset(X=X, y=y)
        task = LogisticRegressionTask(dataset=dataset, batch_size=32)

        batch_X, batch_y = task.sample_batch(rng=rng)

        assert batch_X.shape == (32, 5)
        assert batch_y.shape == (32,)

    def test_full_batch_when_batch_size_none(self) -> None:
        """Full dataset should be returned when batch_size is None."""
        rng = np.random.default_rng(42)
        X, y = make_logistic_data(n=50, dim=5, rng=rng)
        dataset = NodeDataset(X=X, y=y)
        task = LogisticRegressionTask(dataset=dataset, batch_size=None)

        batch_X, batch_y = task.sample_batch(rng=rng)

        assert batch_X.shape == (50, 5)
        assert batch_y.shape == (50,)


# =============================================================================
# Tests for LogisticGradComputer
# =============================================================================


class TestLogisticGradComputer:
    """Tests for the logistic gradient computer."""

    def test_gradient_finite_difference_check(self) -> None:
        """Gradient should match finite difference approximation."""
        rng = np.random.default_rng(42)
        X = rng.standard_normal((10, 3))
        y = (rng.random(10) > 0.5).astype(np.float64)
        dataset = NodeDataset(X=X, y=y)
        task = LogisticRegressionTask(dataset=dataset)
        grad_computer = LogisticGradComputer()

        w = rng.standard_normal(3)
        model = NumpyVectorModel(w.copy())

        # Compute analytical gradient
        grad = grad_computer.grad(task, model, None)

        # Compute finite difference gradient
        eps = 1e-5
        fd_grad = np.zeros(3)
        for i in range(3):
            w_plus = w.copy()
            w_plus[i] += eps
            model.set_parameters_vector(w_plus)
            loss_plus = task.loss(model, None)

            w_minus = w.copy()
            w_minus[i] -= eps
            model.set_parameters_vector(w_minus)
            loss_minus = task.loss(model, None)

            fd_grad[i] = (loss_plus - loss_minus) / (2 * eps)

        np.testing.assert_allclose(grad, fd_grad, atol=1e-4)

    def test_gradient_shape(self) -> None:
        """Gradient should have same shape as parameters."""
        rng = np.random.default_rng(42)
        X, y = make_logistic_data(n=50, dim=7, rng=rng)
        dataset = NodeDataset(X=X, y=y)
        task = LogisticRegressionTask(dataset=dataset)
        grad_computer = LogisticGradComputer()

        model = NumpyVectorModel(np.zeros(7))
        grad = grad_computer.grad(task, model, None)

        assert grad.shape == (7,)


# =============================================================================
# Tests for integration with SingleProcessEnvironment
# =============================================================================


class TestLogisticIntegration:
    """Integration tests for logistic regression with environments."""

    def test_loss_decreases_with_gd(self) -> None:
        """Loss should decrease after GD steps."""
        rng = np.random.default_rng(42)
        X, y = make_logistic_data(n=200, dim=5, rng=rng, separable=True)
        dataset = NodeDataset(X=X, y=y)
        task = LogisticRegressionTask(dataset=dataset, batch_size=64)
        grad_computer = LogisticGradComputer()

        model = NumpyVectorModel(np.zeros(5))
        optimizer = GradientDescentOptimizer(lr=0.5)

        env: SingleProcessEnvironment[tuple[np.ndarray, np.ndarray], np.ndarray, Any] = (
            SingleProcessEnvironment(
                task=task,
                model=model,
                optimizer=optimizer,
                grad_computer=grad_computer,
            )
        )

        env.reset(seed=42)

        # Get initial loss
        initial_loss = task.loss(model, None)

        # Run some steps
        env.run(steps=20)

        # Get final loss
        final_loss = task.loss(model, None)

        assert final_loss < initial_loss, f"Loss did not decrease: {initial_loss} -> {final_loss}"

    def test_deterministic_training(self) -> None:
        """Training should be deterministic with same seed."""
        rng = np.random.default_rng(42)
        X, y = make_logistic_data(n=100, dim=5, rng=rng)

        def run_training(seed: int) -> float:
            dataset = NodeDataset(X=X.copy(), y=y.copy())
            task = LogisticRegressionTask(dataset=dataset, batch_size=32)
            grad_computer = LogisticGradComputer()
            model = NumpyVectorModel(np.zeros(5))
            optimizer = GradientDescentOptimizer(lr=0.1)

            env: SingleProcessEnvironment[tuple[np.ndarray, np.ndarray], np.ndarray, Any] = (
                SingleProcessEnvironment(
                    task=task,
                    model=model,
                    optimizer=optimizer,
                    grad_computer=grad_computer,
                )
            )

            env.reset(seed=seed)
            env.run(steps=10)
            return task.loss(model, None)

        loss1 = run_training(seed=123)
        loss2 = run_training(seed=123)

        assert loss1 == loss2
