"""Logistic Regression task for binary classification.

This module provides:
- Data generation for binary classification
- Per-node data splitting with heterogeneity options
- LogisticRegressionTask implementing the Task protocol
- LogisticGradComputer for gradient computation

Supports both IID and heterogeneous (label-skewed) data distributions
across nodes for studying decentralized optimization.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from core.protocols import Model, Task
from core.types import ParamVector

__all__ = [
    "make_logistic_data",
    "NodeDataset",
    "split_across_nodes",
    "LogisticRegressionTask",
    "LogisticGradComputer",
]


def make_logistic_data(
    *,
    n: int,
    dim: int,
    rng: np.random.Generator,
    separable: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate synthetic binary classification data.

    Creates a dataset where labels are determined by a linear decision boundary
    with optional noise for non-separable cases.

    Args:
        n: Number of samples.
        dim: Feature dimensionality.
        rng: Random number generator.
        separable: If True, data is linearly separable. If False, adds noise.

    Returns:
        Tuple of (X, y) where:
        - X has shape (n, dim), features
        - y has shape (n,), binary labels in {0, 1}
    """
    # Generate features from standard normal
    X = rng.standard_normal((n, dim))

    # Generate true weights
    w_true = rng.standard_normal(dim)
    w_true = w_true / np.linalg.norm(w_true)  # Normalize

    # Compute logits
    logits = X @ w_true

    if separable:
        # Deterministic labels based on sign
        y = (logits > 0).astype(np.float64)
    else:
        # Probabilistic labels with sigmoid
        probs = 1.0 / (1.0 + np.exp(-logits))
        y = (rng.random(n) < probs).astype(np.float64)

    return X, y


@dataclass
class NodeDataset:
    """Dataset for a single node.

    Attributes:
        X: Feature matrix of shape (n_samples, dim).
        y: Label vector of shape (n_samples,) with values in {0, 1}.
        node_id: Identifier for the node.
    """

    X: np.ndarray
    y: np.ndarray
    node_id: int = 0


def split_across_nodes(
    X: np.ndarray,
    y: np.ndarray,
    n_nodes: int,
    heterogeneity: str,
    rng: np.random.Generator,
) -> list[NodeDataset]:
    """Split dataset across nodes with optional heterogeneity.

    Args:
        X: Feature matrix of shape (n, dim).
        y: Label vector of shape (n,).
        n_nodes: Number of nodes to split across.
        heterogeneity: Type of split:
            - "iid": Random uniform split
            - "label_skew": Each node gets biased label proportions
        rng: Random number generator.

    Returns:
        List of NodeDataset, one per node.

    Raises:
        ValueError: If heterogeneity type is unknown.
    """
    n = len(y)

    if heterogeneity == "iid":
        # Random shuffle and split
        indices = rng.permutation(n)
        splits = np.array_split(indices, n_nodes)
        return [NodeDataset(X=X[split], y=y[split], node_id=i) for i, split in enumerate(splits)]

    elif heterogeneity == "label_skew":
        # Sort by label, then distribute with bias
        # Each node gets mostly one class
        idx_0 = np.where(y == 0)[0]
        idx_1 = np.where(y == 1)[0]

        # Shuffle within each class
        rng.shuffle(idx_0)
        rng.shuffle(idx_1)

        # Split each class across nodes with bias
        # Node i gets more of class (i % 2)
        datasets: list[NodeDataset] = []
        n0, n1 = len(idx_0), len(idx_1)

        for i in range(n_nodes):
            # Compute biased proportions
            # Nodes with even i get more class 0, odd i get more class 1
            if i % 2 == 0:
                p0 = 0.8  # 80% class 0, 20% class 1
            else:
                p0 = 0.2  # 20% class 0, 80% class 1

            # Compute number of samples from each class for this node
            samples_per_node = n // n_nodes
            n0_node = int(p0 * samples_per_node)
            n1_node = samples_per_node - n0_node

            # Get indices for this node (with wraparound)
            start_0 = (i * n0 // n_nodes) % max(n0, 1)
            start_1 = (i * n1 // n_nodes) % max(n1, 1)

            node_idx_0 = idx_0[start_0 : start_0 + n0_node] if n0 > 0 else np.array([], dtype=int)
            node_idx_1 = idx_1[start_1 : start_1 + n1_node] if n1 > 0 else np.array([], dtype=int)

            # Handle wraparound if needed
            if len(node_idx_0) < n0_node and n0 > 0:
                remaining = n0_node - len(node_idx_0)
                node_idx_0 = np.concatenate([node_idx_0, idx_0[:remaining]])
            if len(node_idx_1) < n1_node and n1 > 0:
                remaining = n1_node - len(node_idx_1)
                node_idx_1 = np.concatenate([node_idx_1, idx_1[:remaining]])

            node_indices = np.concatenate([node_idx_0, node_idx_1])
            rng.shuffle(node_indices)  # Shuffle combined indices

            datasets.append(NodeDataset(X=X[node_indices], y=y[node_indices], node_id=i))

        return datasets

    else:
        raise ValueError(f"Unknown heterogeneity type: {heterogeneity}")


@dataclass
class LogisticRegressionTask:
    """Binary logistic regression task.

    Implements the Task protocol for binary classification using
    logistic regression (linear model with sigmoid activation).

    Loss: -mean(y * log(p) + (1-y) * log(1-p))
    where p = sigmoid(X @ w)

    Attributes:
        dataset: The node's local dataset.
        batch_size: Number of samples per batch. If None, uses full dataset.
    """

    dataset: NodeDataset
    batch_size: int | None = None
    _eps: float = field(default=1e-15, repr=False)  # For numerical stability

    @property
    def dim(self) -> int:
        """Feature dimensionality."""
        return int(self.dataset.X.shape[1])

    @property
    def n_samples(self) -> int:
        """Number of samples in the dataset."""
        return int(self.dataset.X.shape[0])

    def sample_batch(self, *, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
        """Sample a mini-batch from the dataset.

        Args:
            rng: Random number generator.

        Returns:
            Tuple of (X_batch, y_batch).
        """
        n = self.n_samples
        if self.batch_size is None or self.batch_size >= n:
            return self.dataset.X, self.dataset.y

        indices = rng.choice(n, size=self.batch_size, replace=False)
        return self.dataset.X[indices], self.dataset.y[indices]

    def _sigmoid(self, z: np.ndarray) -> np.ndarray:
        """Numerically stable sigmoid."""
        # Clip to avoid overflow
        z_clipped = np.clip(z, -500, 500)
        result: np.ndarray = 1.0 / (1.0 + np.exp(-z_clipped))
        return result

    def loss(self, model: Model[Any, Any], batch: tuple[np.ndarray, np.ndarray] | None) -> float:
        """Compute binary cross-entropy loss.

        Args:
            model: Model with parameters w.
            batch: Tuple of (X, y) or None for full dataset.

        Returns:
            Average log-loss.
        """
        if batch is None:
            X, y = self.dataset.X, self.dataset.y
        else:
            X, y = batch

        w = model.parameters_vector()
        logits = X @ w
        p = self._sigmoid(logits)

        # Clip probabilities for numerical stability
        p = np.clip(p, self._eps, 1 - self._eps)

        # Binary cross-entropy
        loss = -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))
        return float(loss)

    def metrics(
        self, model: Model[Any, Any], batch: tuple[np.ndarray, np.ndarray] | None
    ) -> Mapping[str, float]:
        """Compute loss and accuracy.

        Args:
            model: Model with parameters w.
            batch: Tuple of (X, y) or None for full dataset.

        Returns:
            Dict with "loss" and "accuracy".
        """
        if batch is None:
            X, y = self.dataset.X, self.dataset.y
        else:
            X, y = batch

        w = model.parameters_vector()
        logits = X @ w
        p = self._sigmoid(logits)

        # Accuracy (threshold at 0.5)
        predictions = (p >= 0.5).astype(np.float64)
        accuracy = float(np.mean(predictions == y))

        return {
            "loss": self.loss(model, batch),
            "accuracy": accuracy,
        }


class LogisticGradComputer:
    """Gradient computer for logistic regression.

    Computes the gradient of the binary cross-entropy loss:
        grad = X^T (p - y) / batch_size

    where p = sigmoid(X @ w).
    """

    def grad(
        self,
        task: Task[Any, Any],
        model: Model[Any, Any],
        batch: tuple[np.ndarray, np.ndarray] | None,
    ) -> ParamVector:
        """Compute gradient of log-loss w.r.t. model parameters.

        Args:
            task: The LogisticRegressionTask.
            model: Model with parameters w.
            batch: Tuple of (X, y) or None for full dataset.

        Returns:
            Gradient vector of same shape as w.
        """
        # Get the actual task (might be wrapped)
        if hasattr(task, "dataset"):
            logistic_task: LogisticRegressionTask = task  # type: ignore[assignment]
        else:
            raise TypeError("LogisticGradComputer requires LogisticRegressionTask")

        if batch is None:
            X, y = logistic_task.dataset.X, logistic_task.dataset.y
        else:
            X, y = batch

        w = model.parameters_vector()
        n = len(y)

        # Compute predictions
        logits = X @ w
        # Numerically stable sigmoid
        logits_clipped = np.clip(logits, -500, 500)
        p = 1.0 / (1.0 + np.exp(-logits_clipped))

        # Gradient: X^T (p - y) / n
        grad = X.T @ (p - y) / n
        result: np.ndarray = grad.astype(np.float64)
        return result
