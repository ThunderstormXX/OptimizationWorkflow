"""Tasks module for the benchmark framework.

This package contains task implementations that define optimization problems,
including data sampling, loss computation, and metrics.

Available tasks:
- SyntheticQuadraticTask: Quadratic optimization (f(x) = 0.5 x^T A x - b^T x)
- LogisticRegressionTask: Binary classification with logistic regression
- MNISTClassificationTask: MNIST digit classification (requires torch)
"""

from __future__ import annotations

from tasks.logistic_regression import (
    LogisticGradComputer,
    LogisticRegressionTask,
    NodeDataset,
    make_logistic_data,
    split_across_nodes,
)
from tasks.mnist import (
    TORCH_AVAILABLE,
    MNISTClassificationTask,
    MNISTSupervisedTask,
    MNISTTinyDataset,
    TorchGradComputer,
    load_mnist,
    load_mnist_or_fake,
    load_mnist_tiny,
    split_mnist_across_nodes,
)
from tasks.synthetic_quadratic import (
    QuadraticGradComputer,
    QuadraticProblem,
    SyntheticQuadraticTask,
    make_spd_quadratic,
)

__all__ = [
    # Quadratic task
    "QuadraticProblem",
    "SyntheticQuadraticTask",
    "QuadraticGradComputer",
    "make_spd_quadratic",
    # Logistic regression task
    "LogisticRegressionTask",
    "LogisticGradComputer",
    "NodeDataset",
    "make_logistic_data",
    "split_across_nodes",
    # MNIST task
    "MNISTClassificationTask",
    "MNISTSupervisedTask",
    "MNISTTinyDataset",
    "TorchGradComputer",
    "load_mnist",
    "load_mnist_or_fake",
    "load_mnist_tiny",
    "split_mnist_across_nodes",
    "TORCH_AVAILABLE",
]
