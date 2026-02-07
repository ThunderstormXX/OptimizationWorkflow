from __future__ import annotations

import numpy as np
import pytest

from models.numpy_vector import NumpyVectorModel
from tasks.logistic_regression import (
    LogisticGradComputer,
    LogisticRegressionTask,
    NodeDataset,
    split_across_nodes,
)


def test_split_across_nodes_label_skew_wraparound() -> None:
    rng = np.random.default_rng(0)
    X = rng.standard_normal((10, 3))
    # Imbalanced labels to trigger wraparound for class 0
    y = np.array([0] + [1] * 9)

    datasets = split_across_nodes(
        X,
        y,
        n_nodes=3,
        heterogeneity="label_skew",
        rng=np.random.default_rng(0),
    )
    assert len(datasets) == 3
    assert sum(ds.X.shape[0] for ds in datasets) == 9  # drop remainder by design


def test_logistic_task_dim_property() -> None:
    X = np.zeros((5, 4))
    y = np.zeros(5)
    dataset = NodeDataset(X=X, y=y, node_id=0)
    task = LogisticRegressionTask(dataset=dataset)
    assert task.dim == 4


def test_logistic_grad_computer_requires_task_type() -> None:
    grad_computer = LogisticGradComputer()
    model = NumpyVectorModel(np.zeros(2))
    with pytest.raises(TypeError):
        grad_computer.grad(task=object(), model=model, batch=None)
