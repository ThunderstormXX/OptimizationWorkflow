from __future__ import annotations

import numpy as np
import pytest

from core.types import History, StepMeta, StepResult


def test_history_basic_aggregation() -> None:
    history = History()
    history.append(StepResult(loss=1.0, metrics={"acc": 0.5}), StepMeta(2, 1))
    history.append(StepResult(loss=3.0, metrics={"acc": 0.9}), StepMeta(1, 2))

    assert len(history) == 2
    assert history.last().loss == 3.0
    assert history.last_meta().num_grad_evals == 1
    assert history.mean_loss() == pytest.approx(2.0)
    assert history.mean_metric("acc") == pytest.approx(0.7)
    assert history.total_grad_evals() == 3
    assert history.total_gossip_rounds() == 3


def test_history_multi_node_aggregation() -> None:
    history = History()
    record = {
        0: StepResult(loss=1.0, metrics={"m": 2.0}),
        1: StepResult(loss=3.0, metrics={"m": 4.0}),
    }
    history.append(record)
    assert history.mean_loss() == pytest.approx(2.0)
    assert history.mean_metric("m") == pytest.approx(3.0)


def test_history_empty_raises() -> None:
    history = History()
    with pytest.raises(ValueError):
        history.mean_loss()
    with pytest.raises(ValueError):
        history.mean_metric("acc")
    with pytest.raises(IndexError):
        _ = history.last()
    with pytest.raises(IndexError):
        _ = history.last_meta()
