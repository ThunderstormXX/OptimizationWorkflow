from __future__ import annotations

import numpy as np
import pytest

from benchmarks.metrics import mean_params


def test_mean_params_empty_raises() -> None:
    with pytest.raises(ValueError):
        mean_params({})


def test_mean_params_orders_nodes() -> None:
    params = {
        1: np.array([1.0, 1.0]),
        0: np.array([3.0, 3.0]),
    }
    result = mean_params(params)
    assert np.allclose(result, np.array([2.0, 2.0]))
