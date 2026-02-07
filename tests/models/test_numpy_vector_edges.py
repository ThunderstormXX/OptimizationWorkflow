from __future__ import annotations

import numpy as np
import pytest

from models.numpy_vector import NumpyVectorModel


def test_numpy_vector_forward_returns_none() -> None:
    model = NumpyVectorModel(np.zeros(3))
    assert model.forward(None) is None


def test_numpy_vector_rejects_wrong_shape() -> None:
    model = NumpyVectorModel(np.zeros(3))
    with pytest.raises(ValueError):
        model.set_parameters_vector(np.zeros(4))
