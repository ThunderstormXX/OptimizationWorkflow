from __future__ import annotations

import numpy as np
import pytest

from models import torch_adapter as torch_adapter_module
from models import torch_nets as torch_nets_module


torch = pytest.importorskip("torch")


def test_torch_adapter_check_torch_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch_adapter_module, "TORCH_AVAILABLE", False)
    with pytest.raises(ImportError):
        torch_adapter_module._check_torch()
    monkeypatch.setattr(torch_adapter_module, "TORCH_AVAILABLE", True)


def test_torch_nets_check_torch_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch_nets_module, "TORCH_AVAILABLE", False)
    with pytest.raises(ImportError):
        torch_nets_module._check_torch()
    monkeypatch.setattr(torch_nets_module, "TORCH_AVAILABLE", True)


def test_torch_adapter_set_params_wrong_shape() -> None:
    from models.torch_nets import MLP3

    model = torch_adapter_module.TorchModelAdapter(MLP3(hidden=8))
    with pytest.raises(ValueError):
        model.set_parameters_vector(np.zeros(model.dim + 1))


def test_torch_adapter_get_grad_vector_requires_backward() -> None:
    from models.torch_nets import MLP3

    model = torch_adapter_module.TorchModelAdapter(MLP3(hidden=8))
    with pytest.raises(RuntimeError):
        model.get_grad_vector()


def test_torch_adapter_forward_with_numpy() -> None:
    from models.torch_nets import MLP3

    model = torch_adapter_module.TorchModelAdapter(MLP3(hidden=8))
    x = np.random.randn(2, 1, 28, 28).astype(np.float32)
    logits = model.forward(x)
    assert logits.shape == (2, 10)


def test_torch_adapter_forward_with_list() -> None:
    from models.torch_nets import MLP3

    model = torch_adapter_module.TorchModelAdapter(MLP3(hidden=8))
    x = [[[[0.0] * 28] * 28]]
    logits = model.forward(x)
    assert logits.shape[0] == 1
