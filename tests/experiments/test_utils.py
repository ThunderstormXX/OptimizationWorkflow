from __future__ import annotations

import pytest


torch = pytest.importorskip("torch")


def test_apply_model_init_and_optimizer_checks() -> None:
    from experiments.utils import apply_model_init, is_torch_optimizer, is_torch_optimizer_class

    model = torch.nn.Linear(4, 3)
    opt = torch.optim.SGD(model.parameters(), lr=0.1)

    assert is_torch_optimizer(opt)
    assert is_torch_optimizer_class(torch.optim.SGD)

    init_specs = [
        {"type": "zeros"},
        {"type": "ones"},
        {"type": "constant", "value": 0.5},
        {"type": "normal", "mean": 0.0, "std": 0.1},
        {"type": "uniform", "a": -0.2, "b": 0.2},
        {"type": "xavier_uniform"},
        {"type": "xavier_normal"},
        {"type": "kaiming_uniform", "a": 0, "nonlinearity": "relu"},
        {"type": "kaiming_normal", "a": 0, "nonlinearity": "relu"},
    ]

    for spec in init_specs:
        apply_model_init(model, spec, seed=0)

    apply_model_init(model, None)
    apply_model_init(model, "normal", seed=0)

    with pytest.raises(ValueError):
        apply_model_init(model, {"type": "unknown"})

    with pytest.raises(TypeError):
        apply_model_init(model, 123)


def test_utils_torch_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    import experiments.utils as utils

    monkeypatch.setattr(utils, "TORCH_AVAILABLE", False)
    with pytest.raises(ImportError):
        utils._check_torch()
    assert utils.is_torch_optimizer(object()) is False
    assert utils.is_torch_optimizer_class(object) is False
