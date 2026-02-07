from __future__ import annotations

import numpy as np
import pytest


torch = pytest.importorskip("torch")


def test_muon_step_with_closure_updates_params() -> None:
    from optim.muon import Muon

    model = torch.nn.Linear(3, 1)
    x = torch.randn(8, 3)
    y = torch.randn(8, 1)

    optimizer = Muon(model.parameters(), lr=0.1, beta=0.5)

    def closure() -> torch.Tensor:
        return torch.nn.functional.mse_loss(model(x), y)

    optimizer.zero_grad()
    loss = torch.nn.functional.mse_loss(model(x), y)
    loss.backward()
    before = np.concatenate([p.detach().cpu().numpy().reshape(-1) for p in model.parameters()])
    optimizer.step(closure)
    after = np.concatenate([p.detach().cpu().numpy().reshape(-1) for p in model.parameters()])

    assert not np.allclose(before, after)


def test_torch_gd_step_with_closure_updates_params() -> None:
    from optim.torch_gd import TorchGD

    model = torch.nn.Linear(2, 1)
    x = torch.randn(6, 2)
    y = torch.randn(6, 1)

    optimizer = TorchGD(model.parameters(), lr=0.1, weight_decay=0.01)

    def closure() -> torch.Tensor:
        return torch.nn.functional.mse_loss(model(x), y)

    optimizer.zero_grad()
    loss = torch.nn.functional.mse_loss(model(x), y)
    loss.backward()
    before = np.concatenate([p.detach().cpu().numpy().reshape(-1) for p in model.parameters()])
    optimizer.step(closure)
    after = np.concatenate([p.detach().cpu().numpy().reshape(-1) for p in model.parameters()])

    assert not np.allclose(before, after)


def test_optimizers_skip_none_grads() -> None:
    from optim.muon import Muon
    from optim.torch_gd import TorchGD

    class TwoParamModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.used = torch.nn.Linear(2, 1)
            self.unused = torch.nn.Parameter(torch.zeros(1))

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.used(x)

    model = TwoParamModel()
    x = torch.randn(4, 2)
    y = torch.randn(4, 1)

    muon = Muon(model.parameters(), lr=0.1)
    gd = TorchGD(model.parameters(), lr=0.1)

    for opt in (muon, gd):
        opt.zero_grad()
        loss = torch.nn.functional.mse_loss(model(x), y)
        loss.backward()
        opt.step()


def test_optimizers_torch_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    import optim.muon as muon_module
    import optim.torch_gd as gd_module

    monkeypatch.setattr(muon_module, "TORCH_AVAILABLE", False)
    with pytest.raises(ImportError):
        muon_module._check_torch()

    monkeypatch.setattr(gd_module, "TORCH_AVAILABLE", False)
    with pytest.raises(ImportError):
        gd_module._check_torch()
