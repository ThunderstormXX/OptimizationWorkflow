from __future__ import annotations

import numpy as np
import pytest


torch = pytest.importorskip("torch")


def test_muon_step_with_closure_updates_params() -> None:
    from optim.muon import SingleDeviceMuon

    model = torch.nn.Linear(3, 1)
    x = torch.randn(8, 3)
    y = torch.randn(8, 1)

    optimizer = SingleDeviceMuon(model.parameters(), lr=0.1, momentum=0.5)

    def closure() -> torch.Tensor:
        return torch.nn.functional.mse_loss(model(x), y)

    optimizer.zero_grad()
    loss = torch.nn.functional.mse_loss(model(x), y)
    loss.backward()
    before = np.concatenate([p.detach().cpu().numpy().reshape(-1) for p in model.parameters()])
    optimizer.step(closure)
    after = np.concatenate([p.detach().cpu().numpy().reshape(-1) for p in model.parameters()])

    assert not np.allclose(before, after)


def test_torch_sgd_step_with_closure_updates_params() -> None:
    model = torch.nn.Linear(2, 1)
    x = torch.randn(6, 2)
    y = torch.randn(6, 1)

    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, weight_decay=0.01)

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
    from optim.muon import SingleDeviceMuon

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

    muon = SingleDeviceMuon(model.parameters(), lr=0.1)
    gd = torch.optim.SGD(model.parameters(), lr=0.1)

    for opt in (muon, gd):
        opt.zero_grad()
        loss = torch.nn.functional.mse_loss(model(x), y)
        loss.backward()
        opt.step()


def test_optimizers_torch_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    import optim.muon as muon_module

    monkeypatch.setattr(muon_module, "TORCH_AVAILABLE", False)
    with pytest.raises(ImportError):
        muon_module._check_torch()


def test_step_sizes_module() -> None:
    from optim.legacy_frankwolfe import constant_step_size, harmonic_step_size

    schedule = constant_step_size(0.5)
    assert schedule(0) == pytest.approx(0.5)
    assert schedule(10) == pytest.approx(0.5)

    harmonic = harmonic_step_size()
    assert harmonic(0) == pytest.approx(1.0)
    assert harmonic(1) == pytest.approx(2 / 3)

    with pytest.raises(ValueError):
        constant_step_size(0.0)


def test_scion_updates_params_and_init() -> None:
    from optim.scion import Scion

    model = torch.nn.Linear(2, 2, bias=False)
    optimizer = Scion(
        model.parameters(),
        lr=0.1,
        momentum=0.9,
        norm="RowNorm",
        norm_kwargs={"normalized": True},
    )
    optimizer.init()

    x = torch.randn(4, 2)
    y = torch.randn(4, 2)
    optimizer.zero_grad()
    loss = torch.nn.functional.mse_loss(model(x), y)
    loss.backward()
    before = np.concatenate([p.detach().cpu().numpy().reshape(-1) for p in model.parameters()])
    optimizer.step()
    after = np.concatenate([p.detach().cpu().numpy().reshape(-1) for p in model.parameters()])
    assert not np.allclose(before, after)


def test_scion_light_state_roundtrip() -> None:
    from optim.scion import ScionLight

    model = torch.nn.Linear(3, 1, bias=False)
    optimizer = ScionLight(model.parameters(), lr=0.05, momentum=0.9)

    x = torch.randn(5, 3)
    y = torch.randn(5, 1)
    optimizer.zero_grad()
    loss = torch.nn.functional.mse_loss(model(x), y)
    loss.backward()
    optimizer.step()

    state = optimizer.state_dict()
    optimizer.load_state_dict(state)


def test_scion_norms_basic_paths() -> None:
    import optim.scion as scion_module

    col = scion_module.ColNorm(normalized=True, transpose=True)
    row = scion_module.RowNorm(normalized=False, transpose=True)
    bias = scion_module.BiasRMS()
    spec_conv = scion_module.SpectralConv(steps=2)
    spec = scion_module.Spectral(max=True, normalized=False, steps=2)
    sign = scion_module.Sign(zero_init=True, normalized=True)
    auto = scion_module.Auto()

    g2 = torch.randn(4, 3)
    g1 = torch.randn(3)
    g4 = torch.randn(2, 2, 2, 2)

    col.lmo(g2)
    row.lmo(g2)
    bias.lmo(g1)
    spec.lmo(g2)
    sign.lmo(g2)
    spec_conv.lmo(g4)

    w2 = torch.randn(4, 3)
    w1 = torch.randn(3)
    w4 = torch.randn(2, 2, 2, 2)
    col.init(w2)
    row.init(w2)
    bias.init(w1)
    spec.init(w2)
    sign.init(w2)
    spec_conv.init(w4)

    auto.lmo(g2)
    auto.lmo(g4)
    auto.lmo(g1)
    auto.init(w2)
    auto.init(w4)
    auto.init(w1)
