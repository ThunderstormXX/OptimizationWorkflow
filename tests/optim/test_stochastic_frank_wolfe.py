from __future__ import annotations

import numpy as np
import pytest


torch = pytest.importorskip("torch")


def _params_to_numpy(model: torch.nn.Module) -> np.ndarray:
    return np.concatenate([p.detach().cpu().numpy().reshape(-1) for p in model.parameters()])


def test_stochastic_frank_wolfe_step_updates_params() -> None:
    from optim.stochastic_frank_wolfe import StochasticFrankWolfe

    model = torch.nn.Linear(2, 1, bias=False)
    x = torch.randn(8, 2)
    y = torch.randn(8, 1)

    optimizer = StochasticFrankWolfe(
        model.parameters(),
        constraint={"class": "optim.constraints.L2BallConstraint", "params": {"radius": "auto"}},
        step_size="constant",
        gamma=0.5,
    )

    before = _params_to_numpy(model)
    loss = torch.nn.functional.mse_loss(model(x), y)
    loss.backward()
    optimizer.step()
    after = _params_to_numpy(model)

    assert not np.allclose(before, after)


def test_stochastic_frank_wolfe_projection_and_harmonic() -> None:
    from optim.constraints import L2BallConstraint
    from optim.stochastic_frank_wolfe import StochasticFrankWolfe

    model = torch.nn.Linear(4, 1, bias=False)
    with torch.no_grad():
        for p in model.parameters():
            p.add_(10.0)

    constraint = L2BallConstraint(radius=0.1)
    optimizer = StochasticFrankWolfe(model.parameters(), constraint=constraint, step_size="harmonic")

    params = _params_to_numpy(model)
    assert np.linalg.norm(params) <= 0.1 + 1e-6


def test_stochastic_frank_wolfe_invalid_step_size() -> None:
    from optim.stochastic_frank_wolfe import StochasticFrankWolfe

    model = torch.nn.Linear(2, 1)
    with pytest.raises(ValueError):
        StochasticFrankWolfe(
            model.parameters(),
            constraint={"class": "optim.constraints.L2BallConstraint", "params": {"radius": 1.0}},
            step_size="constant",
        )


def test_stochastic_frank_wolfe_numeric_and_callable_step_size() -> None:
    from optim.constraints import L2BallConstraint
    from optim.stochastic_frank_wolfe import StochasticFrankWolfe

    model = torch.nn.Linear(2, 1)
    StochasticFrankWolfe(model.parameters(), constraint=L2BallConstraint(radius=1.0), step_size=0.5)
    StochasticFrankWolfe(model.parameters(), constraint=L2BallConstraint(radius=1.0), step_size=lambda t: 0.5)

    StochasticFrankWolfe(
        model.parameters(),
        constraint={"class": "optim.constraints:L2BallConstraint", "params": {"radius": 1.0}},
        step_size="harmonic",
    )


def test_stochastic_frank_wolfe_dict_step_size() -> None:
    from optim.constraints import L2BallConstraint
    from optim.stochastic_frank_wolfe import StochasticFrankWolfe

    model = torch.nn.Linear(2, 1)
    StochasticFrankWolfe(
        model.parameters(),
        constraint=L2BallConstraint(radius=1.0),
        step_size={"type": "harmonic"},
    )

    with pytest.raises(ValueError):
        StochasticFrankWolfe(
            model.parameters(),
            constraint=L2BallConstraint(radius=1.0),
            step_size={"type": "constant"},
        )


def test_stochastic_frank_wolfe_missing_constraint() -> None:
    from optim.stochastic_frank_wolfe import StochasticFrankWolfe

    model = torch.nn.Linear(1, 1)
    with pytest.raises(ValueError):
        StochasticFrankWolfe(model.parameters(), constraint=None)


def test_stochastic_frank_wolfe_project_not_implemented() -> None:
    from optim.stochastic_frank_wolfe import StochasticFrankWolfe

    class Constraint:
        def lmo(self, grad):
            return grad * 0.0

        def project(self, x):
            raise NotImplementedError

    model = torch.nn.Linear(1, 1, bias=False)
    StochasticFrankWolfe(model.parameters(), constraint=Constraint(), step_size="harmonic")


def test_stochastic_frank_wolfe_project_not_callable() -> None:
    from optim.stochastic_frank_wolfe import StochasticFrankWolfe

    class Constraint:
        def lmo(self, grad):
            return grad * 0.0

    model = torch.nn.Linear(1, 1, bias=False)
    StochasticFrankWolfe(model.parameters(), constraint=Constraint(), step_size="harmonic")


def test_stochastic_frank_wolfe_empty_params_step() -> None:
    from optim.constraints import L2BallConstraint
    from optim.stochastic_frank_wolfe import StochasticFrankWolfe

    with pytest.raises(ValueError):
        StochasticFrankWolfe([], constraint=L2BallConstraint(radius=1.0), step_size="harmonic")


def test_stochastic_frank_wolfe_invalid_gamma_value() -> None:
    from optim.constraints import L2BallConstraint
    from optim.stochastic_frank_wolfe import StochasticFrankWolfe

    model = torch.nn.Linear(1, 1)
    optimizer = StochasticFrankWolfe(
        model.parameters(),
        constraint=L2BallConstraint(radius=1.0),
        step_size=lambda t: 2.0,
    )
    model(torch.randn(1, 1)).sum().backward()
    with pytest.raises(ValueError):
        optimizer.step()


def test_stochastic_frank_wolfe_torch_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    import optim.stochastic_frank_wolfe as fw_module

    monkeypatch.setattr(fw_module, "TORCH_AVAILABLE", False)
    with pytest.raises(ImportError):
        fw_module._check_torch()
