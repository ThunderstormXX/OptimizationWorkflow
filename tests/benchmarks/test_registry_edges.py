from __future__ import annotations

import pytest

import pytest

from benchmarks import registry
from optim.legacy_frankwolfe import FrankWolfeOptimizer


def test_register_optimizer_duplicate() -> None:
    def _factory(_config):
        return object()

    name = "_tmp_opt"
    registry.register_optimizer(name, _factory)
    try:
        with pytest.raises(ValueError):
            registry.register_optimizer(name, _factory)
    finally:
        registry.OPTIMIZERS.pop(name, None)


def test_get_optimizer_unknown() -> None:
    with pytest.raises(KeyError):
        registry.get_optimizer("missing_opt", {})


def test_register_strategy_duplicate() -> None:
    def _factory():
        return object()

    name = "_tmp_strat"
    registry.register_strategy(name, _factory)
    try:
        with pytest.raises(ValueError):
            registry.register_strategy(name, _factory)
    finally:
        registry.STRATEGIES.pop(name, None)


def test_get_strategy_unknown() -> None:
    with pytest.raises(KeyError):
        registry.get_strategy("missing_strategy")


def test_get_strategy_with_config_gradient_tracking() -> None:
    strategy = registry.get_strategy_with_config("gradient_tracking", {"lr": 0.123})
    assert getattr(strategy, "lr") == pytest.approx(0.123)


def test_get_optimizer_fw_with_simplex_constraint() -> None:
    opt = registry.get_optimizer("fw", {"constraint": "simplex", "dim": 3})
    assert isinstance(opt, FrankWolfeOptimizer)


def test_get_strategy_gradient_tracking_factory() -> None:
    strategy = registry.get_strategy("gradient_tracking")
    assert getattr(strategy, "lr") == pytest.approx(0.1)
