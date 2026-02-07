from __future__ import annotations

import itertools
import numpy as np
import pytest

import benchmarks.checks as checks


def test_get_init_state_for_gd() -> None:
    state = checks._get_init_state("gd")
    assert state.t == 0


def test_check_single_decreases_suboptimality_invalid_initial(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(checks, "suboptimality", lambda task, x: float("inf"))
    result = checks.check_single_decreases_suboptimality({"optimizer": "gd", "steps": 1})
    assert result.passed is False
    assert "Initial suboptimality" in result.details.get("error", "")


def test_check_single_decreases_suboptimality_invalid_final(monkeypatch: pytest.MonkeyPatch) -> None:
    values = itertools.cycle([1.0, float("inf")])

    def _fake_subopt(task, x):
        return next(values)

    monkeypatch.setattr(checks, "suboptimality", _fake_subopt)
    result = checks.check_single_decreases_suboptimality({"optimizer": "gd", "steps": 1})
    assert result.passed is False
    assert "Final suboptimality" in result.details.get("error", "")


def test_check_constraint_feasibility_gossip_ring_violation(monkeypatch: pytest.MonkeyPatch) -> None:
    config = {
        "env": "gossip",
        "optimizer": "fw",
        "constraint": "l2ball",
        "radius": 1e-6,
        "steps": 1,
        "dim": 3,
        "seed": 0,
        "n_nodes": 2,
        "topology": "ring",
        "strategy": "local_then_gossip",
    }
    monkeypatch.setattr(
        checks.GossipEnvironment,
        "get_params_by_node",
        lambda self: {0: np.array([1.0, 0.0, 0.0]), 1: np.array([2.0, 0.0, 0.0])},
    )
    result = checks.check_constraint_feasibility(config)
    assert result.passed is False
    assert "node_norms" in result.details


def test_check_gossip_consensus_decreases_zero_initial(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(checks, "consensus_error", lambda params: 0.0)
    config = {
        "env": "gossip",
        "optimizer": "gd",
        "steps": 1,
        "dim": 3,
        "seed": 0,
        "n_nodes": 2,
        "topology": "ring",
        "strategy": "local_then_gossip",
    }
    result = checks.check_gossip_consensus_decreases(config)
    assert result.passed is True
