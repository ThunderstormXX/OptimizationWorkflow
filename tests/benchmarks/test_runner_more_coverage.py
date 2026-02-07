from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from benchmarks import legacy_runner as runner
from models.torch_adapter import TorchModelAdapter
from models.torch_nets import MLP3


torch = pytest.importorskip("torch")


def _quadratic_config(env: str) -> dict[str, object]:
    base = {
        "env": env,
        "task": "quadratic",
        "optimizer": "fw",
        "constraint": "l2ball",
        "radius": 1.0,
        "step_schedule": "harmonic",
        "gamma": 0.2,
        "steps": 2,
        "seed": 0,
        "dim": 3,
        "cond": 10.0,
        "lr": 0.1,
    }
    if env == "gossip":
        base.update({"n_nodes": 2, "topology": "ring", "strategy": "local_then_gossip"})
    return base


def test_run_single_and_gossip_quadratic_with_trace(tmp_path: Path) -> None:
    out_dir = tmp_path / "single"
    out_dir.mkdir(parents=True, exist_ok=True)
    runner._run_single_env_quadratic(  # type: ignore[attr-defined]
        _quadratic_config("single"),
        runner.SyntheticQuadraticTask(
            runner.make_spd_quadratic(dim=3, rng=np.random.default_rng(0), cond=5.0)
        ),
        out_dir,
        save_history=True,
        save_trace=True,
    )
    assert (out_dir / "trace.jsonl").exists()

    out_dir_gossip = tmp_path / "gossip"
    out_dir_gossip.mkdir(parents=True, exist_ok=True)
    runner._run_gossip_env_quadratic(  # type: ignore[attr-defined]
        _quadratic_config("gossip"),
        runner.SyntheticQuadraticTask(
            runner.make_spd_quadratic(dim=3, rng=np.random.default_rng(1), cond=5.0)
        ),
        out_dir_gossip,
        save_history=True,
        save_trace=True,
    )
    assert (out_dir_gossip / "trace.jsonl").exists()


def test_run_single_logistic_with_trace(tmp_path: Path) -> None:
    config = {
        "env": "single",
        "task": "logistic",
        "optimizer": "gd",
        "steps": 2,
        "seed": 0,
        "dim": 3,
        "n_samples": 30,
        "batch_size": 10,
        "lr": 0.1,
        "constraint": "none",
        "step_schedule": "na",
        "gamma": 0.0,
        "radius": 0.0,
    }
    out_dir = tmp_path / "logistic"
    out_dir.mkdir(parents=True, exist_ok=True)
    runner._run_single_env_logistic(  # type: ignore[attr-defined]
        config,
        out_dir,
        save_history=True,
        save_trace=True,
    )
    assert (out_dir / "trace.jsonl").exists()


def test_run_single_mnist_fake_and_fixture(tmp_path: Path) -> None:
    config_fake = {
        "env": "single",
        "task": "mnist",
        "model": "mlp3",
        "optimizer": "fw",
        "constraint": "l2ball",
        "radius": "auto",
        "step_schedule": "harmonic",
        "gamma": 0.2,
        "steps": 2,
        "seed": 0,
        "batch_size": 4,
        "n_train": 20,
        "n_val": 10,
        "data_root": ".data",
        "device": "cpu",
        "eval_every": 1,
        "eval_batches": 1,
        "use_fake_data": True,
        "mnist_fixture": None,
        "download": False,
        "lr": 0.001,
        "hidden": 16,
    }
    out_dir = tmp_path / "mnist_fake"
    out_dir.mkdir(parents=True, exist_ok=True)
    runner._run_single_env_mnist(  # type: ignore[attr-defined]
        config_fake,
        out_dir,
        save_history=True,
        save_trace=True,
    )
    assert (out_dir / "trace.jsonl").exists()

    config_fixture = dict(config_fake)
    config_fixture["mnist_fixture"] = "tests/assets/mnist_tiny"
    config_fixture["steps"] = 1
    out_dir_fixture = tmp_path / "mnist_fixture"
    out_dir_fixture.mkdir(parents=True, exist_ok=True)
    runner._run_single_env_mnist(  # type: ignore[attr-defined]
        config_fixture,
        out_dir_fixture,
        save_history=True,
        save_trace=False,
    )


def test_run_gossip_mnist_fake(tmp_path: Path) -> None:
    config = {
        "env": "gossip",
        "task": "mnist",
        "model": "mlp3",
        "optimizer": "adam",
        "constraint": "none",
        "radius": 1.0,
        "steps": 1,
        "seed": 0,
        "batch_size": 4,
        "n_train": 20,
        "n_val": 10,
        "data_root": ".data",
        "device": "cpu",
        "eval_every": 1,
        "eval_batches": 1,
        "use_fake_data": True,
        "mnist_fixture": None,
        "download": False,
        "lr": 0.001,
        "hidden": 16,
        "n_nodes": 2,
        "topology": "ring",
        "strategy": "local_then_gossip",
        "heterogeneity": "iid",
    }
    out_dir = tmp_path / "mnist_gossip"
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = runner._run_gossip_env_mnist(  # type: ignore[attr-defined]
        config,
        out_dir,
        save_history=True,
        save_trace=True,
    )
    assert summary["env"] == "gossip"


def test_build_mnist_optimizer_branches() -> None:
    model = TorchModelAdapter(MLP3(hidden=8), device="cpu")
    base_config = {"lr": 0.001, "step_schedule": "harmonic", "gamma": 0.2}

    runner._build_mnist_optimizer("adam", base_config, model, "l2ball", 1.0)
    runner._build_mnist_optimizer("adam", base_config, model, "none", 1.0)
    runner._build_mnist_optimizer("gd", base_config, model, "none", 1.0)
    runner._build_mnist_optimizer("pgd", base_config, model, "l2ball", 1.0)
    runner._build_mnist_optimizer("fw", base_config, model, "l2ball", 1.0)

    with pytest.raises(ValueError):
        runner._build_mnist_optimizer("pgd", base_config, model, "none", 1.0)
    with pytest.raises(ValueError):
        runner._build_mnist_optimizer("fw", base_config, model, "none", 1.0)
    with pytest.raises(ValueError):
        runner._build_mnist_optimizer("unknown", base_config, model, "none", 1.0)


def test_summary_to_csv_row_mnist() -> None:
    row = runner.summary_to_csv_row(
        "run_0000",
        {"task": "mnist", "model": "cnn", "optimizer": "adam"},
    )
    assert row["model"] == "cnn"


def test_run_checks_mode_default_steps(tmp_path: Path) -> None:
    args = runner.parse_args(["--mode", "checks"])
    exp_dir = tmp_path / "checks"
    exp_dir.mkdir(parents=True, exist_ok=True)
    summary, passed = runner.run_checks_mode(args, exp_dir)
    assert "num_checks" in summary
    assert isinstance(passed, bool)


def test_run_single_mode_animation_symlink_fallback(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    args = runner.parse_args(
        [
            "--mode",
            "single",
            "--env",
            "single",
            "--steps",
            "1",
            "--dim",
            "2",
            "--animate",
            "--animate-format",
            "gif",
        ]
    )

    exp_dir = tmp_path / "single_mode"
    exp_dir.mkdir(parents=True, exist_ok=True)

    def _fake_render_animation(*, out_path: Path, **_kwargs):  # type: ignore[no-untyped-def]
        out_path.write_bytes(b"gif")

    monkeypatch.setattr(runner, "render_animation", _fake_render_animation)
    monkeypatch.setattr(Path, "symlink_to", lambda *args, **kwargs: (_ for _ in ()).throw(OSError()))

    summary = runner.run_single_mode(args, exp_dir, render=True, animate=True)
    assert "final_mean_loss" in summary


def test_main_prints_final_accuracy(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        runner,
        "run_single_mode",
        lambda *_args, **_kwargs: {"final_mean_loss": 1.0, "final_accuracy": 0.5},
    )
    exit_code = runner.main(["--mode", "single"])
    assert exit_code == 0
