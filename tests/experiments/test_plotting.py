from __future__ import annotations

from pathlib import Path

from experiments.plotting import plot_metric_by_optimizer


def test_plot_metric_by_optimizer_creates_file(tmp_path: Path) -> None:
    aggregates = {
        "opt": {
            "metrics": {"loss": {"mean": [1.0, 0.5], "std": [0.1, 0.05]}},
            "epochs": 2,
        }
    }
    out_path = tmp_path / "plot.png"
    plot_metric_by_optimizer(aggregates, "loss", out_path, title="demo")
    assert out_path.exists()


def test_plot_metric_by_optimizer_no_data(tmp_path: Path) -> None:
    aggregates = {"opt": {"metrics": {}, "epochs": 0}}
    out_path = tmp_path / "missing.png"
    plot_metric_by_optimizer(aggregates, "loss", out_path)
    assert not out_path.exists()


def test_plot_metric_by_optimizer_empty_mean(tmp_path: Path) -> None:
    aggregates = {"opt": {"metrics": {"loss": {"mean": [], "std": []}}, "epochs": 0}}
    out_path = tmp_path / "empty_mean.png"
    plot_metric_by_optimizer(aggregates, "loss", out_path)
    assert not out_path.exists()


def test_plot_metric_by_optimizer_upper_only_loglog(tmp_path: Path) -> None:
    aggregates = {
        "opt": {
            "metrics": {"loss": {"mean": [1.0, 0.5, 0.25], "std": [0.1, 0.05, 0.02]}},
            "epochs": 3,
        }
    }
    out_path = tmp_path / "loglog.png"
    plot_metric_by_optimizer(aggregates, "loss", out_path, upper_only=True, loglog=True)
    assert out_path.exists()
