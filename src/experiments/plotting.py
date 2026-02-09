"""Plotting helpers for experiments."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def plot_metric_by_optimizer(
    aggregates: dict[str, dict[str, Any]],
    metric: str,
    out_path: Path,
    *,
    title: str | None = None,
    upper_only: bool = False,
    loglog: bool = False,
) -> None:
    """Plot mean +- std curves for a metric across optimizers.

    Args:
        aggregates: Mapping optimizer_name -> aggregate dict with metrics.
        metric: Metric key to plot.
        out_path: Output PNG path.
        title: Optional plot title.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(8, 4.5))

    has_data = False
    for opt_name, aggregate in aggregates.items():
        metric_data = aggregate.get("metrics", {}).get(metric)
        if not metric_data:
            continue
        mean = np.asarray(metric_data.get("mean", []), dtype=np.float64)
        std = np.asarray(metric_data.get("std", []), dtype=np.float64)
        if mean.size == 0:
            continue
        if loglog:
            x = np.arange(1, mean.size + 1)
            mask = np.isfinite(mean) & (mean > 0)
        else:
            x = np.arange(mean.size)
            mask = np.isfinite(mean)

        if not np.any(mask):
            continue

        x_plot = x[mask]
        mean_plot = mean[mask]
        if loglog:
            mean_plot = np.clip(mean_plot, 1e-12, None)

        plt.plot(x_plot, mean_plot, label=opt_name)

        if std.size == mean.size:
            upper = mean + std
            upper = upper[mask]
            if loglog:
                upper = np.clip(upper, 1e-12, None)
            if upper_only:
                plt.plot(x_plot, upper, linestyle="--", alpha=0.7)
            else:
                lower = mean - std
                lower = lower[mask]
                if loglog:
                    lower = np.clip(lower, 1e-12, None)
                plt.fill_between(x_plot, lower, upper, alpha=0.2)
        has_data = True

    if not has_data:
        plt.close()
        return

    plt.xlabel("epoch")
    plt.ylabel(metric)
    if title:
        plt.title(title)
    plt.legend(loc="best")
    if loglog:
        plt.xscale("log")
        plt.yscale("log")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
