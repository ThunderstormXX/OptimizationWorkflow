from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from benchmarks import animate as animate_module


def _make_trace() -> list[dict[str, object]]:
    return [
        {
            "t": 0,
            "env": "gossip",
            "node_metrics": {
                "0": {"loss": 1.0, "accuracy": 0.5, "param_norm": 1.0},
                "1": {"loss": 1.2, "accuracy": 0.4, "param_norm": 1.1},
            },
            "mean_metrics": {"mean_loss": 1.1, "mean_accuracy": 0.45, "consensus_error": 0.1},
        },
        {
            "t": 1,
            "env": "gossip",
            "node_metrics": {
                "0": {"loss": 0.9, "accuracy": 0.6, "param_norm": 0.9},
                "1": {"loss": 1.0, "accuracy": 0.5, "param_norm": 1.0},
            },
            "mean_metrics": {"mean_loss": 0.95, "mean_accuracy": 0.55, "consensus_error": 0.05},
        },
    ]


def test_build_metadata_strings() -> None:
    title1, title2, filename = animate_module.build_metadata_strings(
        {
            "task": "mnist",
            "model": "cnn",
            "optimizer": "fw",
            "constraint": "l2ball",
            "radius": 2.0,
            "env": "gossip",
            "topology": "ring",
            "strategy": "local_then_gossip",
            "seed": 0,
            "steps": 5,
        }
    )

    assert "MNIST" in title1
    assert "cnn" in title1
    assert "FW" in title1
    assert "env=gossip" in title2
    assert filename.startswith("animation__")


def test_build_metadata_strings_defaults() -> None:
    title1, _title2, _filename = animate_module.build_metadata_strings(
        {"task": "mnist", "model": "NumpyVector", "constraint": "l1ball", "radius": 1.0}
    )
    assert "NumpyVector" in title1
    assert "L1" in title1


def test_read_write_trace_jsonl(tmp_path: Path) -> None:
    trace_path = tmp_path / "trace.jsonl"
    with trace_path.open("w", encoding="utf-8") as f:
        animate_module.write_trace_line(
            f,
            t=0,
            env="single",
            node_metrics={"0": {"loss": 1.0}},
            mean_metrics={"mean_loss": 1.0},
        )

    records = animate_module.read_trace_jsonl(trace_path)
    assert records[0]["t"] == 0


def test_internal_edge_helpers() -> None:
    assert animate_module._get_edges_from_comm_graph(None) == []
    edges = animate_module._get_edges_fallback(3, "complete")
    assert len(edges) == 3


def test_draw_panels_edge_cases(tmp_path: Path) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    animate_module._draw_graph_panel_with_edges(
        ax,
        node_metrics={},
        topology="ring",
        n_nodes=0,
        is_gossip=False,
        comm_graph=None,
        current_idx=0,
        total_frames=1,
    )
    plt.close(fig)

    # Table panel with empty data
    fig, ax = plt.subplots()
    animate_module._draw_table_panel(ax, node_metrics={}, task="quadratic")
    plt.close(fig)

    # Table panel with >10 rows
    fig, ax = plt.subplots()
    node_metrics = {str(i): {"loss": float(i)} for i in range(12)}
    animate_module._draw_table_panel(ax, node_metrics=node_metrics, task="quadratic")
    plt.close(fig)


def test_render_frame_consensus_only() -> None:
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    trace = [
        {
            "t": 0,
            "env": "gossip",
            "node_metrics": {"0": {"loss": 1.0}, "1": {"loss": 1.2}},
            "mean_metrics": {"mean_loss": 1.1, "consensus_error": 0.2},
        }
    ]
    fig = plt.figure(figsize=(6, 4))
    gs = GridSpec(2, 2, figure=fig)
    animate_module._render_frame(
        fig,
        gs,
        trace,
        0,
        topology="ring",
        config={"task": "quadratic"},
        title_line1="t1",
        title_line2="t2",
    )
    plt.close(fig)


def test_render_animation_gif(tmp_path: Path) -> None:
    out_path = tmp_path / "anim.gif"
    trace = _make_trace()

    animate_module.render_animation(
        trace=trace,
        out_path=out_path,
        topology="ring",
        fps=2,
        max_steps=5,
        title="test",
        config={"task": "logistic"},
        output_format="gif",
    )

    assert out_path.exists()
    assert out_path.stat().st_size > 0


def test_render_animation_subsample_and_mp4(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    trace = _make_trace() * 3

    monkeypatch.setattr(animate_module, "check_ffmpeg_available", lambda: True)
    monkeypatch.setattr(animate_module.imageio, "mimsave", lambda *args, **kwargs: None)
    monkeypatch.setattr(animate_module.np, "linspace", lambda *args, **kwargs: np.array([0, 0]))

    out_path = tmp_path / "anim.mp4"
    animate_module.render_animation(
        trace=trace,
        out_path=out_path,
        topology="ring",
        fps=2,
        max_steps=2,
        title="test",
        config={"task": "logistic"},
        output_format="mp4",
    )


def test_render_animation_mp4_requires_ffmpeg(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(animate_module, "check_ffmpeg_available", lambda: False)
    out_path = tmp_path / "anim.mp4"
    trace = _make_trace()

    with pytest.raises(RuntimeError):
        animate_module.render_animation(
            trace=trace,
            out_path=out_path,
            topology="ring",
            fps=2,
            max_steps=5,
            title="test",
            config={"task": "logistic"},
            output_format="mp4",
        )


def test_render_animation_legacy_handles_mp4_failure(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    out_gif = tmp_path / "legacy.gif"
    out_mp4 = tmp_path / "legacy.mp4"
    trace = _make_trace()

    def _fake_render_animation(*, output_format: str, **_kwargs):  # type: ignore[no-untyped-def]
        if output_format == "mp4":
            raise RuntimeError("no ffmpeg")
        return None

    monkeypatch.setattr(animate_module, "render_animation", _fake_render_animation)
    animate_module.render_animation_legacy(
        trace=trace,
        out_gif=out_gif,
        out_mp4=out_mp4,
        topology="ring",
        fps=2,
        max_steps=2,
        title="legacy",
        config={"task": "logistic"},
    )
