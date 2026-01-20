"""Animation renderer for optimization visualizations.

This module provides functions for:
- Reading trace.jsonl files with per-step metrics
- Rendering animated GIFs/MP4s showing optimization progress
- Dynamic edge highlighting for communication visualization

Uses matplotlib for frame rendering and imageio for GIF/MP4 assembly.
"""

from __future__ import annotations

import io
import json
from pathlib import Path
from typing import Any

import imageio.v2 as imageio
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Circle

# Use Agg backend to avoid display issues
matplotlib.use("Agg")

__all__ = [
    "build_metadata_strings",
    "read_trace_jsonl",
    "render_animation",
    "write_trace_line",
]


# =============================================================================
# Metadata Helpers
# =============================================================================


def build_metadata_strings(config: dict[str, Any]) -> tuple[str, str, str]:
    """Build metadata strings for animation title and filename.

    Args:
        config: Configuration dictionary with experiment parameters.

    Returns:
        Tuple of (title_line1, title_line2, filename_base).
        - title_line1: Dataset | Model | Task formulation
        - title_line2: env/topo/strat/opt/constraint details
        - filename_base: Safe filename with metadata
    """
    task = config.get("task", "quadratic")

    # Dataset name
    if task == "mnist":
        dataset_name = "MNIST"
    elif task == "logistic":
        dataset_name = "SyntheticLogit"
    else:
        dataset_name = "SyntheticQuadratic"

    # Model name
    if task == "mnist":
        model_name = "MLP3(h=256)"
    else:
        model_name = "NumpyVector"

    # Task formulation
    if task in ("mnist", "logistic"):
        task_formulation = "classification"
    else:
        task_formulation = "convex quadratic"

    # Environment details
    env = config.get("env", "single")
    topology = config.get("topology", "none")
    strategy = config.get("strategy", "none")
    optimizer = config.get("optimizer", "gd")
    seed = config.get("seed", 0)
    steps = config.get("steps", 0)

    # Constraint string
    constraint = config.get("constraint", "none")
    radius = config.get("radius", 0.0)
    if constraint == "l1":
        constraint_str = f"L1(R={radius})"
    elif constraint == "l2":
        constraint_str = f"L2(R={radius})"
    else:
        constraint_str = "none"

    # Build title lines
    title_line1 = f"{dataset_name} | {model_name} | {task_formulation}"
    title_line2 = (
        f"env={env} topo={topology} strat={strategy} "
        f"opt={optimizer} constraint={constraint_str} seed={seed} steps={steps}"
    )

    # Build filename base (safe characters only)
    filename_base = (
        f"animation__{dataset_name}__{model_name}__{env}__{topology}"
        f"__{strategy}__{constraint_str.replace('(', '').replace(')', '').replace('=', '')}"
    )
    # Clean up filename
    filename_base = filename_base.replace(" ", "_").replace("/", "_")

    return title_line1, title_line2, filename_base


# =============================================================================
# Trace I/O
# =============================================================================


def read_trace_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read a trace.jsonl file and return list of step dictionaries.

    Each line in the file is a JSON object representing one step's full state.

    Args:
        path: Path to the trace.jsonl file.

    Returns:
        List of dictionaries, one per step.

    Raises:
        FileNotFoundError: If the file doesn't exist.
        json.JSONDecodeError: If a line is not valid JSON.
    """
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def write_trace_line(
    f: Any,
    *,
    t: int,
    env: str,
    node_metrics: dict[str, dict[str, float]],
    mean_metrics: dict[str, float],
    params_by_node: dict[str, list[float]] | None = None,
    comm: dict[str, Any] | None = None,
) -> None:
    """Write a single trace line to a file.

    Args:
        f: File handle opened for writing.
        t: Step number.
        env: Environment type ("single" or "gossip").
        node_metrics: Per-node metrics dict.
        mean_metrics: Aggregated mean metrics.
        params_by_node: Optional per-node parameter vectors.
        comm: Optional communication info (mode, active_edges, etc.).
    """
    record: dict[str, Any] = {
        "t": t,
        "env": env,
        "node_metrics": node_metrics,
        "mean_metrics": mean_metrics,
    }
    if params_by_node is not None:
        record["params_by_node"] = params_by_node
    if comm is not None:
        record["comm"] = comm
    f.write(json.dumps(record) + "\n")


# =============================================================================
# Graph Layout
# =============================================================================


def _compute_node_positions(n_nodes: int, topology: str | None) -> dict[int, tuple[float, float]]:
    """Compute 2D positions for nodes based on topology.

    Args:
        n_nodes: Number of nodes.
        topology: Topology type ("ring", "complete", or None).

    Returns:
        Dictionary mapping node ID to (x, y) position.
    """
    positions: dict[int, tuple[float, float]] = {}

    # Place nodes on a circle regardless of topology
    for i in range(n_nodes):
        angle = 2 * np.pi * i / n_nodes - np.pi / 2  # Start from top
        x = np.cos(angle)
        y = np.sin(angle)
        positions[i] = (float(x), float(y))

    return positions


def _get_edges_from_comm_graph(
    comm_graph: dict[str, Any] | None,
) -> list[tuple[int, int, float]]:
    """Extract edges and weights from comm_graph.

    Args:
        comm_graph: Communication graph dict with "edges" and "weights" keys.

    Returns:
        List of (i, j, weight) tuples.
    """
    if comm_graph is None:
        return []

    edges = comm_graph.get("edges", [])
    weights = comm_graph.get("weights", [])

    result: list[tuple[int, int, float]] = []
    for idx, edge in enumerate(edges):
        i, j = edge[0], edge[1]
        weight = weights[idx] if idx < len(weights) else 0.5
        result.append((i, j, weight))

    return result


def _get_edges_fallback(n_nodes: int, topology: str | None) -> list[tuple[int, int, float]]:
    """Get edges based on topology (fallback when no comm_graph).

    Args:
        n_nodes: Number of nodes.
        topology: Topology type.

    Returns:
        List of (i, j, weight) tuples with default weight 0.5.
    """
    edges: list[tuple[int, int, float]] = []

    if topology == "ring":
        for i in range(n_nodes):
            j = (i + 1) % n_nodes
            edges.append((min(i, j), max(i, j), 0.5))
    elif topology == "complete":
        for i in range(n_nodes):
            for j in range(i + 1, n_nodes):
                edges.append((i, j, 0.5))
    # For single env or unknown topology, no edges

    return edges


# =============================================================================
# Frame Rendering
# =============================================================================


def _render_frame(
    fig: Figure,
    gs: GridSpec,
    trace: list[dict[str, Any]],
    current_idx: int,
    topology: str | None,
    config: dict[str, Any],
    title_line1: str,
    title_line2: str,
    comm_graph: dict[str, Any] | None = None,
) -> None:
    """Render a single frame of the animation.

    Args:
        fig: Matplotlib figure.
        gs: GridSpec for layout.
        trace: Full trace data.
        current_idx: Index of current step in trace.
        topology: Topology type for graph layout.
        config: Configuration dictionary for banner.
        title_line1: First line of title (dataset/model/task).
        title_line2: Second line of title (env/topo/strat details).
        comm_graph: Optional communication graph with edges and weights.
    """
    fig.clear()

    current = trace[current_idx]
    t = current["t"]
    node_metrics = current.get("node_metrics", {})
    mean_metrics = current.get("mean_metrics", {})

    n_nodes = len(node_metrics)
    is_gossip = current.get("env") == "gossip" or n_nodes > 1

    # Determine task type from config or metrics
    task = config.get("task", "quadratic")
    has_accuracy = "mean_accuracy" in mean_metrics or any(
        "accuracy" in m for m in node_metrics.values()
    )
    has_suboptimality = "suboptimality" in mean_metrics
    has_consensus = "consensus_error" in mean_metrics

    # Two-line title banner
    fig.suptitle(
        f"{title_line1}\n{title_line2}\nstep {t}/{trace[-1]['t']}",
        fontsize=9,
        y=0.98,
    )

    # Top-left: Graph panel with dynamic edge highlighting
    ax_graph = fig.add_subplot(gs[0, 0])
    _draw_graph_panel_with_edges(
        ax_graph,
        node_metrics,
        topology,
        n_nodes,
        is_gossip,
        comm_graph,
        current_idx,
        len(trace),
    )

    # Top-right: Node table panel
    ax_table = fig.add_subplot(gs[0, 1])
    _draw_table_panel(ax_table, node_metrics, task, comm_graph)

    # Bottom-left: Loss curve
    ax_loss = fig.add_subplot(gs[1, 0])
    _draw_loss_curve(ax_loss, trace, current_idx)

    # Bottom-right: Secondary metric curve
    ax_secondary = fig.add_subplot(gs[1, 1])
    if has_accuracy:
        _draw_accuracy_curve(ax_secondary, trace, current_idx)
    elif has_suboptimality:
        _draw_suboptimality_curve(ax_secondary, trace, current_idx)
    elif has_consensus:
        _draw_consensus_curve(ax_secondary, trace, current_idx)
    else:
        ax_secondary.text(
            0.5,
            0.5,
            "No secondary metric",
            ha="center",
            va="center",
            transform=ax_secondary.transAxes,
        )
        ax_secondary.set_axis_off()

    fig.tight_layout(rect=(0, 0, 1, 0.90))


def _draw_graph_panel_with_edges(
    ax: Any,
    node_metrics: dict[str, dict[str, float]],
    topology: str | None,
    n_nodes: int,
    is_gossip: bool,
    comm_graph: dict[str, Any] | None,
    current_idx: int,
    total_frames: int,
) -> None:
    """Draw the graph panel with dynamic edge highlighting."""
    ax.set_title("Node Graph (comm: sync)", fontsize=10)

    if n_nodes == 0:
        ax.text(0.5, 0.5, "No nodes", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        return

    positions = _compute_node_positions(n_nodes, topology)

    # Get edges with weights
    if comm_graph is not None:
        edges_with_weights = _get_edges_from_comm_graph(comm_graph)
    elif is_gossip:
        edges_with_weights = _get_edges_fallback(n_nodes, topology)
    else:
        edges_with_weights = []

    # Compute max weight for normalization
    max_weight = max((w for _, _, w in edges_with_weights), default=1.0)
    if max_weight <= 0:
        max_weight = 1.0

    # Pulse parameters for animation
    # Pulse phase based on current frame
    pulse_phase = 2 * np.pi * current_idx / max(total_frames, 1)
    pulse_factor = 0.5 + 0.5 * np.sin(pulse_phase)  # 0 to 1

    # Limit edges for complete graphs (pulse only top 40)
    edges_to_draw = edges_with_weights
    if len(edges_to_draw) > 40:
        # Sort by weight (descending), then by (i, j) for determinism
        edges_to_draw = sorted(edges_to_draw, key=lambda e: (-e[2], e[0], e[1]))[:40]

    # Draw edges with pulsing effect
    for i, j, weight in edges_to_draw:
        x1, y1 = positions[i]
        x2, y2 = positions[j]

        # Normalized weight for scaling
        norm_weight = weight / max_weight

        # Base style
        base_alpha = 0.15
        base_lw = 0.5

        # Pulse-enhanced style
        pulse_alpha = base_alpha + 0.5 * pulse_factor * norm_weight
        pulse_lw = base_lw + 2.5 * norm_weight * (0.5 + 0.5 * pulse_factor)

        # Draw pulsing edge
        ax.plot(
            [x1, x2],
            [y1, y2],
            color="#2196F3",  # Blue color for communication
            alpha=min(pulse_alpha, 0.9),
            linewidth=pulse_lw,
            solid_capstyle="round",
        )

    # Get loss values for node coloring
    losses = []
    for node_id in range(n_nodes):
        metrics = node_metrics.get(str(node_id), {})
        loss = metrics.get("loss", 0.0)
        losses.append(loss)

    # Normalize for coloring
    if losses and max(losses) > min(losses):
        loss_min, loss_max = min(losses), max(losses)
        colors = [(loss - loss_min) / (loss_max - loss_min) for loss in losses]
    else:
        colors = [0.5] * n_nodes

    # Draw nodes
    for node_id in range(n_nodes):
        x, y = positions[node_id]
        metrics = node_metrics.get(str(node_id), {})
        loss = metrics.get("loss", 0.0)
        acc = metrics.get("accuracy")

        # Color based on loss (red = high, green = low)
        color = plt.cm.RdYlGn(1 - colors[node_id])  # type: ignore[attr-defined]

        circle = Circle((x, y), 0.15, color=color, ec="black", linewidth=1.5)
        ax.add_patch(circle)

        # Label
        label = f"{node_id}\nL={loss:.2f}"
        if acc is not None:
            label += f"\nA={acc:.2f}"
        ax.text(x, y, label, ha="center", va="center", fontsize=6)

    # Add edge count info
    n_edges = len(edges_with_weights)
    ax.text(
        0.02,
        0.02,
        f"edges: {n_edges}",
        transform=ax.transAxes,
        fontsize=7,
        color="gray",
    )

    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    ax.set_aspect("equal")
    ax.set_axis_off()


def _draw_table_panel(
    ax: Any,
    node_metrics: dict[str, dict[str, float]],
    task: str,
    comm_graph: dict[str, Any] | None = None,
) -> None:
    """Draw the table panel showing per-node metrics."""
    ax.set_title("Node Metrics", fontsize=10)
    ax.set_axis_off()

    if not node_metrics:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        return

    # Determine columns based on available metrics
    sample_metrics = next(iter(node_metrics.values()), {})
    columns = ["Node", "Loss"]
    if "accuracy" in sample_metrics:
        columns.append("Acc")
    if "dist_to_opt" in sample_metrics:
        columns.append("Dist")
    if "param_norm" in sample_metrics:
        columns.append("||x||")

    # Build table data
    rows = []
    for node_id in sorted(node_metrics.keys(), key=int):
        metrics = node_metrics[node_id]
        row = [node_id, f"{metrics.get('loss', 0):.3f}"]
        if "accuracy" in sample_metrics:
            row.append(f"{metrics.get('accuracy', 0):.3f}")
        if "dist_to_opt" in sample_metrics:
            row.append(f"{metrics.get('dist_to_opt', 0):.3f}")
        if "param_norm" in sample_metrics:
            row.append(f"{metrics.get('param_norm', 0):.3f}")
        rows.append(row)

    # Limit to 10 rows for readability
    if len(rows) > 10:
        rows = rows[:10]
        rows.append(["...", "...", *["..." for _ in range(len(columns) - 2)]])

    table = ax.table(
        cellText=rows,
        colLabels=columns,
        loc="center",
        cellLoc="center",
        colColours=["#f0f0f0"] * len(columns),
    )
    table.auto_set_font_size(False)
    table.set_fontsize(7)
    table.scale(1, 1.2)


def _draw_loss_curve(ax: Any, trace: list[dict[str, Any]], current_idx: int) -> None:
    """Draw the loss time series with current step marker."""
    steps = [r["t"] for r in trace]
    losses = [r.get("mean_metrics", {}).get("mean_loss", 0) for r in trace]

    ax.plot(steps, losses, "b-", linewidth=1.5, label="Mean Loss")
    ax.axvline(x=trace[current_idx]["t"], color="r", linestyle="--", linewidth=1, alpha=0.7)

    # Mark current point
    current_loss = losses[current_idx]
    ax.plot(trace[current_idx]["t"], current_loss, "ro", markersize=6)

    ax.set_xlabel("Step", fontsize=8)
    ax.set_ylabel("Mean Loss", fontsize=8)
    ax.set_title("Loss vs Step", fontsize=10)
    ax.tick_params(axis="both", labelsize=7)
    ax.grid(True, alpha=0.3)


def _draw_accuracy_curve(ax: Any, trace: list[dict[str, Any]], current_idx: int) -> None:
    """Draw the accuracy time series with current step marker."""
    steps = [r["t"] for r in trace]
    accs = [r.get("mean_metrics", {}).get("mean_accuracy", 0) for r in trace]

    ax.plot(steps, accs, "g-", linewidth=1.5, label="Mean Accuracy")
    ax.axvline(x=trace[current_idx]["t"], color="r", linestyle="--", linewidth=1, alpha=0.7)

    current_acc = accs[current_idx]
    ax.plot(trace[current_idx]["t"], current_acc, "ro", markersize=6)

    ax.set_xlabel("Step", fontsize=8)
    ax.set_ylabel("Mean Accuracy", fontsize=8)
    ax.set_title("Accuracy vs Step", fontsize=10)
    ax.set_ylim(0, 1.05)
    ax.tick_params(axis="both", labelsize=7)
    ax.grid(True, alpha=0.3)


def _draw_suboptimality_curve(ax: Any, trace: list[dict[str, Any]], current_idx: int) -> None:
    """Draw the suboptimality time series with current step marker."""
    steps = [r["t"] for r in trace]
    subopt = [r.get("mean_metrics", {}).get("suboptimality", 0) for r in trace]

    ax.plot(steps, subopt, "m-", linewidth=1.5, label="Suboptimality")
    ax.axvline(x=trace[current_idx]["t"], color="r", linestyle="--", linewidth=1, alpha=0.7)

    current_subopt = subopt[current_idx]
    ax.plot(trace[current_idx]["t"], current_subopt, "ro", markersize=6)

    ax.set_xlabel("Step", fontsize=8)
    ax.set_ylabel("Suboptimality", fontsize=8)
    ax.set_title("Suboptimality vs Step", fontsize=10)
    if all(s > 0 for s in subopt if s):
        ax.set_yscale("log")
    ax.tick_params(axis="both", labelsize=7)
    ax.grid(True, alpha=0.3)


def _draw_consensus_curve(ax: Any, trace: list[dict[str, Any]], current_idx: int) -> None:
    """Draw the consensus error time series with current step marker."""
    steps = [r["t"] for r in trace]
    cons = [r.get("mean_metrics", {}).get("consensus_error", 0) for r in trace]

    ax.plot(steps, cons, "c-", linewidth=1.5, label="Consensus Error")
    ax.axvline(x=trace[current_idx]["t"], color="r", linestyle="--", linewidth=1, alpha=0.7)

    current_cons = cons[current_idx]
    ax.plot(trace[current_idx]["t"], current_cons, "ro", markersize=6)

    ax.set_xlabel("Step", fontsize=8)
    ax.set_ylabel("Consensus Error", fontsize=8)
    ax.set_title("Consensus Error vs Step", fontsize=10)
    if all(c > 0 for c in cons if c):
        ax.set_yscale("log")
    ax.tick_params(axis="both", labelsize=7)
    ax.grid(True, alpha=0.3)


# =============================================================================
# Animation Rendering
# =============================================================================


def render_animation(
    *,
    trace: list[dict[str, Any]],
    out_gif: Path,
    topology: str | None,
    fps: int,
    max_steps: int,
    title: str,
    config: dict[str, Any],
    comm_graph: dict[str, Any] | None = None,
    out_mp4: Path | None = None,
) -> None:
    """Render an animated GIF (and optionally MP4) from trace data.

    Args:
        trace: List of trace records (one per step).
        out_gif: Output path for the GIF file.
        topology: Topology type for graph layout.
        fps: Frames per second.
        max_steps: Maximum number of frames (subsample if needed).
        title: Animation title (used if no metadata strings built).
        config: Configuration dictionary for banner.
        comm_graph: Optional communication graph with edges and weights.
        out_mp4: Optional output path for MP4 file.
    """
    if not trace:
        return

    # Ensure output directory exists
    out_gif.parent.mkdir(parents=True, exist_ok=True)

    # Build metadata strings for title
    title_line1, title_line2, _ = build_metadata_strings(config)

    # Subsample if needed
    n_steps = len(trace)
    if n_steps > max_steps:
        indices_arr = np.linspace(0, n_steps - 1, max_steps, dtype=int)
        # Ensure last step is included
        indices_list = list(indices_arr)
        if indices_list[-1] != n_steps - 1:
            indices_list[-1] = n_steps - 1
    else:
        indices_list = list(range(n_steps))

    frames: list[Any] = []

    for idx in indices_list:
        fig = plt.figure(figsize=(12, 8), dpi=100)
        gs = GridSpec(2, 2, figure=fig, height_ratios=[1, 1], width_ratios=[1, 1])

        _render_frame(fig, gs, trace, idx, topology, config, title_line1, title_line2, comm_graph)

        # Convert figure to RGB array using io buffer
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=100, bbox_inches="tight", pad_inches=0.1)
        buf.seek(0)
        frame = imageio.imread(buf)
        frames.append(frame)
        buf.close()
        plt.close(fig)

    # Write GIF (duration is in ms, so 1000/fps)
    duration_ms = int(1000 / fps) if fps > 0 else 125
    imageio.mimsave(str(out_gif), frames, duration=duration_ms, loop=0)

    # Write MP4 if requested
    if out_mp4 is not None:
        out_mp4.parent.mkdir(parents=True, exist_ok=True)
        try:
            # Use imageio-ffmpeg for MP4 if available
            imageio.mimsave(str(out_mp4), frames, fps=fps, codec="libx264", quality=8)
        except Exception:
            # Fall back to GIF-only if ffmpeg not available
            pass
