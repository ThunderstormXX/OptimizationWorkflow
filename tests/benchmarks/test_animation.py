"""Tests for animation functionality.

This module tests:
- Trace file reading and writing
- Animation rendering from trace data
- Runner integration with animation flags
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from benchmarks.animate import (
    build_metadata_strings,
    read_trace_jsonl,
    render_animation,
    write_trace_line,
)
from benchmarks.legacy_runner import main


class TestTraceIO:
    """Tests for trace file I/O."""

    def test_write_and_read_trace(self, tmp_path: Path) -> None:
        """Test writing and reading trace.jsonl."""
        trace_path = tmp_path / "trace.jsonl"

        # Write trace
        with trace_path.open("w", encoding="utf-8") as f:
            for t in range(3):
                write_trace_line(
                    f,
                    t=t,
                    env="gossip",
                    node_metrics={
                        "0": {"loss": 1.0 - t * 0.1, "accuracy": 0.5 + t * 0.1},
                        "1": {"loss": 1.2 - t * 0.1, "accuracy": 0.4 + t * 0.1},
                    },
                    mean_metrics={
                        "mean_loss": 1.1 - t * 0.1,
                        "mean_accuracy": 0.45 + t * 0.1,
                        "consensus_error": 0.1 - t * 0.02,
                    },
                    params_by_node={"0": [1.0, 2.0], "1": [1.1, 2.1]},
                )

        # Read trace
        trace = read_trace_jsonl(trace_path)

        assert len(trace) == 3
        assert trace[0]["t"] == 0
        assert trace[0]["env"] == "gossip"
        assert trace[0]["node_metrics"]["0"]["loss"] == 1.0
        assert trace[2]["t"] == 2
        assert trace[2]["mean_metrics"]["mean_loss"] == pytest.approx(0.9)

    def test_read_empty_trace(self, tmp_path: Path) -> None:
        """Test reading empty trace file."""
        trace_path = tmp_path / "trace.jsonl"
        trace_path.write_text("")

        trace = read_trace_jsonl(trace_path)
        assert trace == []


class TestRenderAnimation:
    """Tests for animation rendering."""

    def test_render_simple_animation_gif(self, tmp_path: Path) -> None:
        """Test rendering a simple GIF animation from synthetic trace."""
        # Create synthetic trace
        trace: list[dict[str, Any]] = []
        for t in range(5):
            trace.append(
                {
                    "t": t,
                    "env": "gossip",
                    "node_metrics": {
                        "0": {"loss": 1.0 - t * 0.1, "accuracy": 0.5 + t * 0.05},
                        "1": {"loss": 1.1 - t * 0.1, "accuracy": 0.45 + t * 0.05},
                        "2": {"loss": 0.9 - t * 0.1, "accuracy": 0.55 + t * 0.05},
                    },
                    "mean_metrics": {
                        "mean_loss": 1.0 - t * 0.1,
                        "mean_accuracy": 0.5 + t * 0.05,
                        "consensus_error": 0.1 - t * 0.01,
                    },
                }
            )

        out_path = tmp_path / "animation.gif"
        config: dict[str, Any] = {
            "env": "gossip",
            "task": "logistic",
            "optimizer": "gd",
            "topology": "ring",
            "strategy": "local_then_gossip",
            "seed": 0,
        }

        render_animation(
            trace=trace,
            out_path=out_path,
            topology="ring",
            fps=2,
            max_steps=10,
            title="Test Animation",
            config=config,
            output_format="gif",
        )

        assert out_path.exists()
        assert out_path.stat().st_size > 1000  # GIF should be at least 1KB

    def test_render_simple_animation_mp4(self, tmp_path: Path) -> None:
        """Test rendering a simple MP4 animation from synthetic trace."""
        from benchmarks.animate import check_ffmpeg_available

        if not check_ffmpeg_available():
            pytest.skip("ffmpeg not available")

        # Create synthetic trace
        trace: list[dict[str, Any]] = []
        for t in range(5):
            trace.append(
                {
                    "t": t,
                    "env": "gossip",
                    "node_metrics": {
                        "0": {"loss": 1.0 - t * 0.1, "accuracy": 0.5 + t * 0.05},
                        "1": {"loss": 1.1 - t * 0.1, "accuracy": 0.45 + t * 0.05},
                        "2": {"loss": 0.9 - t * 0.1, "accuracy": 0.55 + t * 0.05},
                    },
                    "mean_metrics": {
                        "mean_loss": 1.0 - t * 0.1,
                        "mean_accuracy": 0.5 + t * 0.05,
                        "consensus_error": 0.1 - t * 0.01,
                    },
                }
            )

        out_path = tmp_path / "animation.mp4"
        config: dict[str, Any] = {
            "env": "gossip",
            "task": "logistic",
            "optimizer": "gd",
            "topology": "ring",
            "strategy": "local_then_gossip",
            "seed": 0,
        }

        render_animation(
            trace=trace,
            out_path=out_path,
            topology="ring",
            fps=2,
            max_steps=10,
            title="Test Animation",
            config=config,
            output_format="mp4",
        )

        assert out_path.exists()
        assert out_path.stat().st_size > 10000  # MP4 should be larger

    def test_render_quadratic_animation(self, tmp_path: Path) -> None:
        """Test rendering animation for quadratic task with suboptimality."""
        trace: list[dict[str, Any]] = []
        for t in range(5):
            trace.append(
                {
                    "t": t,
                    "env": "single",
                    "node_metrics": {
                        "0": {"loss": 10.0 - t * 2.0, "dist_to_opt": 5.0 - t * 1.0},
                    },
                    "mean_metrics": {
                        "mean_loss": 10.0 - t * 2.0,
                        "suboptimality": 0.5 - t * 0.1,
                    },
                }
            )

        out_path = tmp_path / "quadratic.gif"
        config: dict[str, Any] = {
            "env": "single",
            "task": "quadratic",
            "optimizer": "fw",
            "seed": 0,
        }

        render_animation(
            trace=trace,
            out_path=out_path,
            topology=None,
            fps=2,
            max_steps=10,
            title="Quadratic Test",
            config=config,
            output_format="gif",
        )

        assert out_path.exists()
        assert out_path.stat().st_size > 1000

    def test_render_with_subsampling(self, tmp_path: Path) -> None:
        """Test that long traces are subsampled correctly."""
        # Create trace with 50 steps
        trace: list[dict[str, Any]] = []
        for t in range(50):
            trace.append(
                {
                    "t": t,
                    "env": "single",
                    "node_metrics": {"0": {"loss": 1.0 - t * 0.01}},
                    "mean_metrics": {"mean_loss": 1.0 - t * 0.01},
                }
            )

        out_path = tmp_path / "subsampled.gif"
        config: dict[str, Any] = {"env": "single", "task": "quadratic", "optimizer": "gd"}

        # Limit to 10 frames
        render_animation(
            trace=trace,
            out_path=out_path,
            topology=None,
            fps=2,
            max_steps=10,
            title="Subsampled",
            config=config,
            output_format="gif",
        )

        assert out_path.exists()
        assert out_path.stat().st_size > 500

    def test_empty_trace_no_crash(self, tmp_path: Path) -> None:
        """Test that empty trace doesn't crash."""
        out_path = tmp_path / "empty.gif"
        config: dict[str, Any] = {"env": "single", "task": "quadratic"}

        render_animation(
            trace=[],
            out_path=out_path,
            topology=None,
            fps=2,
            max_steps=10,
            title="Empty",
            config=config,
            output_format="gif",
        )

        # No file should be created for empty trace
        assert not out_path.exists()


class TestRunnerSingleModeAnimation:
    """Tests for runner single mode with animation."""

    def test_single_mode_creates_animation_gif(self, tmp_path: Path) -> None:
        """Test that single mode with --animate --animate-format gif creates animation."""
        workflow_dir = tmp_path / "workflow"
        workflow_dir.mkdir()

        exit_code = main(
            [
                "--mode",
                "single",
                "--env",
                "gossip",
                "--task",
                "logistic",
                "--steps",
                "5",
                "--n-nodes",
                "3",
                "--heterogeneity",
                "iid",
                "--optimizer",
                "gd",
                "--lr",
                "0.1",
                "--dim",
                "5",
                "--n-samples",
                "100",
                "--batch-size",
                "32",
                "--animate",
                "--animate-format",
                "gif",
                "--animate-fps",
                "2",
                "--workflow-dir",
                str(workflow_dir),
            ]
        )

        assert exit_code == 0

        # Find experiment directory
        exp_dirs = list(workflow_dir.glob("exp_*"))
        assert len(exp_dirs) == 1
        exp_dir = exp_dirs[0]

        # Check trace.jsonl exists
        trace_path = exp_dir / "artifacts" / "trace.jsonl"
        assert trace_path.exists()

        # Check trace has correct number of lines
        with trace_path.open() as f:
            lines = [line for line in f if line.strip()]
        assert len(lines) == 5  # 5 steps

        # Check animation.gif exists
        animation_path = exp_dir / "artifacts" / "animation.gif"
        assert animation_path.exists()
        assert animation_path.stat().st_size > 1000

        # Check report.md mentions animation
        report_path = exp_dir / "artifacts" / "report.md"
        assert report_path.exists()
        content = report_path.read_text()
        assert "animation.gif" in content.lower() or "Animation" in content

    def test_single_mode_creates_animation_mp4(self, tmp_path: Path) -> None:
        """Test that single mode with --animate creates MP4 animation when ffmpeg available."""
        from benchmarks.animate import check_ffmpeg_available

        if not check_ffmpeg_available():
            pytest.skip("ffmpeg not available")

        workflow_dir = tmp_path / "workflow"
        workflow_dir.mkdir()

        exit_code = main(
            [
                "--mode",
                "single",
                "--env",
                "gossip",
                "--task",
                "logistic",
                "--steps",
                "5",
                "--n-nodes",
                "3",
                "--heterogeneity",
                "iid",
                "--optimizer",
                "gd",
                "--lr",
                "0.1",
                "--dim",
                "5",
                "--n-samples",
                "100",
                "--batch-size",
                "32",
                "--animate",
                "--animate-format",
                "mp4",
                "--animate-fps",
                "2",
                "--workflow-dir",
                str(workflow_dir),
            ]
        )

        assert exit_code == 0

        # Find experiment directory
        exp_dirs = list(workflow_dir.glob("exp_*"))
        assert len(exp_dirs) == 1
        exp_dir = exp_dirs[0]

        # Check animation.mp4 exists
        animation_path = exp_dir / "artifacts" / "animation.mp4"
        assert animation_path.exists()
        assert animation_path.stat().st_size > 10000  # MP4 should be larger

        # Check report.md mentions animation
        report_path = exp_dir / "artifacts" / "report.md"
        assert report_path.exists()
        content = report_path.read_text()
        assert "animation.mp4" in content.lower() or "Animation" in content


class TestRunnerMatrixModeAnimation:
    """Tests for runner matrix mode with animation."""

    def test_matrix_mode_animates_top_k_gif(self, tmp_path: Path) -> None:
        """Test that matrix mode with --animate creates GIF animations for top-K runs."""
        workflow_dir = tmp_path / "workflow"
        workflow_dir.mkdir()

        exit_code = main(
            [
                "--mode",
                "matrix",
                "--matrix",
                "small",
                "--env",
                "gossip",
                "--task",
                "logistic",
                "--steps",
                "3",
                "--n-nodes",
                "2",
                "--heterogeneity",
                "iid",
                "--optimizer",
                "gd",
                "--lr",
                "0.1",
                "--dim",
                "3",
                "--n-samples",
                "50",
                "--batch-size",
                "16",
                "--seeds",
                "0",
                "--animate",
                "--animate-format",
                "gif",
                "--animate-top-k",
                "1",
                "--animate-fps",
                "2",
                "--workflow-dir",
                str(workflow_dir),
            ]
        )

        assert exit_code == 0

        # Find experiment directory
        exp_dirs = list(workflow_dir.glob("exp_*"))
        assert len(exp_dirs) == 1
        exp_dir = exp_dirs[0]

        # Check animations.md exists
        animations_md = exp_dir / "artifacts" / "animations.md"
        assert animations_md.exists()

        # Check that exactly 1 run has animation.gif
        runs_dir = exp_dir / "artifacts" / "runs"
        animated_runs = list(runs_dir.glob("*/animation.gif"))
        assert len(animated_runs) == 1

        # Check that the animated run also has trace.jsonl
        animated_run_dir = animated_runs[0].parent
        trace_path = animated_run_dir / "trace.jsonl"
        assert trace_path.exists()

    def test_matrix_mode_animates_top_k_mp4(self, tmp_path: Path) -> None:
        """Test that matrix mode with --animate creates MP4 animations when ffmpeg available."""
        from benchmarks.animate import check_ffmpeg_available

        if not check_ffmpeg_available():
            pytest.skip("ffmpeg not available")

        workflow_dir = tmp_path / "workflow"
        workflow_dir.mkdir()

        exit_code = main(
            [
                "--mode",
                "matrix",
                "--matrix",
                "small",
                "--env",
                "gossip",
                "--task",
                "logistic",
                "--steps",
                "3",
                "--n-nodes",
                "2",
                "--heterogeneity",
                "iid",
                "--optimizer",
                "gd",
                "--lr",
                "0.1",
                "--dim",
                "3",
                "--n-samples",
                "50",
                "--batch-size",
                "16",
                "--seeds",
                "0",
                "--animate",
                "--animate-format",
                "mp4",
                "--animate-top-k",
                "1",
                "--animate-fps",
                "2",
                "--workflow-dir",
                str(workflow_dir),
            ]
        )

        assert exit_code == 0

        # Find experiment directory
        exp_dirs = list(workflow_dir.glob("exp_*"))
        assert len(exp_dirs) == 1
        exp_dir = exp_dirs[0]

        # Check animations.md exists
        animations_md = exp_dir / "artifacts" / "animations.md"
        assert animations_md.exists()
        content = animations_md.read_text()
        assert "MP4" in content

        # Check that exactly 1 run has animation.mp4
        runs_dir = exp_dir / "artifacts" / "runs"
        animated_runs = list(runs_dir.glob("*/animation.mp4"))
        assert len(animated_runs) == 1


class TestTraceContent:
    """Tests for trace content correctness."""

    def test_trace_contains_required_fields(self, tmp_path: Path) -> None:
        """Test that trace lines contain all required fields."""
        workflow_dir = tmp_path / "workflow"
        workflow_dir.mkdir()

        exit_code = main(
            [
                "--mode",
                "single",
                "--env",
                "gossip",
                "--task",
                "logistic",
                "--steps",
                "3",
                "--n-nodes",
                "2",
                "--optimizer",
                "gd",
                "--lr",
                "0.1",
                "--dim",
                "3",
                "--n-samples",
                "50",
                "--batch-size",
                "16",
                "--animate",
                "--animate-format",
                "gif",
                "--workflow-dir",
                str(workflow_dir),
            ]
        )

        assert exit_code == 0

        exp_dir = list(workflow_dir.glob("exp_*"))[0]
        trace_path = exp_dir / "artifacts" / "trace.jsonl"

        trace = read_trace_jsonl(trace_path)
        assert len(trace) == 3

        for record in trace:
            # Required fields
            assert "t" in record
            assert "env" in record
            assert "node_metrics" in record
            assert "mean_metrics" in record

            # Node metrics structure
            assert "0" in record["node_metrics"]
            assert "1" in record["node_metrics"]
            assert "loss" in record["node_metrics"]["0"]

            # Mean metrics
            assert "mean_loss" in record["mean_metrics"]
            assert "consensus_error" in record["mean_metrics"]


class TestCommGraphAndEdgePulsing:
    """Tests for communication graph and edge pulsing features."""

    def test_render_with_comm_graph(self, tmp_path: Path) -> None:
        """Test rendering animation with comm_graph for edge visualization."""
        # Create synthetic trace with 3 nodes in a ring
        trace: list[dict[str, Any]] = []
        for t in range(5):
            trace.append(
                {
                    "t": t,
                    "env": "gossip",
                    "node_metrics": {
                        "0": {"loss": 1.0 - t * 0.1, "accuracy": 0.5 + t * 0.05},
                        "1": {"loss": 1.1 - t * 0.1, "accuracy": 0.45 + t * 0.05},
                        "2": {"loss": 0.9 - t * 0.1, "accuracy": 0.55 + t * 0.05},
                    },
                    "mean_metrics": {
                        "mean_loss": 1.0 - t * 0.1,
                        "mean_accuracy": 0.5 + t * 0.05,
                        "consensus_error": 0.1 - t * 0.01,
                    },
                }
            )

        # Ring topology: edges 0-1, 1-2, 2-0
        comm_graph: dict[str, Any] = {
            "edges": [[0, 1], [1, 2], [0, 2]],
            "weights": [0.4, 0.4, 0.4],
        }

        out_path = tmp_path / "comm_animation.gif"
        config: dict[str, Any] = {
            "env": "gossip",
            "task": "logistic",
            "optimizer": "gd",
            "topology": "ring",
            "strategy": "local_then_gossip",
            "seed": 0,
            "steps": 5,
        }

        render_animation(
            trace=trace,
            out_path=out_path,
            topology="ring",
            fps=2,
            max_steps=10,
            title="Comm Graph Test",
            config=config,
            comm_graph=comm_graph,
            output_format="gif",
        )

        assert out_path.exists()
        assert out_path.stat().st_size > 1000

    def test_metadata_strings_logistic(self) -> None:
        """Test metadata string generation for logistic task."""
        config: dict[str, Any] = {
            "task": "logistic",
            "env": "gossip",
            "topology": "ring",
            "strategy": "local_then_gossip",
            "optimizer": "gd",
            "constraint": "none",
            "seed": 42,
            "steps": 100,
        }

        title_line1, title_line2, filename_base = build_metadata_strings(config)

        # Title line 1: Dataset | Model | Optimizer | Constraint
        assert "SyntheticLogit" in title_line1
        assert "NumpyVector" in title_line1
        assert "GD" in title_line1  # Optimizer uppercase
        assert "none" in title_line1  # Constraint
        # Title line 2: env/topo/strat details
        assert "env=gossip" in title_line2
        assert "topo=ring" in title_line2
        assert "SyntheticLogit" in filename_base

    def test_metadata_strings_quadratic(self) -> None:
        """Test metadata string generation for quadratic task."""
        config: dict[str, Any] = {
            "task": "quadratic",
            "env": "single",
            "optimizer": "fw",
            "constraint": "l2",
            "radius": 1.5,
            "seed": 0,
            "steps": 50,
        }

        title_line1, title_line2, filename_base = build_metadata_strings(config)

        # Title line 1: Dataset | Model | Optimizer | Constraint
        assert "SyntheticQuadratic" in title_line1
        assert "FW" in title_line1  # Optimizer uppercase
        assert "L2(R=1.5)" in title_line1  # Constraint in line 1 now
        assert "SyntheticQuadratic" in filename_base


class TestCommunicatorEdgeWeights:
    """Tests for communicator edge_weights method."""

    def test_ring_edge_weights(self) -> None:
        """Test edge_weights for ring topology."""
        from distributed.communicator import SynchronousGossipCommunicator
        from distributed.topology import RingTopology

        topo = RingTopology(n=4)
        comm = SynchronousGossipCommunicator(topology=topo)

        edges = comm.edge_weights()

        # Ring with 4 nodes should have 4 edges: 0-1, 1-2, 2-3, 0-3
        assert len(edges) == 4

        # All edges should have positive weights
        for i, j, w in edges:
            assert i < j  # Undirected, i < j
            assert w > 0

    def test_complete_edge_weights(self) -> None:
        """Test edge_weights for complete topology."""
        from distributed.communicator import SynchronousGossipCommunicator
        from distributed.topology import CompleteTopology

        topo = CompleteTopology(n=4)
        comm = SynchronousGossipCommunicator(topology=topo)

        edges = comm.edge_weights()

        # Complete graph with 4 nodes should have 4*3/2 = 6 edges
        assert len(edges) == 6

        # All edges should have positive weights
        for i, j, w in edges:
            assert i < j
            assert w > 0
