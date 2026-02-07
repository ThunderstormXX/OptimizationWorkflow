from __future__ import annotations

from pathlib import Path

from benchmarks.plotting import group_mean_std, plot_matrix_results, plot_single_run


def test_plot_matrix_results_all_branches(tmp_path: Path) -> None:
    rows = [
        {
            "env": "gossip",
            "model": "mlp3",
            "optimizer": "adam",
            "constraint": "l2ball",
            "radius": "10",
            "topology": "ring",
            "strategy": "local_then_gossip",
            "final_mean_loss": "0.5",
            "final_mean_accuracy": "0.95",
            "final_suboptimality": "0.1",
            "final_consensus_error": "0.05",
            "heterogeneity": "iid",
        },
        {
            "env": "gossip",
            "model": "cnn",
            "optimizer": "fw",
            "constraint": "l1ball",
            "radius": "5",
            "topology": "complete",
            "strategy": "gossip_then_local",
            "final_mean_loss": "0.4",
            "final_mean_accuracy": "0.97",
            "final_suboptimality": "0.08",
            "final_consensus_error": "0.02",
            "heterogeneity": "label_skew",
        },
    ]

    plots = plot_matrix_results(rows, tmp_path, task="mnist")

    expected = {
        "bar_final_loss_by_optimizer",
        "bar_final_loss_by_model",
        "bar_final_loss_by_model_optimizer",
        "bar_final_accuracy_by_optimizer",
        "bar_final_accuracy_by_model",
        "bar_final_accuracy_by_model_optimizer",
        "bar_final_consensus_by_strategy",
        "bar_final_loss_by_topology",
        "bar_final_loss_by_optimizer_strategy",
        "bar_final_accuracy_by_topology_strategy",
        "bar_final_accuracy_by_model_constraint",
        "bar_final_accuracy_by_constraint",
        "bar_final_accuracy_by_heterogeneity",
    }

    assert expected.issubset(set(plots.keys()))


def test_plot_matrix_results_empty_rows(tmp_path: Path) -> None:
    plots = plot_matrix_results([], tmp_path, task="quadratic")
    assert plots == {}


def test_plot_single_run_empty_history(tmp_path: Path) -> None:
    plots = plot_single_run([], tmp_path, task="quadratic", env="single")
    assert plots == {}


def test_group_mean_std_skips_bad_values() -> None:
    rows = [
        {"optimizer": "fw", "final_mean_loss": "bad"},
        {"optimizer": "fw", "final_mean_loss": "1.0"},
    ]
    stats = group_mean_std(rows, ["optimizer"], "final_mean_loss")
    assert stats[0][1] == 1.0
