"""Thin CLI runner for experiment configs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

from benchmarks.workflow import next_experiment_dir, write_run_files
from experiments.config import apply_overrides, load_json, resolve_spec


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run experiments from JSON config",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", type=str, required=True, help="Path to experiment JSON config")
    parser.add_argument(
        "--workflow-dir",
        type=str,
        default=None,
        help="Override workflow output directory",
    )
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        help="Override config values (dot.path=value)",
    )
    parser.add_argument(
        "--exp-id",
        type=int,
        default=None,
        help="Reuse specific experiment directory (exp_XXXX) instead of creating a new one",
    )
    parser.add_argument(
        "--no-skip-existing",
        action="store_true",
        help="Do not skip runs that already have history/summary files",
    )
    return parser.parse_args(argv)


def run_from_config(config: dict[str, Any]) -> dict[str, Any]:
    exp_spec = config.get("experiment")
    if not exp_spec:
        raise ValueError("Config must include 'experiment' section")

    exp_obj = resolve_spec(exp_spec)
    run_fn = getattr(exp_obj, "run", None)
    if not callable(run_fn):
        raise TypeError("Experiment object must provide a run() method")

    return run_fn()


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    config = load_json(Path(args.config))
    if args.set:
        config = apply_overrides(config, args.set)

    if args.workflow_dir:
        config.setdefault("experiment", {}).setdefault("params", {})[
            "workflow_dir"
        ] = args.workflow_dir
    if args.exp_id is not None:
        config.setdefault("experiment", {}).setdefault("params", {})["exp_id"] = args.exp_id
    if args.no_skip_existing:
        config.setdefault("experiment", {}).setdefault("params", {})[
            "skip_existing"
        ] = False

    summary = run_from_config(config)

    # Write minimal metadata for the experiment directory
    workflow_dir = Path(summary.get("workflow_dir", "workflow"))
    if "exp_dir" in summary:
        exp_dir = Path(summary["exp_dir"])
    else:
        exp_dir = next_experiment_dir(workflow_dir)

    meta = {"config": Path(args.config).as_posix()}
    write_run_files(exp_dir, meta=meta, config=config, readme_text=summary.get("readme", ""))

    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
