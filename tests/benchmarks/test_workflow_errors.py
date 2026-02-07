from __future__ import annotations

import subprocess

import benchmarks.workflow as workflow


def test_try_get_git_commit_handles_errors(monkeypatch) -> None:
    def _raise(*_args, **_kwargs):
        raise FileNotFoundError

    monkeypatch.setattr(subprocess, "run", _raise)
    assert workflow.try_get_git_commit() is None
