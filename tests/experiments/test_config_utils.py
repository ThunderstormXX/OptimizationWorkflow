from __future__ import annotations

import pytest

from experiments.config import apply_overrides, import_class, load_task_dir, resolve_spec, resolve_values


def test_resolve_spec_instantiates_class() -> None:
    spec = {"class": "optim.legacy_frankwolfe.L2BallConstraint", "params": {"radius": 1.5}}
    obj = resolve_spec(spec)
    assert obj.radius == 1.5
    assert resolve_spec(5) == 5


def test_resolve_values_skip_keys() -> None:
    payload = {
        "constraint": {"class": "optim.legacy_frankwolfe.L2BallConstraint", "params": {"radius": "auto"}},
        "lr": 0.1,
    }
    resolved = resolve_values(payload, skip_keys={"constraint"})
    assert isinstance(resolved["constraint"], dict)
    assert resolved["lr"] == 0.1
    assert resolve_values([1, {"a": 2}]) == [1, {"a": 2}]
    obj = resolve_values({"class": "optim.legacy_frankwolfe.L2BallConstraint", "params": {"radius": 2.0}})
    assert obj.radius == 2.0


def test_apply_overrides_updates_nested() -> None:
    config = {"a": {"b": 1}, "c": 2}
    updated = apply_overrides(config, ["a.b=3", "d=4"])
    assert updated["a"]["b"] == 3
    assert updated["d"] == 4
    updated = apply_overrides(config, ["name=hello"])
    assert updated["name"] == "hello"
    updated = apply_overrides({}, ["x.y=1"])
    assert updated["x"]["y"] == 1


def test_import_class_with_colon_path() -> None:
    cls = import_class("tests.experiments.test_config_utils:Dummy")
    assert cls is Dummy


def test_resolve_spec_extra_kwargs() -> None:
    spec = {"class": "tests.experiments.test_config_utils:Dummy", "params": {"value": 1}}
    obj = resolve_spec(spec, extra=2)
    assert obj.value == 1
    assert obj.extra == 2


def test_apply_overrides_invalid() -> None:
    with pytest.raises(ValueError):
        apply_overrides({"a": 1}, ["invalid"])


def test_load_task_dir(tmp_path) -> None:
    (tmp_path / "task.json").write_text("{\"class\": \"tests.experiments.test_config_utils:Dummy\", \"params\": {}}")
    (tmp_path / "models.json").write_text("[]")
    (tmp_path / "optimizers.json").write_text("[]")
    (tmp_path / "trainer.json").write_text("{\"class\": \"tests.experiments.test_config_utils:Dummy\", \"params\": {}}")
    bundle = load_task_dir(tmp_path)
    assert bundle["name"] == tmp_path.name
    assert bundle["trainer"]["class"].endswith("Dummy")


class Dummy:
    def __init__(self, value: int = 0, extra: int = 0) -> None:
        self.value = value
        self.extra = extra
