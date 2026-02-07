"""Config loading and dynamic instantiation utilities."""

from __future__ import annotations

import importlib
import json
from pathlib import Path
from typing import Any


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def import_class(path: str) -> type[Any]:
    if ":" in path:
        module_name, class_name = path.split(":", 1)
    else:
        module_name, class_name = path.rsplit(".", 1)
    module = importlib.import_module(module_name)
    return getattr(module, class_name)


def resolve_spec(spec: Any, **extra_kwargs: Any) -> Any:
    """Instantiate an object from a {class, params} spec or return spec as-is."""
    if isinstance(spec, dict) and "class" in spec:
        cls = import_class(spec["class"])
        params = spec.get("params", {})
        resolved = resolve_values(params)
        resolved.update(extra_kwargs)
        return cls(**resolved)
    return resolve_values(spec)


def resolve_values(value: Any, *, skip_keys: set[str] | None = None) -> Any:
    if isinstance(value, dict):
        if "class" in value:
            return resolve_spec(value)
        resolved: dict[str, Any] = {}
        for key, val in value.items():
            if skip_keys and key in skip_keys:
                resolved[key] = val
            else:
                resolved[key] = resolve_values(val, skip_keys=skip_keys)
        return resolved
    if isinstance(value, list):
        return [resolve_values(v, skip_keys=skip_keys) for v in value]
    return value


def _parse_value(raw: str) -> Any:
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return raw


def apply_overrides(config: dict[str, Any], overrides: list[str]) -> dict[str, Any]:
    result = json.loads(json.dumps(config))
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"Override must be key=value, got: {item}")
        path, raw_val = item.split("=", 1)
        keys = path.split(".")
        target = result
        for key in keys[:-1]:
            if key not in target or not isinstance(target[key], dict):
                target[key] = {}
            target = target[key]
        target[keys[-1]] = _parse_value(raw_val)
    return result


def load_task_dir(task_dir: Path) -> dict[str, Any]:
    task_spec = load_json(task_dir / "task.json")
    models_spec = load_json(task_dir / "models.json")
    optimizers_spec = load_json(task_dir / "optimizers.json")
    trainer_spec_path = task_dir / "trainer.json"
    trainer_spec = load_json(trainer_spec_path) if trainer_spec_path.exists() else None
    return {
        "name": task_dir.name,
        "task": task_spec,
        "models": models_spec,
        "optimizers": optimizers_spec,
        "trainer": trainer_spec,
    }
