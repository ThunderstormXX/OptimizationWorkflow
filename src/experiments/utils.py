"""Utilities for experiments."""

from __future__ import annotations

from typing import Any

try:
    import torch
    import torch.nn as nn

    TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover - torch optional at import time
    TORCH_AVAILABLE = False
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]


def _check_torch() -> None:
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch is required for torch-based experiments.")


def is_torch_optimizer(obj: Any) -> bool:
    """Return True if obj is a torch optimizer instance."""
    if not TORCH_AVAILABLE:
        return False
    return isinstance(obj, torch.optim.Optimizer)


def is_torch_optimizer_class(cls: type[Any]) -> bool:
    """Return True if cls is a torch optimizer class."""
    if not TORCH_AVAILABLE:
        return False
    return issubclass(cls, torch.optim.Optimizer)


def apply_model_init(model: Any, init_spec: Any, *, seed: int | None = None) -> None:
    """Apply weight initialization to a torch model.

    Args:
        model: torch.nn.Module instance.
        init_spec: Dict or string describing initialization.
        seed: Optional seed for deterministic init.
    """
    if init_spec is None:
        return
    _check_torch()

    if seed is not None:
        torch.manual_seed(seed)

    if isinstance(init_spec, str):
        init_type = init_spec
        params: dict[str, Any] = {}
    elif isinstance(init_spec, dict):
        init_type = str(init_spec.get("type", ""))
        params = {k: v for k, v in init_spec.items() if k != "type"}
    else:
        raise TypeError("init_spec must be a dict or string")

    init_type = init_type.lower()

    def init_tensor(tensor: torch.Tensor) -> None:
        if init_type in {"zeros", "zero"}:
            nn.init.zeros_(tensor)
        elif init_type in {"ones", "one"}:
            nn.init.ones_(tensor)
        elif init_type in {"constant", "const"}:
            value = float(params.get("value", 0.0))
            nn.init.constant_(tensor, value)
        elif init_type in {"normal", "gaussian"}:
            mean = float(params.get("mean", 0.0))
            std = float(params.get("std", 0.02))
            nn.init.normal_(tensor, mean=mean, std=std)
        elif init_type in {"uniform"}:
            a = float(params.get("a", -0.1))
            b = float(params.get("b", 0.1))
            nn.init.uniform_(tensor, a=a, b=b)
        elif init_type in {"xavier_uniform", "xavier_uniform_"}:
            nn.init.xavier_uniform_(tensor)
        elif init_type in {"xavier_normal", "xavier_normal_"}:
            nn.init.xavier_normal_(tensor)
        elif init_type in {"kaiming_uniform", "kaiming_uniform_"}:
            a = float(params.get("a", 0))
            nonlinearity = str(params.get("nonlinearity", "relu"))
            nn.init.kaiming_uniform_(tensor, a=a, nonlinearity=nonlinearity)
        elif init_type in {"kaiming_normal", "kaiming_normal_"}:
            a = float(params.get("a", 0))
            nonlinearity = str(params.get("nonlinearity", "relu"))
            nn.init.kaiming_normal_(tensor, a=a, nonlinearity=nonlinearity)
        else:
            raise ValueError(f"Unknown init type: {init_type}")

    for param in model.parameters():
        if param.dim() == 1 and init_type not in {"normal", "uniform", "constant"}:
            nn.init.zeros_(param)
        else:
            init_tensor(param)
