"""Vanilla gradient descent optimizer for torch models."""

from __future__ import annotations

from typing import Any, Callable

try:
    import torch

    TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover - torch optional at import time
    TORCH_AVAILABLE = False
    torch = None  # type: ignore[assignment]


def _check_torch() -> None:
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch is required for TorchGD.")


class TorchGD(torch.optim.Optimizer):  # type: ignore[misc]
    """Simple gradient descent (no momentum)."""

    def __init__(self, params: Any, *, lr: float = 0.1, weight_decay: float = 0.0) -> None:
        _check_torch()
        defaults = {"lr": lr, "weight_decay": weight_decay}
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure: Callable[[], Any] | None = None) -> Any:
        _check_torch()
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            weight_decay = group["weight_decay"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                if weight_decay != 0:
                    grad = grad.add(p, alpha=weight_decay)
                p.add_(grad, alpha=-lr)
        return loss
