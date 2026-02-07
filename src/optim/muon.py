"""Muon optimizer (SGD with momentum-style update)."""

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
        raise ImportError("PyTorch is required for Muon optimizer.")


class Muon(torch.optim.Optimizer):  # type: ignore[misc]
    """Momentum optimizer used as a placeholder for Muon."""

    def __init__(
        self,
        params: Any,
        *,
        lr: float = 0.01,
        beta: float = 0.9,
        weight_decay: float = 0.0,
    ) -> None:
        _check_torch()
        defaults = {"lr": lr, "beta": beta, "weight_decay": weight_decay}
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure: Callable[[], Any] | None = None) -> Any:
        _check_torch()
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            beta = group["beta"]
            weight_decay = group["weight_decay"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                if weight_decay != 0:
                    grad = grad.add(p, alpha=weight_decay)
                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(p)
                buf = state["momentum_buffer"]
                buf.mul_(beta).add_(grad)
                p.add_(buf, alpha=-lr)
        return loss
