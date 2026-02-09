"""Supervised training loop for torch models and tasks."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import json

try:
    import torch
    from torch.utils.data import DataLoader

    TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover - torch optional at import time
    TORCH_AVAILABLE = False
    torch = None  # type: ignore[assignment]
    DataLoader = object  # type: ignore[assignment]

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover - tqdm optional
    tqdm = None  # type: ignore[assignment]


def _check_torch() -> None:
    global TORCH_AVAILABLE, torch, DataLoader
    if TORCH_AVAILABLE:
        return
    try:  # pragma: no cover - optional at runtime
        import torch as torch_module
        from torch.utils.data import DataLoader as TorchDataLoader

        torch = torch_module  # type: ignore[assignment]
        DataLoader = TorchDataLoader  # type: ignore[assignment]
        TORCH_AVAILABLE = True
    except ImportError as exc:  # pragma: no cover
        raise ImportError("PyTorch is required for SupervisedTrainer.") from exc


def _log(msg: str, *, use_tqdm: bool) -> None:
    if use_tqdm and tqdm is not None:
        tqdm.write(f"[Trainer] {msg}")
    else:
        print(f"[Trainer] {msg}", flush=True)


def _get_loader(task: Any, name: str) -> Any | None:
    attr = f"{name}_loader"
    loader = getattr(task, attr, None)
    if loader is not None:
        return loader
    getter = getattr(task, f"get_{name}_loader", None)
    if callable(getter):
        return getter()
    return None


def _batch_size(batch: Any) -> int:
    if isinstance(batch, (tuple, list)) and batch:
        first = batch[0]
    else:
        first = batch
    if hasattr(first, "size"):
        return int(first.size(0))
    return 1


def _move_batch(batch: Any, device: str) -> Any:
    if isinstance(batch, (tuple, list)):
        return tuple(x.to(device) if hasattr(x, "to") else x for x in batch)
    if hasattr(batch, "to"):
        return batch.to(device)
    return batch


def _default_metrics(logits: Any, labels: Any) -> dict[str, float]:
    if not TORCH_AVAILABLE:
        return {}
    if not isinstance(logits, torch.Tensor) or not isinstance(labels, torch.Tensor):
        return {}
    if logits.ndim < 2:
        return {}
    if labels.ndim != 1:
        return {}
    preds = logits.argmax(dim=1)
    accuracy = (preds == labels).float().mean().item()
    return {"accuracy": float(accuracy)}


def _metrics_for(task: Any, logits: Any, labels: Any) -> dict[str, float]:
    metrics_fn = getattr(task, "metrics_fn", None)
    if callable(metrics_fn):
        return metrics_fn(logits, labels)
    return _default_metrics(logits, labels)


def _loss_for(task: Any, logits: Any, labels: Any) -> Any:
    loss_fn = getattr(task, "loss_fn", None)
    if callable(loss_fn):
        return loss_fn(logits, labels)
    raise AttributeError("Task must provide loss_fn(logits, labels)")


def _constraint_info_for(optimizer: Any) -> dict[str, float] | None:
    info_fn = getattr(optimizer, "constraint_info", None)
    if callable(info_fn):
        info = info_fn()
        if isinstance(info, dict) and info:
            return {k: float(v) for k, v in info.items() if isinstance(v, (int, float))}
    return None


@dataclass
class SupervisedTrainer:
    """Trainer for supervised learning on tasks with dataloaders."""

    epochs: int = 10
    device: str = "cpu"
    progress: bool = True
    val_every: int = 1
    test_every: int | None = None
    grad_clip: float | None = None
    full_loss_every: int = 100
    batch_log_every: int = 1
    batch_metrics_every: dict[str, int] | None = None
    full_metrics_every: dict[str, int] | None = None

    def train(
        self,
        *,
        model: Any,
        optimizer: Any,
        task: Any,
        seed: int,
        run_dir: Path,
    ) -> tuple[list[dict[str, float]], dict[str, Any]]:
        _check_torch()

        if seed is not None:
            torch.manual_seed(seed)

        device = self.device
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"

        if hasattr(model, "to"):
            model = model.to(device)

        train_loader = _get_loader(task, "train")
        val_loader = _get_loader(task, "val")
        test_loader = _get_loader(task, "test")
        if train_loader is None:
            raise AttributeError("Task must provide train_loader")

        progress_enabled = self.progress and tqdm is not None
        if self.progress and tqdm is None:
            _log("tqdm is not installed; progress bars disabled", use_tqdm=False)

        train_len = len(train_loader.dataset) if hasattr(train_loader, "dataset") else None
        val_len = (
            len(val_loader.dataset)
            if (val_loader is not None and hasattr(val_loader, "dataset"))
            else None
        )
        test_len = (
            len(test_loader.dataset)
            if (test_loader is not None and hasattr(test_loader, "dataset"))
            else None
        )
        _log(
            f"start epochs={self.epochs} device={device} "
            f"train={train_len} val={val_len} test={test_len}",
            use_tqdm=progress_enabled,
        )

        history: list[dict[str, float]] = []
        last_constraint_info: dict[str, float] | None = None

        epoch_bar = None
        if progress_enabled:
            epoch_bar = tqdm(
                total=self.epochs,
                position=0,
                leave=True,
                dynamic_ncols=True,
                bar_format="{desc} {bar} {elapsed}<{remaining}",
            )

        global_step = 0
        full_train_loader = self._make_eval_loader(train_loader)

        for epoch in range(self.epochs):
            if epoch_bar is not None:
                epoch_bar.set_description(f"{epoch + 1}/{self.epochs} epoch")
            train_metrics = self._run_train_epoch(
                model,
                optimizer,
                train_loader,
                task,
                device,
                epoch,
                progress_enabled,
                run_dir,
                full_train_loader,
                global_step,
            )
            global_step = int(train_metrics.pop("_global_step", global_step))
            last_constraint_info = train_metrics.pop("_constraint_info", None) or last_constraint_info
            row: dict[str, float] = {
                "epoch": float(epoch),
                "train_loss": train_metrics["loss"],
            }
            if "accuracy" in train_metrics:
                row["train_accuracy"] = train_metrics["accuracy"]

            if val_loader is not None and self.val_every > 0 and epoch % self.val_every == 0:
                val_metrics = self._run_eval(model, val_loader, task, device)
                row["val_loss"] = val_metrics["loss"]
                if "accuracy" in val_metrics:
                    row["val_accuracy"] = val_metrics["accuracy"]

            if (
                test_loader is not None
                and self.test_every is not None
                and self.test_every > 0
                and epoch % self.test_every == 0
            ):
                test_metrics = self._run_eval(model, test_loader, task, device)
                row["test_loss"] = test_metrics["loss"]
                if "accuracy" in test_metrics:
                    row["test_accuracy"] = test_metrics["accuracy"]

            if last_constraint_info:
                row.update(last_constraint_info)

            history.append(row)
            self._append_history(run_dir / "history.jsonl", row)

            if epoch_bar is not None:
                epoch_bar.update(1)

        if epoch_bar is not None:
            epoch_bar.close()

        summary = {
            "epochs": self.epochs,
            "device": device,
            "final": history[-1] if history else {},
        }
        if last_constraint_info:
            summary["constraint"] = last_constraint_info

        if test_loader is not None and self.test_every is None:
            test_metrics = self._run_eval(model, test_loader, task, device)
            summary["test"] = test_metrics

        self._write_summary(run_dir / "summary.json", summary)
        return history, summary

    def _run_train_epoch(
        self,
        model: Any,
        optimizer: Any,
        loader: DataLoader,
        task: Any,
        device: str,
        epoch: int,
        progress_enabled: bool,
        run_dir: Path,
        full_train_loader: DataLoader | None,
        global_step: int,
    ) -> dict[str, float]:
        model.train()

        total_loss = 0.0
        total_correct = 0.0
        total_seen = 0
        last_constraint_info: dict[str, float] | None = None

        pbar = None
        batch_iter: Iterable[Any] = loader
        if progress_enabled:
            total_iters = None
            try:
                total_iters = len(loader)  # type: ignore[arg-type]
            except TypeError:
                total_iters = None
            pbar = tqdm(
                total=total_iters,
                desc="",
                position=1,
                leave=False,
                dynamic_ncols=True,
                unit="iter",
                bar_format="{n_fmt}/{total_fmt} iteration {bar} {elapsed}<{remaining}",
            )

        batch_log_every = max(int(self.batch_log_every), 1) if self.batch_log_every else 0
        full_loss_every = max(int(self.full_loss_every), 1) if self.full_loss_every else 0

        batch_metrics_every = self.batch_metrics_every or {}
        full_metrics_every = self.full_metrics_every or {}

        for batch in batch_iter:
            batch = _move_batch(batch, device)
            inputs, labels = batch  # type: ignore[misc]
            optimizer.zero_grad(set_to_none=True) if hasattr(optimizer, "zero_grad") else None

            logits = model(inputs)
            loss = _loss_for(task, logits, labels)
            loss.backward()

            if self.grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), self.grad_clip)

            optimizer.step()
            info = _constraint_info_for(optimizer)
            if info:
                last_constraint_info = info

            batch_size = _batch_size(batch)
            total_loss += float(loss.item()) * batch_size
            total_seen += batch_size

            metrics = _metrics_for(task, logits, labels)
            if "accuracy" in metrics:
                total_correct += metrics["accuracy"] * batch_size

            global_step += 1
            step_row: dict[str, float] = {"iter": float(global_step)}

            if batch_log_every and global_step % batch_log_every == 0:
                step_row["batch_loss"] = float(loss.item())

            for name, every in batch_metrics_every.items():
                if every and global_step % int(every) == 0 and name in metrics:
                    step_row[f"batch_{name}"] = float(metrics[name])

            need_full_eval = False
            if full_train_loader is not None:
                if full_loss_every and global_step % full_loss_every == 0:
                    need_full_eval = True
                for every in full_metrics_every.values():
                    if every and global_step % int(every) == 0:
                        need_full_eval = True
                        break

            if need_full_eval and full_train_loader is not None:
                was_training = getattr(model, "training", None)
                full_metrics = self._run_eval(model, full_train_loader, task, device)
                if was_training is True:
                    model.train()
                elif was_training is False:
                    model.eval()

                if full_loss_every and global_step % full_loss_every == 0:
                    step_row["loss"] = full_metrics["loss"]

                for name, every in full_metrics_every.items():
                    if every and global_step % int(every) == 0 and name in full_metrics:
                        step_row[name] = float(full_metrics[name])

            if last_constraint_info and len(step_row) > 1:
                step_row.update(last_constraint_info)

            if len(step_row) > 1:
                self._append_history(run_dir / "steps.jsonl", step_row)

            if pbar is not None:
                pbar.update(1)

        if pbar is not None:
            pbar.close()

        avg_loss = total_loss / max(total_seen, 1)
        result = {"loss": avg_loss}
        if total_seen > 0:
            result["accuracy"] = total_correct / total_seen
        result["_global_step"] = float(global_step)
        if last_constraint_info:
            result["_constraint_info"] = last_constraint_info
        return result

    @staticmethod
    def _make_eval_loader(loader: DataLoader) -> DataLoader | None:
        if not TORCH_AVAILABLE:
            return None
        dataset = getattr(loader, "dataset", None)
        if dataset is None:
            return None
        batch_size = getattr(loader, "batch_size", 32)
        num_workers = getattr(loader, "num_workers", 0)
        pin_memory = getattr(loader, "pin_memory", False)
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )

    def _run_eval(
        self,
        model: Any,
        loader: DataLoader,
        task: Any,
        device: str,
    ) -> dict[str, float]:
        model.eval()

        total_loss = 0.0
        total_correct = 0.0
        total_seen = 0

        with torch.no_grad():
            for batch in loader:
                batch = _move_batch(batch, device)
                inputs, labels = batch  # type: ignore[misc]
                logits = model(inputs)
                loss = _loss_for(task, logits, labels)

                batch_size = _batch_size(batch)
                total_loss += float(loss.item()) * batch_size
                total_seen += batch_size

                metrics = _metrics_for(task, logits, labels)
                if "accuracy" in metrics:
                    total_correct += metrics["accuracy"] * batch_size

        avg_loss = total_loss / max(total_seen, 1)
        result = {"loss": avg_loss}
        if total_seen > 0:
            result["accuracy"] = total_correct / total_seen
        return result

    @staticmethod
    def _append_history(path: Path, row: dict[str, float]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")

    @staticmethod
    def _write_summary(path: Path, summary: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, sort_keys=True)
