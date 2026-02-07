from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest


torch = pytest.importorskip("torch")


@dataclass
class TinyTask:
    train_loader: object
    val_loader: object
    test_loader: object

    def loss_fn(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.cross_entropy(logits, labels)

    def metrics_fn(self, logits: torch.Tensor, labels: torch.Tensor) -> dict[str, float]:
        preds = logits.argmax(dim=1)
        acc = (preds == labels).float().mean().item()
        return {"accuracy": float(acc)}


def test_supervised_trainer_runs(tmp_path: Path) -> None:
    from trainers.supervised import SupervisedTrainer
    from optim.torch_gd import TorchGD

    x = torch.randn(16, 4)
    y = torch.randint(0, 3, (16,))
    ds = torch.utils.data.TensorDataset(x, y)
    loader = torch.utils.data.DataLoader(ds, batch_size=4, shuffle=False)

    task = TinyTask(train_loader=loader, val_loader=loader, test_loader=loader)
    model = torch.nn.Linear(4, 3)
    optimizer = TorchGD(model.parameters(), lr=0.1)

    trainer = SupervisedTrainer(epochs=2, progress=False, test_every=None)
    history, summary = trainer.train(
        model=model,
        optimizer=optimizer,
        task=task,
        seed=0,
        run_dir=tmp_path,
    )

    assert len(history) == 2
    assert "train_loss" in history[0]
    assert "val_loss" in history[0]
    assert "test" in summary


class TinyTaskNoMetrics:
    def __init__(self, loader: object) -> None:
        self.train_loader = loader
        self.val_loader = loader
        self.test_loader = loader

    def loss_fn(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.cross_entropy(logits, labels)


def test_supervised_trainer_progress_and_default_metrics(tmp_path: Path) -> None:
    from trainers.supervised import SupervisedTrainer
    from optim.torch_gd import TorchGD

    x = torch.randn(8, 2)
    y = torch.randint(0, 2, (8,))
    ds = torch.utils.data.TensorDataset(x, y)
    loader = torch.utils.data.DataLoader(ds, batch_size=2, shuffle=False)

    task = TinyTaskNoMetrics(loader)
    model = torch.nn.Linear(2, 2)
    optimizer = TorchGD(model.parameters(), lr=0.1)

    trainer = SupervisedTrainer(epochs=1, progress=True, test_every=1)
    history, summary = trainer.train(
        model=model,
        optimizer=optimizer,
        task=task,
        seed=0,
        run_dir=tmp_path,
    )

    assert "train_accuracy" in history[0]
    assert "test_loss" in history[0]
    assert summary["final"]["epoch"] == 0.0


def test_supervised_trainer_requires_train_loader(tmp_path: Path) -> None:
    from trainers.supervised import SupervisedTrainer
    from optim.torch_gd import TorchGD

    model = torch.nn.Linear(2, 2)
    optimizer = TorchGD(model.parameters(), lr=0.1)

    class BadTask:
        pass

    trainer = SupervisedTrainer(epochs=1, progress=False)
    with pytest.raises(AttributeError):
        trainer.train(model=model, optimizer=optimizer, task=BadTask(), seed=0, run_dir=tmp_path)


def test_supervised_helpers_cover_branches() -> None:
    from trainers import supervised as sup

    class TaskWithGetter:
        def get_train_loader(self) -> str:
            return "loader"

    assert sup._get_loader(TaskWithGetter(), "train") == "loader"

    tensor = torch.zeros(3, 2)
    assert sup._batch_size(tensor) == 3
    assert sup._move_batch(tensor, "cpu").shape == tensor.shape

    assert sup._default_metrics(torch.randn(2), torch.randn(2)) == {}
    assert sup._default_metrics(torch.randn(2, 2), torch.randn(2, 1)) == {}

    with pytest.raises(AttributeError):
        sup._loss_for(object(), torch.randn(2, 2), torch.randint(0, 2, (2,)))


def test_supervised_trainer_grad_clip(tmp_path: Path) -> None:
    from trainers.supervised import SupervisedTrainer
    from optim.torch_gd import TorchGD

    x = torch.randn(8, 2)
    y = torch.randint(0, 2, (8,))
    ds = torch.utils.data.TensorDataset(x, y)
    loader = torch.utils.data.DataLoader(ds, batch_size=2, shuffle=False)

    task = TinyTask(train_loader=loader, val_loader=loader, test_loader=loader)
    model = torch.nn.Linear(2, 2)
    optimizer = TorchGD(model.parameters(), lr=0.1)

    trainer = SupervisedTrainer(epochs=1, progress=False, grad_clip=0.5)
    history, _ = trainer.train(
        model=model,
        optimizer=optimizer,
        task=task,
        seed=0,
        run_dir=tmp_path,
    )
    assert history
