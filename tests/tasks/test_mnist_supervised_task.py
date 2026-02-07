from __future__ import annotations

from pathlib import Path

import pytest


torch = pytest.importorskip("torch")
torchvision = pytest.importorskip("torchvision")


def test_mnist_supervised_task_split_report_clamps() -> None:
    from tasks.mnist import MNISTSupervisedTask

    task = MNISTSupervisedTask(
        use_fake_data=True,
        fake_train_size=20,
        fake_test_size=5,
        train_size=10,
        val_size=10,
        test_size=3,
        val_from="test",
        test_from="test",
        batch_size=4,
    )

    assert task.split_report["train"] == 10
    assert task.split_report["test"] == 3
    # test set has 5 samples, so val is clamped to 2
    assert task.split_report["val"] == 2


def test_mnist_supervised_task_basic_lengths() -> None:
    from tasks.mnist import MNISTSupervisedTask

    task = MNISTSupervisedTask(
        use_fake_data=True,
        fake_train_size=30,
        fake_test_size=10,
        train_size=12,
        val_size=4,
        test_size=3,
        val_from="test",
        test_from="test",
        batch_size=4,
    )

    assert len(task.train_loader.dataset) == 12
    assert len(task.val_loader.dataset) == 4
    assert len(task.test_loader.dataset) == 3


def test_mnist_supervised_task_device_auto() -> None:
    from tasks.mnist import MNISTSupervisedTask

    task = MNISTSupervisedTask(
        use_fake_data=True,
        fake_train_size=10,
        fake_test_size=5,
        train_size=6,
        val_size=2,
        test_size=2,
        val_from="test",
        test_from="test",
        device="auto",
        batch_size=2,
    )
    assert task.device in {"cpu", "cuda"}


def test_mnist_supervised_task_defaults_for_fake_sizes(monkeypatch: pytest.MonkeyPatch) -> None:
    from tasks import mnist as mnist_module

    class DummyDataset(torch.utils.data.Dataset):
        def __len__(self) -> int:
            return 3

        def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
            return torch.zeros(1, 28, 28), 0

    dummy = DummyDataset()
    monkeypatch.setattr(mnist_module, "load_mnist_or_fake", lambda **kwargs: dummy)

    task = mnist_module.MNISTSupervisedTask(use_fake_data=True)
    assert len(task.train_loader.dataset) == 3


def test_mnist_supervised_task_train_test_from_train() -> None:
    from tasks.mnist import MNISTSupervisedTask

    task = MNISTSupervisedTask(
        use_fake_data=True,
        fake_train_size=20,
        fake_test_size=5,
        train_size=10,
        val_size=0,
        test_size=5,
        val_from="train",
        test_from="train",
        batch_size=2,
    )
    assert task.split_report["test"] == 5


def test_mnist_supervised_task_empty_val_test() -> None:
    from tasks.mnist import MNISTSupervisedTask

    task = MNISTSupervisedTask(
        use_fake_data=True,
        fake_train_size=10,
        fake_test_size=5,
        train_size=6,
        val_size=0,
        test_size=0,
        val_from="train",
        test_from="test",
        batch_size=2,
    )
    assert len(task.val_loader.dataset) == 0
    assert len(task.test_loader.dataset) == 0


def test_load_mnist_or_fake_real_path(monkeypatch: pytest.MonkeyPatch) -> None:
    from tasks import mnist as mnist_module

    class DummyDataset(torch.utils.data.Dataset):
        def __len__(self) -> int:
            return 5

        def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
            return torch.zeros(1, 28, 28), 0

    dummy = DummyDataset()
    monkeypatch.setattr(mnist_module, "load_mnist", lambda root, train, download: dummy)

    ds = mnist_module.load_mnist_or_fake(
        use_fake_data=False,
        n_samples=5,
        seed=0,
        root=Path("."),
        train=True,
        download=False,
    )
    assert ds is dummy


def test_load_mnist_or_fake_requires_root() -> None:
    from tasks.mnist import load_mnist_or_fake

    with pytest.raises(ValueError):
        load_mnist_or_fake(use_fake_data=False, n_samples=1, seed=0, root=None, train=True, download=False)


def test_mnist_supervised_task_real_path(monkeypatch: pytest.MonkeyPatch) -> None:
    from tasks import mnist as mnist_module

    class DummyDataset(torch.utils.data.Dataset):
        def __len__(self) -> int:
            return 10

        def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
            return torch.zeros(1, 28, 28), 0

    dummy = DummyDataset()
    monkeypatch.setattr(mnist_module, "load_mnist", lambda root, train, download: dummy)

    task = mnist_module.MNISTSupervisedTask(
        data_root=".data",
        download=False,
        use_fake_data=False,
        train_size=5,
        val_size=0,
        test_size=2,
        val_from="train",
        test_from="test",
        batch_size=2,
    )
    assert len(task.train_loader.dataset) == 5


def test_mnist_supervised_task_invalid_sources() -> None:
    from tasks.mnist import MNISTSupervisedTask

    with pytest.raises(ValueError):
        MNISTSupervisedTask(use_fake_data=True, fake_train_size=10, fake_test_size=5, val_from="bad")

    with pytest.raises(ValueError):
        MNISTSupervisedTask(use_fake_data=True, fake_train_size=10, fake_test_size=5, test_from="bad")


def test_mnist_supervised_task_fixture_metrics() -> None:
    from tasks.mnist import MNISTSupervisedTask

    task = MNISTSupervisedTask(
        mnist_fixture="tests/assets/mnist_tiny",
        train_size=4,
        val_size=2,
        test_size=2,
        val_from="train",
        test_from="test",
        batch_size=2,
    )
    batch = next(iter(task.train_loader))
    images, labels = batch
    logits = torch.randn(images.size(0), 10)
    metrics = task.metrics_fn(logits, labels)
    loss = task.loss_fn(logits, labels)
    assert "accuracy" in metrics
    assert torch.isfinite(loss)
