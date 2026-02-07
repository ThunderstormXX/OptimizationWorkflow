from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from tasks import mnist as mnist_module


torch = pytest.importorskip("torch")
torchvision = pytest.importorskip("torchvision")


def test_check_torch_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(mnist_module, "TORCH_AVAILABLE", False)
    with pytest.raises(ImportError):
        mnist_module._check_torch()
    monkeypatch.setattr(mnist_module, "TORCH_AVAILABLE", True)


def test_load_mnist_requires_data(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError):
        mnist_module.load_mnist(tmp_path, train=True, download=False)


def test_load_mnist_or_fake_requires_root() -> None:
    with pytest.raises(ValueError):
        mnist_module.load_mnist_or_fake(use_fake_data=False, n_samples=10, seed=0, root=None)


def test_load_mnist_or_fake_fake_data() -> None:
    ds = mnist_module.load_mnist_or_fake(use_fake_data=True, n_samples=12, seed=1)
    assert len(ds) == 12


def test_split_mnist_across_nodes_label_skew() -> None:
    ds = mnist_module.load_mnist_or_fake(use_fake_data=True, n_samples=20, seed=0)
    subsets = mnist_module.split_mnist_across_nodes(
        ds, n_nodes=2, heterogeneity="label_skew", seed=0
    )
    assert len(subsets) == 2
    assert sum(len(s) for s in subsets) <= len(ds)


def test_mnist_tiny_dataset_missing_files(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        mnist_module.MNISTTinyDataset(tmp_path, train=True)


def test_mnist_tiny_dataset_missing_labels(tmp_path: Path) -> None:
    images_path = tmp_path / "train_images.npy"
    images = (np.zeros((2, 28, 28)) * 255).astype(np.uint8)
    np.save(images_path, images)
    with pytest.raises(FileNotFoundError):
        mnist_module.MNISTTinyDataset(tmp_path, train=True)


def test_mnist_task_compute_val_metrics_no_samples() -> None:
    train_ds = mnist_module.load_mnist_or_fake(use_fake_data=True, n_samples=8, seed=0)
    val_ds = mnist_module.load_mnist_or_fake(use_fake_data=True, n_samples=8, seed=1)

    task = mnist_module.MNISTClassificationTask(
        train_dataset=train_ds,
        val_dataset=val_ds,
        batch_size=4,
        eval_every=1,
        eval_batches=0,
    )

    from models.torch_nets import CNN
    from models.torch_adapter import TorchModelAdapter

    model = TorchModelAdapter(CNN(), device="cpu")
    loss, acc = task._compute_val_metrics(model)
    assert loss == 0.0
    assert acc == 0.0


def test_mnist_task_properties_and_batch_reset() -> None:
    train_ds = mnist_module.load_mnist_or_fake(use_fake_data=True, n_samples=6, seed=0)
    val_ds = mnist_module.load_mnist_or_fake(use_fake_data=True, n_samples=4, seed=1)
    task = mnist_module.MNISTClassificationTask(
        train_dataset=train_ds,
        val_dataset=val_ds,
        batch_size=2,
        eval_every=10,
        eval_batches=1,
    )
    assert task.n_train == 6
    assert task.n_val == 4

    # Trigger StopIteration branch by exhausting iterator
    for _ in range(5):
        images, labels = task.sample_batch(rng=np.random.default_rng(0))
        assert images.shape[0] == 2
        assert labels.shape[0] == 2


def test_torch_grad_computer_with_none_batch() -> None:
    train_ds = mnist_module.load_mnist_or_fake(use_fake_data=True, n_samples=8, seed=0)
    val_ds = mnist_module.load_mnist_or_fake(use_fake_data=True, n_samples=8, seed=1)

    task = mnist_module.MNISTClassificationTask(
        train_dataset=train_ds,
        val_dataset=val_ds,
        batch_size=4,
        eval_every=10,
        eval_batches=1,
    )

    from models.torch_nets import CNN
    from models.torch_adapter import TorchModelAdapter

    model = TorchModelAdapter(CNN(), device="cpu")
    grad_computer = mnist_module.TorchGradComputer()
    grad = grad_computer.grad(task, model, batch=None)

    assert isinstance(grad, np.ndarray)
    assert grad.ndim == 1


def test_split_mnist_across_nodes_iid_remainder() -> None:
    ds = mnist_module.load_mnist_or_fake(use_fake_data=True, n_samples=11, seed=0)
    subsets = mnist_module.split_mnist_across_nodes(ds, n_nodes=2, heterogeneity="iid", seed=0)
    assert len(subsets) == 2
    assert sum(len(s) for s in subsets) == len(ds)


def test_split_mnist_across_nodes_unknown_heterogeneity() -> None:
    ds = mnist_module.load_mnist_or_fake(use_fake_data=True, n_samples=6, seed=0)
    with pytest.raises(ValueError):
        mnist_module.split_mnist_across_nodes(ds, n_nodes=2, heterogeneity="unknown", seed=0)


def test_split_mnist_across_nodes_targets_branch() -> None:
    fixture = Path("tests/assets/mnist_tiny")
    train_ds, _ = mnist_module.load_mnist_tiny(fixture)
    subsets = mnist_module.split_mnist_across_nodes(
        train_ds, n_nodes=2, heterogeneity="label_skew", seed=0
    )
    assert len(subsets) == 2


def test_load_mnist_or_fake_subsets_with_patch(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_ds = mnist_module.datasets.FakeData(  # type: ignore[attr-defined]
        size=10,
        image_size=(1, 28, 28),
        num_classes=10,
        transform=mnist_module.transforms.ToTensor(),  # type: ignore[attr-defined]
    )

    monkeypatch.setattr(mnist_module, "load_mnist", lambda *args, **kwargs: fake_ds)
    ds = mnist_module.load_mnist_or_fake(
        use_fake_data=False,
        n_samples=5,
        seed=0,
        root=Path("."),
        train=True,
        download=False,
    )
    assert len(ds) == 5


def test_load_mnist_returns_dataset_with_stub(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    (tmp_path / "MNIST" / "raw").mkdir(parents=True, exist_ok=True)

    def _fake_mnist(*_args, **_kwargs):  # type: ignore[no-untyped-def]
        return mnist_module.datasets.FakeData(  # type: ignore[attr-defined]
            size=3,
            image_size=(1, 28, 28),
            num_classes=10,
            transform=mnist_module.transforms.ToTensor(),  # type: ignore[attr-defined]
        )

    monkeypatch.setattr(mnist_module.datasets, "MNIST", _fake_mnist)  # type: ignore[attr-defined]
    ds = mnist_module.load_mnist(tmp_path, train=True, download=False)
    assert len(ds) == 3
