# gossip-fw-bench

Benchmark framework for Frank-Wolfe, SGD, and distributed optimization.

## Quickstart

```bash
# Install dependencies
uv sync

# Run tests
uv run pytest -q
```

## Architecture

This repo has a **config-driven supervised training pipeline** and a **legacy step-based benchmark pipeline**.

### Core Concepts (Supervised Pipeline)

- **Task**: Owns datasets, splits, loss, and metrics. Exposes dataloaders `train_loader`, `val_loader`, `test_loader`.
  - Example: `tasks.mnist.MNISTSupervisedTask`
  - Provides `loss_fn(logits, labels)` and optional `metrics_fn(logits, labels)`.
- **Model**: The trainable agent (e.g., `torch.nn.Module`).
- **Optimizer**: Update rule for model parameters. Referenced by class path in JSON (no registry). Core optimizers live in `optim`: `muon.py`, `scion.py`, `adam.py` (legacy, no projection), `legacy_frankwolfe.py` (legacy Frank-Wolfe + constraints + torch SFW).
  - Each optimizer has a display `name` in config (e.g. `SGD(lr=0.1)` or `SFW-L2(r=auto*5,step=harmonic)`).
- **Trainer**: Executes batch/epoch loops on a task using a model and optimizer.
  - Example: `trainers.supervised.SupervisedTrainer`
  - Produces:
    - `history.jsonl` (per-epoch metrics)
    - `steps.jsonl` (per-iteration metrics, configurable frequencies)
- **Experiment**: Orchestrates tasks/models/optimizers and multiple runs, aggregates metrics, plots mean/std.
  - Example: `experiments.base_training.ExperimentBaseTraining`
- **Runner**: Thin CLI that reads JSON config and executes an experiment.
  - Example: `benchmarks.runner`

### Call Order (Supervised Pipeline)

1. **Runner** (`benchmarks.runner`) loads a config JSON.
2. **Experiment** (`ExperimentBaseTraining`) is instantiated from the config.
3. For each `task_dir`:
   - Task spec is resolved and instantiated (datasets are loaded, split, wrapped in dataloaders).
4. For each `model` and `optimizer`:
   - Model is created and initialized.
   - Optimizer is created.
   - Trainer runs `epochs` and logs:
     - `history.jsonl` (per epoch)
     - `steps.jsonl` (per iteration; `batch_loss`, `loss`, and optional metrics)
5. Experiment aggregates **mean/std across runs** and writes plots to:
   - `workflow/<exp>/plots/<task>/<model>/`
   - step-level plots in `.../steps/`

### Legacy Pipeline (Step-Based)

- **Environment**: Generic wrapper for agent–task interaction (used by legacy step-based benchmarks).
- **Task/Model/Optimizer/GradComputer**: Protocol-based components run inside environments.
- **Legacy Runner**: `benchmarks.legacy_runner` supports matrix mode, checks, gossip, and animations.

## Config-Based Experiments

Task directories contain JSON specs:

- `task.json` – task class + params
- `models.json` – list of model specs
- `optimizers.json` – list of optimizer specs

Experiment configs point at task directories and a trainer.

Example config: `configs/experiments/mnist_cnn.json`

```bash
python -m benchmarks.runner \
  --config configs/experiments/mnist_cnn.json
```

A convenience script is available:

```bash
bash scripts/bash/frankwolfe/run_mnist_cnn_fw.sh
```

This writes runs to `workflow/frankwolfe/exp_XXXX/`.

### Reusing an Experiment Directory

You can reuse a specific experiment id (and skip runs that already have results):

```bash
python -m benchmarks.runner \
  --config configs/experiments/mnist_cnn.json \
  --exp-id 13
```

To force re-run all existing runs:

```bash
python -m benchmarks.runner \
  --config configs/experiments/mnist_cnn.json \
  --exp-id 13 \
  --no-skip-existing
```

### Notes on MNIST splits

The default config requests 60k train, 10k val, 1k test. MNIST has 60k train + 10k test, so the task clamps validation size to keep test size intact. The effective split is recorded in each run summary.

## Legacy CLI

The original CLI (matrix mode, checks, gossip, animations) is preserved as:

```bash
python -m benchmarks.legacy_runner ...
```

## Project Structure

```
src/
├── core/                # Protocols and core types
├── tasks/               # Tasks and datasets (MNIST, logistic, quadratic)
├── models/              # Torch models and adapters
├── optim/               # Optimizers + constraints + legacy Frank-Wolfe
├── trainers/            # Training loops
├── experiments/         # Experiments + config utilities
├── environments/        # Environments (single-process, gossip)
├── distributed/         # Topologies, strategies, communicators
└── benchmarks/          # Runner, legacy runner, workflow utilities
```

## Development

```bash
uv run ruff check .
uv run ruff format .
uv run mypy src
uv run pytest -q
```
