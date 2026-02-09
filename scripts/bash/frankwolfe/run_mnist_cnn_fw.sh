#!/usr/bin/env bash
set -euo pipefail

CONFIG=${CONFIG:-configs/experiments/mnist_cnn.json}
WORKFLOW_DIR=${WORKFLOW_DIR:-workflow/frankwolfe}
EPOCHS=${EPOCHS:-10}
NUM_RUNS=${NUM_RUNS:-1}
BATCH_SIZE=${BATCH_SIZE:-128}
N_TRAIN=${N_TRAIN:-1000}
N_VAL=${N_VAL:-100}
N_TEST=${N_TEST:-100}
DATA_ROOT=${DATA_ROOT:-.data}
DEVICE=${DEVICE:-auto}
SEED=${SEED:-0}
FULL_LOSS_EVERY=${FULL_LOSS_EVERY:-10}
BATCH_LOG_EVERY=${BATCH_LOG_EVERY:-1}
FULL_METRICS_EVERY=${FULL_METRICS_EVERY:-}
BATCH_METRICS_EVERY=${BATCH_METRICS_EVERY:-}
EXP_ID=${EXP_ID:-}
NO_SKIP_EXISTING=${NO_SKIP_EXISTING:-0}

if [ -z "${PYTHONPATH:-}" ]; then
  export PYTHONPATH="src"
else
  export PYTHONPATH="${PYTHONPATH}:src"
fi

export PYTHONUNBUFFERED=1
export TQDM_MININTERVAL=0.1
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-1}
export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-1}
export VECLIB_MAXIMUM_THREADS=${VECLIB_MAXIMUM_THREADS:-1}
export NUMEXPR_NUM_THREADS=${NUMEXPR_NUM_THREADS:-1}
export KMP_SHM_DISABLE=${KMP_SHM_DISABLE:-1}
export KMP_DUPLICATE_LIB_OK=${KMP_DUPLICATE_LIB_OK:-TRUE}
export MPLCONFIGDIR=${MPLCONFIGDIR:-/tmp/matplotlib}
export SCION_COMPILE=${SCION_COMPILE:-0}

PYTHON_BIN=${PYTHON_BIN:-}
if [ -z "$PYTHON_BIN" ] && [ -x ".venv/bin/python" ]; then
  PYTHON_BIN=".venv/bin/python"
fi
if [ -z "$PYTHON_BIN" ]; then
  PYTHON_BIN="python"
fi

if ! "$PYTHON_BIN" - <<'PY'
import torch
import torchvision
import tqdm
print(f"torch={torch.__version__} torchvision={torchvision.__version__}")
print(f"tqdm={tqdm.__version__}")
PY
then
  echo "PyTorch, torchvision, and tqdm are required. Install dependencies (e.g., uv sync) and retry."
  exit 1
fi

echo "Starting MNIST experiment from config: $CONFIG"
echo "WorkflowDir=$WORKFLOW_DIR Epochs=$EPOCHS Runs=$NUM_RUNS BatchSize=$BATCH_SIZE"
echo "Train=$N_TRAIN Val=$N_VAL Test=$N_TEST DataRoot=$DATA_ROOT Device=$DEVICE FullLossEvery=$FULL_LOSS_EVERY BatchLogEvery=$BATCH_LOG_EVERY"

declare -a EXTRA_SETS=()
if [ -n "$FULL_METRICS_EVERY" ]; then
  EXTRA_SETS+=(--set "experiment.params.trainer.params.full_metrics_every=$FULL_METRICS_EVERY")
fi
if [ -n "$BATCH_METRICS_EVERY" ]; then
  EXTRA_SETS+=(--set "experiment.params.trainer.params.batch_metrics_every=$BATCH_METRICS_EVERY")
fi

CMD=(
  "$PYTHON_BIN" -m benchmarks.runner
  --config "$CONFIG"
  --workflow-dir "$WORKFLOW_DIR"
  --set "experiment.params.seed=$SEED"
  --set "experiment.params.num_runs=$NUM_RUNS"
  --set "experiment.params.trainer.params.epochs=$EPOCHS"
  --set "experiment.params.trainer.params.full_loss_every=$FULL_LOSS_EVERY"
  --set "experiment.params.trainer.params.batch_log_every=$BATCH_LOG_EVERY"
  --set "experiment.params.trainer.params.device=\"$DEVICE\""
  --set "experiment.params.task_overrides.data_root=\"$DATA_ROOT\""
  --set "experiment.params.task_overrides.batch_size=$BATCH_SIZE"
  --set "experiment.params.task_overrides.train_size=$N_TRAIN"
  --set "experiment.params.task_overrides.val_size=$N_VAL"
  --set "experiment.params.task_overrides.test_size=$N_TEST"
  --set "experiment.params.task_overrides.device=\"$DEVICE\""
  --set "experiment.params.task_overrides.verbose=true"
)

if [ -n "$EXP_ID" ]; then
  CMD+=(--exp-id "$EXP_ID")
fi
if [ "$NO_SKIP_EXISTING" = "1" ]; then
  CMD+=(--no-skip-existing)
fi

if [ "${#EXTRA_SETS[@]}" -gt 0 ]; then
  CMD+=("${EXTRA_SETS[@]}")
fi

"${CMD[@]}"
