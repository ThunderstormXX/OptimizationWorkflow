#!/usr/bin/env bash
set -euo pipefail

DIST_CONFIG=configs/tuning/mnist_cnn_random/o_sgdm_dist.json
EXP_CONFIG=${EXP_CONFIG:-configs/experiments/mnist_cnn.json}
TASK_DIR=${TASK_DIR:-configs/tasks/mnist_cnn}
WORKFLOW_DIR=${WORKFLOW_DIR:-workflow/tuning_random}
EPOCHS=${EPOCHS:-10}
NUM_RUNS=${NUM_RUNS:-2}
BATCH_SIZE=${BATCH_SIZE:-128}
N_TRAIN=${N_TRAIN:-1000}
N_VAL=${N_VAL:-100}
N_TEST=${N_TEST:-100}
DATA_ROOT=${DATA_ROOT:-.data}
DEVICE=${DEVICE:-cpu}
SEED=${SEED:-0}
FULL_LOSS_EVERY=${FULL_LOSS_EVERY:-10}
BATCH_LOG_EVERY=${BATCH_LOG_EVERY:-1}

if [ -z "${PYTHONPATH:-}" ]; then
  export PYTHONPATH="src"
else
  export PYTHONPATH="${PYTHONPATH}:src"
fi

export PYTHONUNBUFFERED=1
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

"$PYTHON_BIN" scripts/tuning/run_random_search.py   --config "$DIST_CONFIG"   --exp-config "$EXP_CONFIG"   --task-dir "$TASK_DIR"   --workflow-dir "$WORKFLOW_DIR"   --epochs "$EPOCHS"   --num-runs "$NUM_RUNS"   --seed "$SEED"   --device "$DEVICE"   --batch-size "$BATCH_SIZE"   --train-size "$N_TRAIN"   --val-size "$N_VAL"   --test-size "$N_TEST"   --data-root "$DATA_ROOT"   --full-loss-every "$FULL_LOSS_EVERY"   --batch-log-every "$BATCH_LOG_EVERY"
