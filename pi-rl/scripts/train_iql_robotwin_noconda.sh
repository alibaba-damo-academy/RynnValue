#!/bin/bash
# IQL training for pi0.5 on RoboTwin (no conda).
#
# Usage:
#   bash scripts/train_iql_robotwin_noconda.sh
#   bash scripts/train_iql_robotwin_noconda.sh [extra train_iql.py args...]
#
# Env (override as needed):
#   PYTHON     - python interpreter (default: python3).
#   CONFIG     - registered TrainConfig (default: pi05_robotwin_iql).
#   REPO_ID    - dataset path (default: from config).
#   GPUS       - which CUDA devices to use (default: 0,1).
#   EXP_NAME   - wandb experiment name.

set -euo pipefail

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"
echo -e "\033[32m[Info] Working directory: $PROJECT_ROOT\033[0m"

PYTHON="${PYTHON:-python3}"
CONFIG="${CONFIG:-pi05_robotwin_iql}"
GPUS="${GPUS:-0,1}"
EXP_NAME="${EXP_NAME:-iql_robotwin_beat_block_hammer}"
CKPT_BASE_DIR="${CKPT_BASE_DIR:-/path/to/checkpoints/iql_checkpoints}"

echo "==> PYTHON        = ${PYTHON} ($(${PYTHON} --version 2>&1))"
echo "==> CONFIG        = ${CONFIG}"
echo "==> GPUS          = ${GPUS}"
echo "==> EXP_NAME      = ${EXP_NAME}"
echo "==> CKPT_BASE_DIR = ${CKPT_BASE_DIR}"

export CUDA_VISIBLE_DEVICES="${GPUS}"
export OPENPI_DATA_HOME=/path/to/openpi_data

# JAX GPU memory: disable preallocation so JAX doesn't grab 75% of each GPU
# up-front (which can cause cuSolver handle creation to fail with OOM).
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_ALLOCATOR=platform

EXTRA_ARGS=()
if [[ -n "${REPO_ID:-}" ]]; then
    echo "==> REPO_ID  = ${REPO_ID}"
    EXTRA_ARGS+=(--data.repo_id="${REPO_ID}")
fi

"${PYTHON}" scripts/train_iql.py \
    "${CONFIG}" \
    --exp_name="${EXP_NAME}" \
    --checkpoint_base_dir="${CKPT_BASE_DIR}" \
    --overwrite \
    "${EXTRA_ARGS[@]}" \
    "$@"
