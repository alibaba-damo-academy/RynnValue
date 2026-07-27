#!/bin/bash
# IQL training for pi0.5 on LIBERO.
#
# Usage:
#   bash scripts/train_iql_libero.sh
#   bash scripts/train_iql_libero.sh [extra train_iql.py args...]
#
# Env (override as needed):
#   CONDA_ENV  - conda env to activate (default: pi-rl).
#   CONDA_ROOT - conda installation root (default: /opt/conda).
#   CONFIG     - registered TrainConfig (default: pi05_libero_iql).
#   REPO_ID    - dataset path (default: from config).
#   GPUS       - which CUDA devices to use (default: 0,1).
#   EXP_NAME   - wandb experiment name.

# `set -u` deferred until after conda activate (conda's internal bookkeeping trips on unset vars).
set -eo pipefail

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"
echo -e "\033[32m[Info] Working directory: $PROJECT_ROOT\033[0m"

# ----- conda activation -----
CONDA_ROOT="${CONDA_ROOT:-/opt/conda}"
CONDA_ENV="${CONDA_ENV:-pi-rl}"
if [[ -f "${CONDA_ROOT}/etc/profile.d/conda.sh" ]]; then
    # shellcheck disable=SC1091
    source "${CONDA_ROOT}/etc/profile.d/conda.sh"
else
    echo "[error] conda.sh not found at ${CONDA_ROOT}/etc/profile.d/conda.sh; set CONDA_ROOT" >&2
    exit 1
fi
conda activate "${CONDA_ENV}"
echo "==> CONDA_ENV = ${CONDA_ENV} ($(which python))"

set -u

CONFIG="${CONFIG:-pi05_libero_iql}"
GPUS="${GPUS:-0,1}"
EXP_NAME="${EXP_NAME:-iql_libero}"

echo "==> CONFIG   = ${CONFIG}"
echo "==> GPUS     = ${GPUS}"
echo "==> EXP_NAME = ${EXP_NAME}"

export CUDA_VISIBLE_DEVICES="${GPUS}"
export OPENPI_DATA_HOME=./openpi

EXTRA_ARGS=()
if [[ -n "${REPO_ID:-}" ]]; then
    echo "==> REPO_ID  = ${REPO_ID}"
    EXTRA_ARGS+=(--data.repo_id="${REPO_ID}")
fi

python scripts/train_iql.py \
    "${CONFIG}" \
    --exp_name="${EXP_NAME}" \
    --overwrite \
    "${EXTRA_ARGS[@]}" \
    "$@"
