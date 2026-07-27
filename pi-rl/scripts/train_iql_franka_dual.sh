#!/bin/bash
# IQL training for pi0.5 on Franka (dual-arm, no conda).
#
# Usage:
#   REPO_ID=/path/to/task bash scripts/train_iql_franka_dual.sh
#   EXP_NAME=my_exp REPO_ID=/path/to/task bash scripts/train_iql_franka_dual.sh
#
# Env (override as needed):
#   PYTHON         - python interpreter (default: python3).
#   CONFIG         - registered TrainConfig (default: pi05_franka_dual_iql).
#   REPO_ID        - absolute path to the lerobot dataset directory.
#   EXP_NAME       - wandb / checkpoint experiment name (defaults to iql_<basename(REPO_ID)>).
#   GPUS           - which CUDA devices to use (default: 0,1).
#   CKPT_BASE_DIR  - checkpoint root (default: /path/to/checkpoints/iql_checkpoints).
#   SKIP_NORM      - if set, skip the compute_norm_stats step (assumes stats are present).

set -euo pipefail

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"
echo -e "\033[32m[Info] Working directory: $PROJECT_ROOT\033[0m"

PYTHON="${PYTHON:-python3}"
CONFIG="${CONFIG:-pi05_franka_dual_iql}"
NORM_CONFIG="${NORM_CONFIG:-pi05_franka_dual}"
REPO_ID="${REPO_ID:-/path/to/franka_lerobot_data/pick_up_the_banana}"
EXP_NAME="${EXP_NAME:-iql_$(basename "${REPO_ID}")}"
GPUS="${GPUS:-0,1}"
CKPT_BASE_DIR="${CKPT_BASE_DIR:-/path/to/checkpoints/iql_checkpoints}"

if [[ ! -d "${REPO_ID}" ]]; then
    echo "[error] dataset directory not found: ${REPO_ID}" >&2
    exit 1
fi

echo "==> PYTHON        = ${PYTHON} ($(${PYTHON} --version 2>&1))"
echo "==> CONFIG        = ${CONFIG}"
echo "==> NORM_CONFIG   = ${NORM_CONFIG}"
echo "==> REPO_ID       = ${REPO_ID}"
echo "==> EXP_NAME      = ${EXP_NAME}"
echo "==> GPUS          = ${GPUS}"
echo "==> CKPT_BASE_DIR = ${CKPT_BASE_DIR}"

export CUDA_VISIBLE_DEVICES="${GPUS}"
export OPENPI_DATA_HOME=/path/to/openpi_data

export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_ALLOCATOR=platform

# Use NORM_CONFIG (SFT variant) -- the IQL transform chain expects 2-frame
# stacks that only the RL data loader provides; SFT yields identical
# state/action stats and writes to <REPO_ID>/norm_stats.json (same path).
NORM_STATS_FILE="${REPO_ID}/norm_stats.json"
if [[ -n "${SKIP_NORM:-}" || -f "${NORM_STATS_FILE}" ]]; then
    echo "==> norm stats present at ${NORM_STATS_FILE} (or SKIP_NORM set), skipping compute"
else
    echo "==> computing norm stats -> ${NORM_STATS_FILE} (using ${NORM_CONFIG})"
    JAX_PLATFORMS=cpu \
    XLA_PYTHON_CLIENT_PREALLOCATE=false \
        python scripts/compute_norm_stats_fast.py \
            --config-name "${NORM_CONFIG}" \
            --repo-id "${REPO_ID}"
fi

"${PYTHON}" scripts/train_iql.py \
    "${CONFIG}" \
    --exp_name="${EXP_NAME}" \
    --data.repo_id="${REPO_ID}" \
    --checkpoint_base_dir="${CKPT_BASE_DIR}" \
    --overwrite
