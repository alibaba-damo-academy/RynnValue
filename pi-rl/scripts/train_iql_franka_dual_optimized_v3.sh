#!/bin/bash
# Optimized-v3 IQL training for pi0.5 on Franka (dual-arm).
#
# v3 differences vs. v2:
#   - Real joint + gripper state fed into the model (via pi05's
#     discrete_state_input=True path).
#   - Actions remain continuous absolute joint + gripper values.
#   - Norm stats are computed with the v3 SFT config so state/action
#     normalization matches training.

set -euo pipefail

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"
echo -e "\033[32m[Info] Working directory: $PROJECT_ROOT\033[0m"

PYTHON="${PYTHON:-python3}"
CONFIG="${CONFIG:-pi05_franka_dual_iql_optimized_v3}"
NORM_CONFIG="${NORM_CONFIG:-pi05_franka_dual_optimized_v3}"
REPO_ID="${REPO_ID:-/path/to/franka_lerobot_data_clean/pick_up_the_banana}"
EXP_NAME="${EXP_NAME:-iql_v3_$(basename "${REPO_ID}")}"
GPUS="${GPUS:-0,1}"
CKPT_BASE_DIR="${CKPT_BASE_DIR:-/path/to/checkpoints/iql_checkpoints_optimized_v3}"

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

NORM_STATS_FILE="${REPO_ID}/norm_stats.json"
if [[ -n "${SKIP_NORM:-}" || -f "${NORM_STATS_FILE}" ]]; then
    echo "==> norm stats present at ${NORM_STATS_FILE} (or SKIP_NORM set), skipping compute"
else
    echo "==> computing norm stats -> ${NORM_STATS_FILE} (using ${NORM_CONFIG})"
    JAX_PLATFORMS=cpu \
    XLA_PYTHON_CLIENT_PREALLOCATE=false \
        python scripts/compute_norm_stats_fast.py \
            --config-name "${NORM_CONFIG}" \
            --repo-id "${REPO_ID}" \
            ${EPISODE_FILTER_PATH:+--episode-filter-path "${EPISODE_FILTER_PATH}"}
fi

"${PYTHON}" scripts/train_iql.py \
    "${CONFIG}" \
    --exp_name="${EXP_NAME}" \
    --data.repo_id="${REPO_ID}" \
    --checkpoint_base_dir="${CKPT_BASE_DIR}" \
    ${EPISODE_FILTER_PATH:+--data.episode_filter_path="${EPISODE_FILTER_PATH}"} \
    --overwrite \
    "$@"
