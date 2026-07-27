#!/bin/bash
# Optimized IQL training for pi0.5 on Franka (single-arm).
#
# Optimizations vs. train_iql_franka.sh:
#   - 2 cameras (left_side + left_wrist), right cameras masked
#   - Gripper binarized to {0, 1}
#   - discrete_state_input=False (continuous state)
#   - 20k steps, lower LR, ema_decay=0.99
#   - IQL critic uses 2 cameras instead of 4
#
# Usage:
#   REPO_ID=/path/to/task bash scripts/train_iql_franka_optimized.sh
#   EXP_NAME=my_exp REPO_ID=/path/to/task bash scripts/train_iql_franka_optimized.sh

set -euo pipefail

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"
echo -e "\033[32m[Info] Working directory: $PROJECT_ROOT\033[0m"

PYTHON="${PYTHON:-python3}"
CONFIG="${CONFIG:-pi05_franka_single_iql_optimized}"
NORM_CONFIG="${NORM_CONFIG:-pi05_franka_single_optimized}"
REPO_ID="${REPO_ID:-/path/to/franka_lerobot_data/pick_up_the_bread}"
EXP_NAME="${EXP_NAME:-iql_opt_$(basename "${REPO_ID}")}"
GPUS="${GPUS:-0,1}"
CKPT_BASE_DIR="${CKPT_BASE_DIR:-/path/to/checkpoints/iql_checkpoints_optimized}"

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
    ${PROGRESS_REWARD_WEIGHT:+--data.progress_reward_weight="${PROGRESS_REWARD_WEIGHT}"} \
    ${TERMINAL_BONUS:+--data.terminal_bonus="${TERMINAL_BONUS}"} \
    --overwrite \
    "$@"
