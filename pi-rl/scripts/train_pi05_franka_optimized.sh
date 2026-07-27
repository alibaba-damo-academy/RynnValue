#!/bin/bash
# pi0.5 optimized SFT on the single-arm Franka lerobot dump.
#
# Optimizations vs. train_pi05_franka.sh:
#   - 2 cameras (left_side + left_wrist), right cameras masked
#   - Gripper binarized to {0, 1}
#   - discrete_state_input=False (continuous state)
#   - 20k steps, peak_lr=5e-5, ema_decay=0.99
#
# Usage:
#   REPO_ID=/path/to/task bash scripts/train_pi05_franka_optimized.sh
#   EXP_NAME=my_exp REPO_ID=/path/to/task bash scripts/train_pi05_franka_optimized.sh

set -euo pipefail

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"
echo -e "\033[32m[Info] Working directory: $PROJECT_ROOT\033[0m"

CONFIG="${CONFIG:-pi05_franka_single_optimized}"
REPO_ID="${REPO_ID:-/path/to/franka_lerobot_data/pick_up_the_steak}"
EXP_NAME="${EXP_NAME:-$(basename "${REPO_ID}")}"
GPUS="${GPUS:-0,1}"
FSDP_DEVICES="${FSDP_DEVICES:-2}"
CKPT_BASE_DIR="${CKPT_BASE_DIR:-}"

if [[ ! -d "${REPO_ID}" ]]; then
    echo "[error] dataset directory not found: ${REPO_ID}" >&2
    exit 1
fi

echo "==> CONFIG       = ${CONFIG}"
echo "==> REPO_ID      = ${REPO_ID}"
echo "==> EXP_NAME     = ${EXP_NAME}"
echo "==> GPUS         = ${GPUS}"
echo "==> FSDP_DEVICES = ${FSDP_DEVICES}"
if [[ -n "${CKPT_BASE_DIR}" ]]; then
    echo "==> CKPT_BASE    = ${CKPT_BASE_DIR}"
fi

export CUDA_VISIBLE_DEVICES="${GPUS}"
export OPENPI_DATA_HOME=/path/to/openpi_data

NORM_STATS_FILE="${REPO_ID}/norm_stats.json"
if [[ -n "${SKIP_NORM:-}" || -f "${NORM_STATS_FILE}" ]]; then
    echo "==> norm stats present at ${NORM_STATS_FILE} (or SKIP_NORM set), skipping compute"
else
    echo "==> computing norm stats -> ${NORM_STATS_FILE}"
    JAX_PLATFORMS=cpu \
    XLA_PYTHON_CLIENT_PREALLOCATE=false \
        python scripts/compute_norm_stats_fast.py \
            --config-name "${CONFIG}" \
            --repo-id "${REPO_ID}"
fi

extra_args=()
if [[ -n "${CKPT_BASE_DIR}" ]]; then
    extra_args+=(--checkpoint_base_dir="${CKPT_BASE_DIR}")
fi

python scripts/train.py \
    "${CONFIG}" \
    --exp_name="${EXP_NAME}" \
    --data.repo_id="${REPO_ID}" \
    --fsdp_devices="${FSDP_DEVICES}" \
    --overwrite \
    "${extra_args[@]}" \
    "$@"
