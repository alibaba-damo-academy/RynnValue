#!/bin/bash
# pi0.5 SFT on the dual-arm Franka lerobot dump.
#
# Usage:
#   REPO_ID=/path/to/task bash scripts/train_pi05_franka_dual.sh
#   EXP_NAME=my_exp REPO_ID=/path/to/task bash scripts/train_pi05_franka_dual.sh
#
# Env (override as needed):
#   CONFIG       - registered TrainConfig. Default: pi05_franka_dual (14-dim arm + 2-dim
#                  gripper). The Franka data factory auto-detects mode from the lerobot
#                  metadata, but the model action_dim is fixed by the TrainConfig, so this
#                  script pins the dual-arm variant. Use train_pi05_franka.sh for single-arm.
#   REPO_ID      - absolute path to the lerobot dataset directory.
#   EXP_NAME     - experiment name (defaults to basename of REPO_ID).
#   GPUS         - which CUDA devices to use (default: 0,1).
#   FSDP_DEVICES - if >1, shard the model across this many devices (FSDP). Default: 2.
#   SKIP_NORM    - if set, skip the compute_norm_stats step (assumes stats are present).

set -euo pipefail

# ----- workspace (cd to repo root so the relative paths below resolve) -----
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"
echo -e "\033[32m[Info] Working directory: $PROJECT_ROOT\033[0m"

CONFIG="${CONFIG:-pi05_franka_dual}"
REPO_ID="${REPO_ID:-/path/to/franka_lerobot_data/pick_up_the_banana}"
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

# `compute_norm_stats.py` does `assets_dirs / repo_id`, but pathlib drops the left
# side when repo_id is absolute, so norm_stats.json actually lives at
#   <REPO_ID>/norm_stats.json
# That's also the path the runtime loader reads from -- both sides agree.
NORM_STATS_FILE="${REPO_ID}/norm_stats.json"
if [[ -n "${SKIP_NORM:-}" || -f "${NORM_STATS_FILE}" ]]; then
    echo "==> norm stats present at ${NORM_STATS_FILE} (or SKIP_NORM set), skipping compute"
else
    echo "==> computing norm stats -> ${NORM_STATS_FILE}"
    # Force JAX to CPU + disable XLA preallocation: norm_stats is pure-numpy CPU work,
    # but importing openpi pulls in JAX which would otherwise grab 75% of every visible GPU
    # (per worker process under spawn-mode DataLoader). These two env vars only affect this
    # subshell, so the train step below still gets full GPU access.
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
