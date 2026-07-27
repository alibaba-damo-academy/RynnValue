#!/bin/bash
# pi0.5 SFT on the RoboTwin lerobot dump.
#
# Usage:
#   bash scripts/train_pi05_robotwin.sh                                     # default task
#   bash scripts/train_pi05_robotwin.sh adjust_bottle-demo_clean_collect_200-50
#   bash scripts/train_pi05_robotwin.sh <task_dir> <exp_name> [extra train.py args...]
#
# Env (override as needed):
#   CONFIG       - registered TrainConfig (default: pi05_robotwin; use pi05_robotwin_lora for LoRA).
#   ROBOTWIN_ROOT- dataset root containing per-task lerobot subdirs.
#   GPUS         - which CUDA devices to use (default: 0,1).
#   FSDP_DEVICES - if >1, shard the model across this many devices (FSDP). Default: 1
#                  (pure data-parallel, each GPU holds a full model copy).
#   SKIP_NORM    - if set, skip the compute_norm_stats step (assumes stats are present).

set -euo pipefail

# ----- workspace (cd to repo root so the relative paths below resolve) -----
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"
echo -e "\033[32m[Info] Working directory: $PROJECT_ROOT\033[0m"

CONFIG="${CONFIG:-pi05_robotwin}"
ROBOTWIN_ROOT="${ROBOTWIN_ROOT:-/path/to/robotwin_lerobot_data/lerobot_robotwin_eef_clean_50}"
TASK_DIR="${1:-adjust_bottle-demo_clean_collect_200-50}"
EXP_NAME="${2:-${TASK_DIR}}"
GPUS="${GPUS:-0,1}"
FSDP_DEVICES="${FSDP_DEVICES:-1}"

REPO_ID="${ROBOTWIN_ROOT}/${TASK_DIR}"
if [[ ! -d "${REPO_ID}" ]]; then
    echo "[error] dataset directory not found: ${REPO_ID}" >&2
    exit 1
fi

echo "==> CONFIG       = ${CONFIG}"
echo "==> REPO_ID      = ${REPO_ID}"
echo "==> EXP_NAME     = ${EXP_NAME}"
echo "==> GPUS         = ${GPUS}"
echo "==> FSDP_DEVICES = ${FSDP_DEVICES}"

export CUDA_VISIBLE_DEVICES="${GPUS}"
export OPENPI_DATA_HOME=./openpi

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

# Drop the first two positional args ($1=task, $2=exp); pass the rest through to train.py.
shift $(( $# >= 2 ? 2 : $# ))

python scripts/train.py \
    "${CONFIG}" \
    --exp_name="${EXP_NAME}" \
    --data.repo_id="${REPO_ID}" \
    --fsdp_devices="${FSDP_DEVICES}" \
    --overwrite \
    "$@"
